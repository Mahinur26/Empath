"""Train and measure the four-class Nicla Vision model from the Kaggle ZIP."""

import argparse
import io
import json
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf


LABELS = ("angry", "happy", "sad", "surprise")
SIDE = 48
SEED = 761
BOARD_DIR = Path(__file__).resolve().parents[1] / "openmv"


def load_images(archive, labels_wanted=LABELS, equalize=False):
    counts = {"train": [], "test": []}
    with ZipFile(archive) as source:
        for name in sorted(source.namelist()):
            parts = PurePosixPath(name).parts
            if len(parts) < 3 or not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            split, label, _ = parts[-3:]
            if split in counts and label in labels_wanted:
                counts[split].append((name, labels_wanted.index(label)))

        result = {}
        for split, files in counts.items():
            images = np.empty((len(files), SIDE, SIDE, 1), dtype=np.uint8)
            labels = np.empty(len(files), dtype=np.int32)
            for index, (name, label) in enumerate(files):
                with Image.open(io.BytesIO(source.read(name))) as image:
                    gray = image.convert("L").resize((SIDE, SIDE), Image.Resampling.BILINEAR)
                    # Histogram equalisation evens out lighting; web/camera.html applies the same PIL algorithm.
                    images[index, :, :, 0] = ImageOps.equalize(gray) if equalize else gray
                labels[index] = label
            if set(labels) != set(range(len(labels_wanted))):
                raise ValueError(f"Missing categories in the {split} split")
            result[split] = images, labels
            result[split + "_names"] = [name for name, _ in files]
    return result


def split_training(labels, rng, class_count=len(LABELS)):
    train, validation = [], []
    for label in range(class_count):
        indices = rng.permutation(np.flatnonzero(labels == label))
        cutoff = max(1, len(indices) // 10)
        train.extend(indices[cutoff:])
        validation.extend(indices[:cutoff])
    return rng.permutation(train), rng.permutation(validation)


def build_model(class_count=len(LABELS), arch="small"):
    layers = tf.keras.layers
    inputs = layers.Input((SIDE, SIDE, 1), name="grayscale")
    features = inputs
    # "small": one conv per block (the original ~58 KB model).
    # "wide": two convs in the first three blocks and more filters (~150 KB INT8). Uses the same
    # operator types (CONV_2D, MAX_POOL_2D, RESHAPE, FULLY_CONNECTED, SOFTMAX) so it runs on OpenMV
    # and in web/insideout-model.js unchanged.
    blocks = ((16, 1), (32, 1), (48, 1), (64, 1)) if arch == "small" else ((16, 2), (32, 2), (64, 2), (128, 1))
    for filters, repeats in blocks:
        for _ in range(repeats):
            features = layers.Conv2D(filters, 3, padding="same", use_bias=False)(features)
            features = layers.BatchNormalization()(features)
            features = layers.ReLU()(features)
        features = layers.MaxPooling2D()(features)
    features = layers.Flatten()(features)
    features = layers.Dropout(0.25 if arch == "small" else 0.4)(features)
    scores = layers.Dense(class_count, activation="softmax", name="expression")(features)
    model = tf.keras.Model(inputs, scores)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def cutout(pixels, lower_face=0.35, patch=0.5):
    """Hide parts of each face so the model cannot lean on one region (FER's angry = open mouth, sad = covered
    face shortcuts). With probability `lower_face` the mouth area is greyed out; with probability `patch` a
    random 8-24 px rectangle is. Hidden pixels take the image's mean grey."""
    b = tf.shape(pixels)[0]
    rows = tf.range(SIDE, dtype=tf.float32)[None, :, None, None]
    cols = tf.range(SIDE, dtype=tf.float32)[None, None, :, None]
    u = lambda lo, hi: tf.random.uniform((b, 1, 1, 1), lo, hi)
    box = (tf.abs(rows - u(0, SIDE)) < u(4, 12)) & (tf.abs(cols - u(0, SIDE)) < u(4, 12)) & (u(0, 1) < patch)
    lower = (rows >= u(27, 35)) & (u(0, 1) < lower_face)
    mean = tf.reduce_mean(pixels, axis=[1, 2, 3], keepdims=True)
    return tf.where(box | lower, mean * tf.ones_like(pixels), pixels)


def batches(images, labels, training=False, strong=False, hide=False):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        dataset = dataset.shuffle(len(labels), seed=SEED)
    dataset = dataset.batch(128)
    # Stronger augmentation: small shifts, rotations and zoom mimic an imperfect face crop from a live camera.
    geometry = tf.keras.Sequential([
        tf.keras.layers.RandomTranslation(0.08, 0.08, fill_mode="nearest", seed=SEED),
        tf.keras.layers.RandomRotation(0.04, fill_mode="nearest", seed=SEED),
        tf.keras.layers.RandomZoom(0.1, fill_mode="nearest", seed=SEED),
    ]) if strong else None

    def prepare(pixels, classes):
        pixels = tf.cast(pixels, tf.float32) / 255.0
        if training:
            pixels = tf.image.random_flip_left_right(pixels)
            pixels = tf.image.random_brightness(pixels, 0.10)
            if strong:
                pixels = tf.image.random_contrast(pixels, 0.75, 1.25)
                pixels = geometry(pixels, training=True)
            if hide:
                pixels = cutout(pixels)
            pixels = tf.clip_by_value(pixels, 0.0, 1.0)
        return pixels, classes

    return dataset.map(prepare, num_parallel_calls=2).prefetch(1)


def export_int8(model, images, rng):
    def representative_data():
        limits = np.zeros((1, SIDE, SIDE, 1), dtype=np.float32)
        limits[0, SIDE // 2:] = 1.0
        yield [limits]
        for index in rng.choice(len(images), size=min(256, len(images)), replace=False):
            yield [images[index:index + 1].astype(np.float32) / 255.0]

    # A fixed batch size lets the converter resolve shape operations at export.
    signature = [tf.TensorSpec((1, SIDE, SIDE, 1), tf.float32, name="grayscale")]
    with TemporaryDirectory() as exported:
        model.export(exported, input_signature=signature, verbose=False)
        converter = tf.lite.TFLiteConverter.from_saved_model(exported)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        # OpenMV firmware 5.0.1 rejects per-channel fully connected layers ("Failed to allocate tensors").
        converter._experimental_disable_per_channel_quantization_for_dense_layers = True
        return converter.convert()


def evaluate_int8(model, images, labels, names=LABELS):
    interpreter = tf.lite.Interpreter(model_content=model, num_threads=2)
    interpreter.allocate_tensors()
    input_tensor, = interpreter.get_input_details()
    output_tensor, = interpreter.get_output_details()
    scale, zero = input_tensor["quantization"]
    if input_tensor["dtype"] != np.int8 or not np.isclose(scale, 1 / 255) or zero != -128:
        raise RuntimeError("OpenMV expects the INT8 input to map pixel 0 to -128 and 255 to 127")
    if output_tensor["dtype"] != np.int8 or tuple(output_tensor["shape"]) != (1, len(names)):
        raise RuntimeError("Unexpected classifier output tensor")

    def scores(image):
        interpreter.set_tensor(input_tensor["index"], (image[None].astype(np.int16) - 128).astype(np.int8))
        interpreter.invoke()
        return interpreter.get_tensor(output_tensor["index"])[0].astype(np.int32)

    confusion = np.zeros((len(names), len(names)), dtype=int)
    flipped = np.zeros_like(confusion)   # average of the image and its mirror image (test-time flip)
    for image, label in zip(images, labels):
        plain = scores(image)
        confusion[label, int(np.argmax(plain))] += 1
        flipped[label, int(np.argmax(plain + scores(image[:, ::-1])))] += 1
    return {
        "labels_in_model_order": names,
        "test_images": int(confusion.sum()),
        "accuracy": float(np.trace(confusion) / confusion.sum()),
        "recall_by_label": dict(zip(names, (confusion.diagonal() / confusion.sum(axis=1)).tolist())),
        "confusion_matrix_true_rows_predicted_columns": confusion.tolist(),
        "flip_average_accuracy": float(np.trace(flipped) / flipped.sum()),
        "flip_average_recall_by_label": dict(zip(names, (flipped.diagonal() / flipped.sum(axis=1)).tolist())),
        "flip_average_confusion_matrix": flipped.tolist(),
        "tflite_bytes": len(model),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Path to the FER-2013 ZIP from Kaggle")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--out", type=Path, default=BOARD_DIR)
    parser.add_argument("--labels", default=",".join(LABELS),
                        help="Comma-separated FER-2013 folder names, in model output order")
    parser.add_argument("--balance", action="store_true",
                        help="Weight classes inversely to their training counts")
    parser.add_argument("--metrics", type=Path, default=None,
                        help="Metrics JSON path (default: model_metrics.json beside the --out folder)")
    parser.add_argument("--arch", choices=("small", "wide"), default="small",
                        help="small = original ~58 KB CNN; wide = two convs per block, ~150 KB")
    parser.add_argument("--augment", action="store_true",
                        help="Add shift, rotation, zoom and contrast augmentation")
    parser.add_argument("--cutout", action="store_true",
                        help="Randomly hide the lower face or a patch while training, so the model also uses brows and eyes")
    parser.add_argument("--equalize", action="store_true",
                        help="Histogram-equalise every 48x48 face (train and test); the camera page and firmware must do the same")
    parser.add_argument("--soft", type=Path, default=None,
                        help="JSON {path in ZIP: [vote share per label, in --labels order]} for soft-label training (FER+ votes)")
    args = parser.parse_args()
    names = tuple(label.strip() for label in args.labels.split(",") if label.strip())

    tf.keras.utils.set_random_seed(SEED)
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    rng = np.random.default_rng(SEED)
    data = load_images(args.dataset, names, equalize=args.equalize)
    images, labels = data["train"]
    train, validation = split_training(labels, rng, len(names))
    print(f"Training {names} on {len(train)} images, validating on {len(validation)}", flush=True)
    class_weight = None
    if args.balance:
        counts = np.bincount(labels[train], minlength=len(names))
        class_weight = {index: float(len(train) / (len(names) * count))
                        for index, count in enumerate(counts)}
        print("Class weights:", {names[i]: round(w, 3) for i, w in class_weight.items()}, flush=True)
    model = build_model(len(names), args.arch)
    train_targets, val_targets = labels[train], labels[validation]
    if args.soft:
        # Soft labels: train against each face's annotator vote shares instead of a single winning class.
        shares = json.loads(args.soft.read_text())
        targets = np.array([shares[name] for name in data["train_names"]], dtype=np.float32)
        if targets.shape[1] != len(names):
            raise ValueError("Soft-label vectors must have one entry per label, in --labels order")
        train_targets, val_targets = targets[train], targets[validation]
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="categorical_crossentropy", metrics=["accuracy"])
    # Augmented training is noisier epoch to epoch, so give it more patience before cutting the rate or stopping.
    stop_patience, lr_patience = (10, 4) if args.augment else (5, 2)
    history = model.fit(
        batches(images[train], train_targets, training=True, strong=args.augment, hide=args.cutout),
        validation_data=batches(images[validation], val_targets),
        epochs=args.epochs,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=stop_patience,
                                                    restore_best_weights=True),
                   tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=lr_patience)],
        class_weight=class_weight,
        shuffle=False,
        verbose=2,
    )
    lite_model = export_int8(model, images[train], rng)
    report = evaluate_int8(lite_model, *data["test"], names)
    report.update(training_images=len(train), validation_images=len(validation),
                  epochs_run=len(history.epoch), seed=SEED, class_weighted=bool(args.balance), soft_labels=bool(args.soft),
                  equalize=bool(args.equalize), cutout=bool(args.cutout),
                  arch=args.arch, augmented=bool(args.augment))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "expression.tflite").write_bytes(lite_model)
    (args.out / "labels.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    report_path = args.metrics or args.out.parent / "model_metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
