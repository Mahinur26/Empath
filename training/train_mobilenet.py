"""Fine-tune ImageNet MobileNetV2 (alpha 0.35, 96x96, grayscale) on FER-2013 for the Nicla Vision.

This is the configuration the team planned in Edge Impulse Studio, trained locally instead. The
pretrained backbone has already learned general visual features from ImageNet, which usually
transfers to live camera faces better than a CNN trained from scratch on 48x48 FER images.

The ImageNet weights expect RGB in [-1, 1]. The first convolution is folded so the model takes one
grayscale channel in [0, 1] directly, which keeps the TFLite input identical to the other models:
INT8 with scale 1/255 and zero point -128.

Example:
    python training/train_mobilenet.py archive.zip --labels angry,happy,neutral,sad --out openmv/mnv2
"""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import tensorflow as tf

from train import SEED, load_images, split_training

SIDE = 96
ALPHA = 0.35


def grayscale_backbone():
    rgb = tf.keras.applications.MobileNetV2(input_shape=(SIDE, SIDE, 3), alpha=ALPHA,
                                            include_top=False, weights="imagenet")
    gray = tf.keras.applications.MobileNetV2(input_shape=(SIDE, SIDE, 1), alpha=ALPHA,
                                             include_top=False, weights=None)
    folded = None
    for source, target in zip(rgb.layers, gray.layers):
        if isinstance(source, tf.keras.layers.InputLayer):
            continue
        if source.name != target.name:
            raise RuntimeError(f"Layer mismatch: {source.name} vs {target.name}")
        weights = source.get_weights()
        if not weights:
            continue
        if source.name == "Conv1":
            # Gray replicated to RGB and mapped x -> 2x - 1:
            # conv(2x - 1) = conv(x; 2 * sum_rgb(k)) - sum(k), so scale the summed kernel by 2 ...
            kernel = weights[0]
            folded = kernel.sum(axis=(0, 1, 2))
            target.set_weights([kernel.sum(axis=2, keepdims=True) * 2.0])
        elif source.name == "bn_Conv1":
            # ... and absorb the constant "- sum(k)" into the batch-norm mean that follows.
            gamma, beta, mean, variance = weights
            target.set_weights([gamma, beta, mean + folded, variance])
        else:
            target.set_weights(weights)
    return gray


def build_model(class_count, train_bn=False):
    layers = tf.keras.layers
    inputs = layers.Input((SIDE, SIDE, 1), name="grayscale")
    backbone = grayscale_backbone()
    # Frozen batch-norm statistics are the usual fine-tuning default; upsampled grey faces are far from
    # ImageNet, so --train-bn lets the statistics adapt during the full fine-tune.
    features = backbone(inputs, training=None if train_bn else False)
    features = layers.GlobalAveragePooling2D()(features)
    features = layers.Dropout(0.2)(features)
    scores = layers.Dense(class_count, activation="softmax", name="expression")(features)
    return tf.keras.Model(inputs, scores), backbone


def batches(images, labels, training=False):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        dataset = dataset.shuffle(len(labels), seed=SEED)
    dataset = dataset.batch(64)
    geometry = tf.keras.Sequential([
        tf.keras.layers.RandomTranslation(0.08, 0.08, fill_mode="nearest", seed=SEED),
        tf.keras.layers.RandomRotation(0.04, fill_mode="nearest", seed=SEED),
        tf.keras.layers.RandomZoom(0.1, fill_mode="nearest", seed=SEED),
    ])

    def prepare(pixels, classes):
        pixels = tf.image.resize(tf.cast(pixels, tf.float32) / 255.0, (SIDE, SIDE), method="bilinear")
        if training:
            pixels = tf.image.random_flip_left_right(pixels)
            pixels = tf.image.random_brightness(pixels, 0.10)
            pixels = tf.image.random_contrast(pixels, 0.75, 1.25)
            pixels = geometry(pixels, training=True)
        return tf.clip_by_value(pixels, 0.0, 1.0), classes

    return dataset.map(prepare, num_parallel_calls=4).prefetch(2)


def upsample(images):
    """48x48 uint8 -> 96x96 uint8, the same bilinear resize used in training."""
    resized = tf.image.resize(tf.cast(images, tf.float32), (SIDE, SIDE), method="bilinear")
    return np.clip(np.round(resized.numpy()), 0, 255).astype(np.uint8)


def export_int8(model, images, rng):
    def representative_data():
        limits = np.zeros((1, SIDE, SIDE, 1), dtype=np.float32)
        limits[0, SIDE // 2:] = 1.0
        yield [limits]
        for index in rng.choice(len(images), size=min(256, len(images)), replace=False):
            yield [upsample(images[index:index + 1]).astype(np.float32) / 255.0]

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


def evaluate_int8(model, images, labels, names):
    interpreter = tf.lite.Interpreter(model_content=model, num_threads=4)
    interpreter.allocate_tensors()
    input_tensor, = interpreter.get_input_details()
    output_tensor, = interpreter.get_output_details()
    scale, zero = input_tensor["quantization"]
    if input_tensor["dtype"] != np.int8 or not np.isclose(scale, 1 / 255) or zero != -128:
        raise RuntimeError("Expected the INT8 input to map pixel 0 to -128 and 255 to 127")

    def scores(image):
        interpreter.set_tensor(input_tensor["index"], (image[None].astype(np.int16) - 128).astype(np.int8))
        interpreter.invoke()
        return interpreter.get_tensor(output_tensor["index"])[0].astype(np.int32)

    confusion = np.zeros((len(names), len(names)), dtype=int)
    flipped = np.zeros_like(confusion)
    big = upsample(images)
    for image, label in zip(big, labels):
        plain = scores(image)
        confusion[label, int(np.argmax(plain))] += 1
        flipped[label, int(np.argmax(plain + scores(image[:, ::-1])))] += 1
    return {
        "labels_in_model_order": names,
        "input_side": SIDE,
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path, help="Path to the FER-2013 ZIP from Kaggle")
    parser.add_argument("--labels", default="angry,happy,neutral,sad")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30, help="Full fine-tuning epochs after the head")
    parser.add_argument("--train-bn", action="store_true", help="Update batch-norm statistics while fine-tuning")
    parser.add_argument("--lr", type=float, default=1e-4, help="Fine-tuning learning rate")
    args = parser.parse_args()
    names = tuple(label.strip() for label in args.labels.split(",") if label.strip())

    tf.keras.utils.set_random_seed(SEED)
    rng = np.random.default_rng(SEED)
    data = load_images(args.dataset, names)
    images, labels = data["train"]
    train, validation = split_training(labels, rng, len(names))
    counts = np.bincount(labels[train], minlength=len(names))
    class_weight = {i: float(len(train) / (len(names) * c)) for i, c in enumerate(counts)}
    print(f"Training {names} on {len(train)} images, validating on {len(validation)}", flush=True)

    model, backbone = build_model(len(names), args.train_bn)
    train_data = batches(images[train], labels[train], training=True)
    val_data = batches(images[validation], labels[validation])

    backbone.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_data, validation_data=val_data, epochs=args.head_epochs, class_weight=class_weight, verbose=2)

    backbone.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(
        train_data, validation_data=val_data, epochs=args.epochs, class_weight=class_weight, verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
                   tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3)])

    float_accuracy = float(model.evaluate(batches(*data["test"]), verbose=0)[1])   # before INT8, to see quantization loss
    print(f"Float test accuracy: {float_accuracy:.4f}", flush=True)
    lite_model = export_int8(model, images[train], rng)
    report = evaluate_int8(lite_model, *data["test"], names)
    report.update(training_images=len(train), validation_images=len(validation), head_epochs=args.head_epochs,
                  finetune_epochs_run=len(history.epoch), seed=SEED, float_test_accuracy=float_accuracy,
                  train_bn=bool(args.train_bn), finetune_lr=args.lr,
                  architecture=f"MobileNetV2 alpha {ALPHA} {SIDE}x{SIDE}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "expression.tflite").write_bytes(lite_model)
    (args.out / "labels.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    (args.out / "model_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
