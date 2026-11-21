"""Score one or more exported models on the held-out test split of a dataset ZIP.

Uses each model's labels.txt; models whose labels differ from the ZIP's folders are scored on the
classes they share. 96x96 models get the same bilinear upsampling used in training.

Example:
    python training/evaluate.py ferplus4.zip openmv/mvp4 openmv/mnv2
"""

import argparse
import io
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import tensorflow as tf
from PIL import Image


def load_test(archive):
    items = []
    with zipfile.ZipFile(archive) as source:
        for name in sorted(source.namelist()):
            parts = PurePosixPath(name).parts
            if len(parts) >= 3 and parts[-3] == "test" and name.lower().endswith((".jpg", ".jpeg", ".png")):
                with Image.open(io.BytesIO(source.read(name))) as image:
                    items.append((parts[-2], np.asarray(image.convert("L").resize((48, 48), Image.Resampling.BILINEAR))))
    return items


def evaluate(folder, items):
    labels = [line.strip() for line in (folder / "labels.txt").read_text().splitlines() if line.strip()]
    interpreter = tf.lite.Interpreter(model_path=str(folder / "expression.tflite"), num_threads=4)
    interpreter.allocate_tensors()
    inp, = interpreter.get_input_details()
    out, = interpreter.get_output_details()
    side = int(inp["shape"][1])

    def scores(gray):
        if side != 48:
            gray = np.clip(np.round(tf.image.resize(gray[None, :, :, None].astype(np.float32), (side, side),
                                                    method="bilinear").numpy()[0, :, :, 0]), 0, 255).astype(np.uint8)
        interpreter.set_tensor(inp["index"], (gray.astype(np.int16) - 128).astype(np.int8)[None, :, :, None])
        interpreter.invoke()
        return interpreter.get_tensor(out["index"])[0].astype(np.int32)

    shared = [(name, gray) for name, gray in items if name in labels]
    plain = np.zeros((len(labels), len(labels)), dtype=int)
    flipped = np.zeros_like(plain)
    for name, gray in shared:
        a = scores(gray)
        truth = labels.index(name)
        plain[truth, int(np.argmax(a))] += 1
        flipped[truth, int(np.argmax(a + scores(gray[:, ::-1])))] += 1
    recall = flipped.diagonal() / np.maximum(flipped.sum(axis=1), 1)
    return {
        "model": str(folder), "side": side, "images": len(shared),
        "accuracy": float(np.trace(plain) / plain.sum()),
        "flip_accuracy": float(np.trace(flipped) / flipped.sum()),
        "balanced_flip_accuracy": float(recall[flipped.sum(axis=1) > 0].mean()),
        "flip_recall": {label: round(float(r), 3) for label, r in zip(labels, recall)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("models", type=Path, nargs="+", help="Folders containing expression.tflite and labels.txt")
    args = parser.parse_args()
    items = load_test(args.dataset)
    for folder in args.models:
        r = evaluate(folder, items)
        print(f"{r['model']:<28} {r['side']}px  n={r['images']:<5} acc {r['accuracy']:.3f}  flip {r['flip_accuracy']:.3f}  "
              f"balanced {r['balanced_flip_accuracy']:.3f}  {r['flip_recall']}", flush=True)


if __name__ == "__main__":
    main()
