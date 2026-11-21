"""Tune per-class sensitivity so every expression is detected about equally often.

Training data is imbalanced (FER+ has roughly 3x more happy and neutral faces than sad), so a model is
cautious about calling the rare classes. This multiplies each class's score by a factor before picking the
winner, choosing the factors that maximise balanced accuracy (the mean of per-class recall).

The factors are tuned on the FER-2013 PublicTest half of the test set and reported on the PrivateTest half,
which the tuning never sees. They are saved as "sensitivity" in the model's model_metrics.json, where the
camera page (and firmware) can apply them.

Example:
    python training/tune_sensitivity.py ferplus4.zip openmv/plus_cut
"""

import io
import itertools
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import tensorflow as tf
from PIL import Image

PUBLIC_TEST_ROWS = range(28709, 32298)   # FER-2013 row numbers; FER+ files are named fer<row>.png


def load(archive, labels):
    public, private = [], []
    with zipfile.ZipFile(archive) as z:
        for name in sorted(z.namelist()):
            parts = PurePosixPath(name).parts
            if len(parts) == 3 and parts[0] == "test" and parts[1] in labels and name.endswith(".png"):
                row = int(parts[2][3:10])
                gray = np.asarray(Image.open(io.BytesIO(z.read(name))).convert("L").resize((48, 48), Image.Resampling.BILINEAR))
                (public if row in PUBLIC_TEST_ROWS else private).append((labels.index(parts[1]), gray))
    return public, private


def scores(folder, items):
    it = tf.lite.Interpreter(model_path=str(folder / "expression.tflite"), num_threads=4)
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    so, zo = o["quantization"]
    out = []
    for _, g in items:
        s = 0
        for img in (g, g[:, ::-1]):           # flip averaging, as on the camera page
            it.set_tensor(i["index"], (img.astype(np.int16) - 128).astype(np.int8)[None, :, :, None])
            it.invoke()
            s = s + (it.get_tensor(o["index"])[0].astype(np.float64) - zo) * so
        out.append(s / 2)
    return np.array(out), np.array([t for t, _ in items])


def recalls(probs, truth, factors, n):
    pred = (probs * factors).argmax(1)
    return np.array([(pred[truth == c] == c).mean() for c in range(n)]), (pred == truth).mean()


def main():
    archive, folder = Path(sys.argv[1]), Path(sys.argv[2])
    labels = [l.strip() for l in (folder / "labels.txt").read_text().splitlines() if l.strip()]
    public, private = load(archive, labels)
    pp, pt = scores(folder, public)
    qp, qt = scores(folder, private)
    grid = np.round(np.arange(0.5, 4.01, 0.1), 2)
    best, best_bal = np.ones(len(labels)), -1
    anchor = labels.index("happy") if "happy" in labels else 0      # one class stays at 1.0; only ratios matter
    free = [c for c in range(len(labels)) if c != anchor]
    for combo in itertools.product(grid, repeat=len(free)):
        f = np.ones(len(labels)); f[free] = combo
        bal = recalls(pp, pt, f, len(labels))[0].mean()
        if bal > best_bal:
            best_bal, best = bal, f
    for name, f in (("before", np.ones(len(labels))), ("after", best)):
        r, acc = recalls(qp, qt, f, len(labels))
        print(f"PrivateTest {name:<6}  acc {acc:.3f}  balanced {r.mean():.3f}  " +
              "  ".join(f"{l} {v:.2f}" for l, v in zip(labels, r)))
    print("sensitivity:", dict(zip(labels, best.round(2).tolist())))
    metrics_path = folder / "model_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["sensitivity"] = dict(zip(labels, best.round(2).tolist()))
    metrics["sensitivity_note"] = "Per-class score multipliers tuned for balanced accuracy on FER+ PublicTest; apply before argmax"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
