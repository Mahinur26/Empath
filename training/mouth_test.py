"""Measure how much a model depends on the mouth: score the FER+ test set normally, with the lower face
hidden (rows 29-47 set to the image mean), and with the upper face hidden (rows 0-28).

A model that uses the whole face keeps working reasonably when either half is hidden. A mouth-shortcut
model collapses when the lower face is hidden and calls covered faces "sad".

Example:
    python training/mouth_test.py ferplus4.zip openmv/plus_wide
"""

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from evaluate import load_test


def run(folder, items):
    labels = [l.strip() for l in (folder / "labels.txt").read_text().splitlines() if l.strip()]
    it = tf.lite.Interpreter(model_path=str(folder / "expression.tflite"), num_threads=4)
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]

    def predict(g):
        s = 0
        for img in (g, g[:, ::-1]):
            it.set_tensor(i["index"], (img.astype(np.int16) - 128).astype(np.int8)[None, :, :, None])
            it.invoke()
            s = s + it.get_tensor(o["index"])[0].astype(np.int32)
        return int(np.argmax(s))

    shared = [(labels.index(n), g) for n, g in items if n in labels]
    for name, rows in (("full face", None), ("mouth hidden", slice(29, 48)), ("eyes hidden", slice(0, 29))):
        hits, predicted = np.zeros(len(labels)), np.zeros(len(labels))
        totals = np.zeros(len(labels))
        for truth, g in shared:
            g = g.copy()
            if rows is not None:
                g[rows] = int(g.mean())
            p = predict(g)
            totals[truth] += 1; predicted[p] += 1; hits[truth] += p == truth
        recall = {l: round(h / t, 2) for l, h, t in zip(labels, hits, totals)}
        share = {l: round(c / len(shared), 2) for l, c in zip(labels, predicted)}
        print(f"{folder.name:<10} {name:<13} acc {hits.sum() / len(shared):.3f}  recall {recall}  predicted share {share}", flush=True)


if __name__ == "__main__":
    test = load_test(Path(sys.argv[1]))
    for f in sys.argv[2:]:
        run(Path(f), test)
