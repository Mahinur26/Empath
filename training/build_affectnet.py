"""Combine FER+ with an AffectNet subset for training, and build an in-the-wild AffectNet test set.

Inputs:
  ferplus.zip      train/<label>/ and test/<label>/ ZIP from build_ferplus.py
  affect_src       folder with train.parquet and val.parquet from the Hugging Face dataset
                   Mauregato/affectnet_short (96x96 RGB AffectNet face crops; labels 0 anger, 3 happy,
                   4 neutral, 6 sad, ...). AffectNet is licensed for non-commercial research only; this
                   unofficial copy is used here for a hackathon prototype. Do not commit the images.
Outputs:
  combined.zip     train/ = FER+ train + AffectNet train, test/ = FER+ test (unchanged, for comparison)
  affect_test.zip  test/ = AffectNet val split, same crop and size as training

AffectNet crops are looser than FER (eyes 27% of the width apart vs 40%), so each image is cropped about
1.46x around the eyes to match FER's framing before it is shrunk to 48x48 grayscale.

Usage: python training/build_affectnet.py ferplus.zip affect_src combined.zip affect_test.zip
"""

import io
import sys
import zipfile

import pandas as pd
from PIL import Image

AFFECT = {0: "angry", 3: "happy", 4: "neutral", 6: "sad"}
CROP = (15.1, 14.4, 80.8, 80.1)   # in 96x96 pixels: eyes land at FER's 37.5% height, 39.6% apart


def face48(raw):
    image = Image.open(io.BytesIO(raw)).convert("L").resize((96, 96), Image.Resampling.BILINEAR)
    small = image.crop(tuple(round(v) for v in CROP)).resize((48, 48), Image.Resampling.BILINEAR)
    out = io.BytesIO()
    small.save(out, format="PNG")
    return out.getvalue()


def add_affect(z, frame, split, counts):
    for i, (img, label) in enumerate(zip(frame["image"], frame["label"])):
        name = AFFECT.get(int(label))
        if name:
            z.writestr(f"{split}/{name}/affect{i:06d}.png", face48(img["bytes"]))
            counts[(split, name)] = counts.get((split, name), 0) + 1


def main():
    ferplus, src, combined, affect_test = sys.argv[1:5]
    counts = {}
    with zipfile.ZipFile(combined, "w", zipfile.ZIP_STORED) as z:
        with zipfile.ZipFile(ferplus) as f:
            for name in f.namelist():
                if name.endswith(".png"):
                    z.writestr(name, f.read(name))
                    counts[(name.split("/")[0] + "_ferplus", name.split("/")[1])] = counts.get((name.split("/")[0] + "_ferplus", name.split("/")[1]), 0) + 1
        add_affect(z, pd.read_parquet(f"{src}/train.parquet"), "train", counts)
    with zipfile.ZipFile(affect_test, "w", zipfile.ZIP_STORED) as z:
        add_affect(z, pd.read_parquet(f"{src}/val.parquet"), "test", counts)
    for key in sorted(counts):
        print(key, counts[key])


if __name__ == "__main__":
    main()
