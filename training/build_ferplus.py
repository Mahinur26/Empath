# Inputs: a folder with train/publicTest/privateTest parquet files from the Hugging Face dataset
# Aaryan333/fer2013_train_publicTest_privateTest (original FER-2013 row order) and fer2013new.csv from
# https://github.com/microsoft/FERPlus. Needs pandas, pyarrow and numpy.
# Usage: python training/build_ferplus.py <source folder> <out.zip> [labels, default angry,happy,neutral,sad]
"""Build a FER+ (majority vote) ZIP in the same train/<label>/ and test/<label>/ layout train.py reads."""
import io, sys, zipfile, numpy as np, pandas as pd
src, out = sys.argv[1], sys.argv[2]
WANTED = (sys.argv[3] if len(sys.argv) > 3 else "angry,happy,neutral,sad").split(",")
parts = [pd.read_parquet(f"{src}/{n}.parquet") for n in ("train", "publicTest", "privateTest")]
df = pd.concat(parts, ignore_index=True); plus = pd.read_csv(f"{src}/fer2013new.csv")
EMO = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]
NAMES = {"anger": "angry", "happiness": "happy", "neutral": "neutral", "sadness": "sad", "surprise": "surprise",
         "fear": "fear", "disgust": "disgust"}
KEEP = {k: v for k, v in NAMES.items() if v in WANTED}
# Optional 4th argument folds classes together, e.g. "disgust=angry" (a scrunched face counts as angry).
for pair in (sys.argv[4].split(",") if len(sys.argv) > 4 else []):
    source, target = pair.split("=")
    KEEP[next(k for k, v in NAMES.items() if v == source)] = target
votes = plus[EMO].values; total = votes.sum(1) + plus["unknown"].values + plus["NF"].values
top = votes.argmax(1); topv = votes.max(1)
counts = {}
with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
    for i in range(len(df)):
        if total[i] == 0 or topv[i] <= total[i] / 2:      # no clear majority (or not a face): drop, as in FER+ "majority" mode
            continue
        label = KEEP.get(EMO[top[i]])
        if not label:
            continue
        split = "train" if plus["Usage"][i] == "Training" else "test"
        z.writestr(f"{split}/{label}/fer{i:07d}.png", df["image"][i]["bytes"])
        counts[(split, label)] = counts.get((split, label), 0) + 1
for k in sorted(counts): print(k, counts[k])
