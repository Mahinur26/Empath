# Inputs: a folder with train/publicTest/privateTest parquet files from the Hugging Face dataset
# Aaryan333/fer2013_train_publicTest_privateTest (original FER-2013 row order) and fer2013new.csv from
# https://github.com/microsoft/FERPlus. Needs pandas, pyarrow and numpy.
# Usage: python training/build_ferplus_soft.py <source folder> <out.zip> <soft_labels.json>
"""FER+ with soft labels: train = every face whose votes are mostly (>=50%) one of the 4 classes, with the
vote proportions over [angry, happy, neutral, sad] saved to JSON; test = same majority-vote test as ferplus4.zip."""
import json, sys, zipfile, numpy as np, pandas as pd
src, out, soft_out = sys.argv[1:4]
parts = [pd.read_parquet(f"{src}/{n}.parquet") for n in ("train", "publicTest", "privateTest")]
df = pd.concat(parts, ignore_index=True); plus = pd.read_csv(f"{src}/fer2013new.csv")
EMO = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]
FOUR = ["anger", "happiness", "neutral", "sadness"]; NAMES = ["angry", "happy", "neutral", "sad"]
votes = plus[EMO].values.astype(float); total = votes.sum(1) + plus["unknown"].values + plus["NF"].values
four = plus[FOUR].values.astype(float)
soft, counts = {}, {}
with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
    for i in range(len(df)):
        if total[i] == 0 or plus["NF"][i] > 0:
            continue
        if plus["Usage"][i] == "Training":
            if four[i].sum() < total[i] / 2:          # mostly some other emotion (surprise, fear, ...): skip
                continue
            p = four[i] / four[i].sum(); label = NAMES[int(p.argmax())]
            name = f"train/{label}/fer{i:07d}.png"; soft[name] = [round(float(v), 4) for v in p]
        else:                                          # test: identical rule to ferplus4.zip (majority vote)
            if votes[i].max() <= total[i] / 2 or EMO[votes[i].argmax()] not in FOUR:
                continue
            label = NAMES[FOUR.index(EMO[votes[i].argmax()])]; name = f"test/{label}/fer{i:07d}.png"
        z.writestr(name, df["image"][i]["bytes"]); counts[name.split("/")[0], label] = counts.get((name.split("/")[0], label), 0) + 1
json.dump(soft, open(soft_out, "w"))
for k in sorted(counts): print(k, counts[k])
