"""Draw the README figures in docs/ from the recommended model's metrics.

Run from the repo root after retraining:  py docs/make_figures.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "openmv" / "plus_affect_surprise_dis"
OUT = ROOT / "docs"

# Same warm Material palette and expression colours as web/camera.html
BG, CARD, INK, DIM, LINE, ACC = "#f6f2e7", "#efe6cc", "#2f2a1a", "#6e6857", "#ddd3b8", "#9c7a25"
COL = {"angry": "#cf5d4c", "happy": "#d6962a", "neutral": "#6f8a82", "surprise": "#8466d2"}
NAME = {"angry": "Angry", "happy": "Happy", "neutral": "Neutral", "surprise": "Surprised"}
BEFORE = "#c4bca6"
FONT = "'Google Sans','Product Sans','Segoe UI',system-ui,-apple-system,Roboto,Helvetica,Arial,sans-serif"


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'font-family="{FONT}" role="img" aria-label="{title}">\n<title>{title}</title>\n'
            f'<rect width="{w}" height="{h}" rx="28" fill="{BG}"/>\n{body}\n</svg>\n')


def text(x, y, s, size=16, fill=INK, weight=400, anchor="start", extra=""):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" font-weight="{weight}" '
            f'text-anchor="{anchor}" {extra}>{s}</text>')


def banner():
    w, h = 1280, 420
    b = []
    # Title block
    b.append(text(88, 168, "Inside Out", 104, INK, 700, extra='letter-spacing="-3"'))
    b.append(text(92, 222, "Glasses that read facial expressions,", 30, DIM))
    b.append(text(92, 262, "running on a microcontroller with 1 MB of RAM.", 30, DIM))
    chips = ["Arduino Nicla Vision", "166 KB INT8 model", "No cloud"]
    x = 92
    for c in chips:
        cw = 22 + len(c) * 10.2
        b.append(f'<rect x="{x}" y="304" width="{cw:.0f}" height="40" rx="20" fill="{CARD}" stroke="{LINE}"/>')
        b.append(text(x + cw / 2, 330, c, 17, INK, 500, "middle"))
        x += cw + 12
    # Orbit compass: x = pleasantness, y = energy, as on the camera page
    cx, cy, r = 1010, 210, 160
    b.append(f'<circle cx="{cx}" cy="{cy}" r="{r + 26}" fill="{CARD}"/>')
    for rr, op in ((r, 1), (r * 0.66, .8), (r * 0.33, .6)):
        b.append(f'<circle cx="{cx}" cy="{cy}" r="{rr:.0f}" fill="none" stroke="{LINE}" stroke-width="2" opacity="{op}"/>')
    b.append(f'<line x1="{cx - r}" y1="{cy}" x2="{cx + r}" y2="{cy}" stroke="{LINE}" stroke-width="2"/>')
    b.append(f'<line x1="{cx}" y1="{cy - r}" x2="{cx}" y2="{cy + r}" stroke="{LINE}" stroke-width="2"/>')
    pos = {"angry": (-.6, .68), "happy": (.8, .3), "neutral": (0, 0), "surprise": (.2, .86)}
    # reading needle toward happy
    hx, hy = cx + pos["happy"][0] * r, cy - pos["happy"][1] * r
    b.append(f'<line x1="{cx}" y1="{cy}" x2="{hx:.0f}" y2="{hy:.0f}" stroke="{COL["happy"]}" stroke-width="5" stroke-linecap="round"/>')
    for k, (px, py) in pos.items():
        x_, y_ = cx + px * r, cy - py * r
        big = k == "happy"
        if big:
            b.append(f'<circle cx="{x_:.0f}" cy="{y_:.0f}" r="26" fill="{COL[k]}" opacity=".22"/>')
        b.append(f'<circle cx="{x_:.0f}" cy="{y_:.0f}" r="{13 if big else 10}" fill="{COL[k]}" stroke="{BG}" stroke-width="3"/>')
        lx = x_ + (18 if px >= 0 else -18)
        ly = y_ + (5 if k != "neutral" else 30)
        anchor = "start" if px >= 0 else "end"
        if k == "neutral":
            lx, anchor = x_, "middle"
        b.append(text(f"{lx:.0f}", f"{ly:.0f}", NAME[k], 18, INK, 600 if big else 500, anchor))
    return svg(w, h, "\n".join(b), "Inside Out: glasses that read facial expressions on a microcontroller")


def recall(m):
    tests = m["pure_label_tests"]
    labels = m["labels_in_model_order"]
    w, rowh, top = 1000, 86, 176
    h = top + rowh * len(labels) + 40
    x0, x1 = 210, 900
    b = [text(48, 62, "Per-expression recall", 30, INK, 700),
         text(48, 96, "Share of each expression read correctly on faces the model never saw (flip-averaged)", 17, DIM)]
    # legend
    b.append(f'<rect x="48" y="{top - 52}" width="18" height="18" rx="5" fill="{INK}"/>')
    b.append(text(74, top - 37, "FER+ test", 16, INK))
    b.append(f'<rect x="178" y="{top - 52}" width="18" height="18" rx="5" fill="{INK}" opacity=".35"/>')
    b.append(text(204, top - 37, "AffectNet (in the wild)", 16, INK))
    for pct in (0, 25, 50, 75, 100):
        gx = x0 + (x1 - x0) * pct / 100
        b.append(f'<line x1="{gx:.0f}" y1="{top - 8}" x2="{gx:.0f}" y2="{top + rowh * len(labels) - 10}" stroke="{LINE}" stroke-dasharray="{"0" if pct == 0 else "3 5"}"/>')
        b.append(text(f"{gx:.0f}", top + rowh * len(labels) + 14, f"{pct}%", 14, DIM, anchor="middle"))
    for i, l in enumerate(labels):
        y = top + i * rowh
        b.append(f'<circle cx="62" cy="{y + 30}" r="9" fill="{COL[l]}"/>')
        b.append(text(82, y + 36, NAME[l], 20, INK, 600))
        for j, (key, op) in enumerate((("ferplus_test", 1), ("affectnet_test", .4))):
            v = tests[key][l]
            by = y + 6 + j * 30
            bw = (x1 - x0) * v
            b.append(f'<rect x="{x0}" y="{by}" width="{bw:.0f}" height="24" rx="8" fill="{COL[l]}" opacity="{op}"/>')
            b.append(text(f"{x0 + bw + 10:.0f}", by + 18, f"{v * 100:.0f}%", 16, INK, 600))
    return svg(w, h, "\n".join(b), "Per-expression recall on FER+ and AffectNet test faces")


def confusion(m):
    labels = m["labels_in_model_order"]
    cm = m["flip_average_confusion_matrix"]
    n, cell = len(labels), 104
    x0, y0 = 220, 170
    w, h = x0 + n * cell + 60, y0 + n * cell + 60
    b = [text(48, 62, "Where it gets confused", 30, INK, 700),
         text(48, 96, f"FER+ test, {sum(map(sum, cm)):,} faces · rows: true expression · columns: model's reading", 17, DIM),
         text(x0 + n * cell / 2, y0 - 46, "Model reads", 16, DIM, 600, "middle")]
    for j, l in enumerate(labels):
        b.append(text(x0 + j * cell + cell / 2, y0 - 14, NAME[l], 16, INK, 600, "middle"))
    for i, l in enumerate(labels):
        tot = sum(cm[i])
        b.append(f'<circle cx="62" cy="{y0 + i * cell + cell / 2}" r="8" fill="{COL[l]}"/>')
        b.append(text(80, y0 + i * cell + cell / 2 + 6, NAME[l], 18, INK, 600))
        for j in range(n):
            v = cm[i][j] / tot
            x, y = x0 + j * cell, y0 + i * cell
            fill, op = (COL[l], .25 + .75 * v) if i == j else (INK, min(.06 + v * 2.2, .55))
            b.append(f'<rect x="{x + 3}" y="{y + 3}" width="{cell - 6}" height="{cell - 6}" rx="14" fill="{fill}" opacity="{op:.2f}"/>')
            tc = "#ffffff" if (i == j or op > .4) else INK
            b.append(text(x + cell / 2, y + cell / 2 + 8, f"{v * 100:.0f}%", 22 if i == j else 18, tc, 700 if i == j else 500, "middle"))
    return svg(w, h, "\n".join(b), "Confusion matrix of the recommended model on the FER+ test set")


def journey(m):
    t = m["pure_label_tests"]
    rows = [
        ("Neutral faces read as neutral", "FER+ relabels replace noisy FER-2013 labels", .56, .80),
        ("Accuracy with the mouth hidden", "Cutout training stops it relying on the mouth", .37, .67),
        ("Accuracy on in-the-wild faces", "AffectNet photos added to the posed FER+ set", .675, .807),
        ("Everyday angry faces caught", "Angry score weighted ×1.5 before picking", t["affectnet_test"]["angry"], t["with_angry_x1.5"]["affectnet_angry"]),
    ]
    w, rowh, top = 1000, 92, 190
    h = top + rowh * len(rows) + 50
    x0, x1 = 540, 940
    sx = lambda v: x0 + (x1 - x0) * (v - .25) / .75
    b = [text(48, 62, "What moved the needle", 30, INK, 700),
         text(48, 96, "Each fix targeted a failure we saw in live tests", 17, DIM)]
    for pct in (25, 50, 75, 100):
        gx = sx(pct / 100)
        b.append(f'<line x1="{gx:.0f}" y1="{top - 20}" x2="{gx:.0f}" y2="{top + rowh * len(rows) - 20}" stroke="{LINE}" stroke-dasharray="3 5"/>')
        b.append(text(f"{gx:.0f}", top + rowh * len(rows) + 8, f"{pct}%", 14, DIM, anchor="middle"))
    b.append(f'<circle cx="56" cy="{top - 52}" r="8" fill="{BEFORE}"/>')
    b.append(text(72, top - 46, "before", 15, DIM))
    b.append(f'<circle cx="146" cy="{top - 52}" r="9" fill="{ACC}"/>')
    b.append(text(162, top - 46, "after", 15, DIM))
    for i, (title, how, a, c) in enumerate(rows):
        y = top + i * rowh
        b.append(text(48, y + 4, title, 20, INK, 600))
        b.append(text(48, y + 32, how, 15, DIM))
        xa, xc = sx(a), sx(c)
        b.append(f'<line x1="{xa:.0f}" y1="{y}" x2="{xc:.0f}" y2="{y}" stroke="{ACC}" stroke-width="6" stroke-linecap="round" opacity=".35"/>')
        b.append(f'<circle cx="{xa:.0f}" cy="{y}" r="9" fill="{BEFORE}"/>')
        b.append(f'<circle cx="{xc:.0f}" cy="{y}" r="11" fill="{ACC}"/>')
        b.append(text(f"{xa - 16:.0f}", y + 6, f"{a * 100:.0f}%", 15, DIM, anchor="end"))
        b.append(text(f"{xc + 18:.0f}", y + 6, f"{c * 100:.0f}%", 17, INK, 700))
    return svg(w, h, "\n".join(b), "Accuracy improvements from each training fix")


def main():
    m = json.loads((MODEL / "model_metrics.json").read_text())
    for name, s in (("banner", banner()), ("recall", recall(m)), ("confusion", confusion(m)), ("journey", journey(m))):
        (OUT / f"{name}.svg").write_text(s, encoding="utf-8")
        print("wrote", OUT / f"{name}.svg")


if __name__ == "__main__":
    main()
