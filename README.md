# Empath on Arduino Nicla Vision

Empath classifies visible facial expressions on the Nicla Vision. The recommended model, [`openmv/plus_affect_surprise_dis`](openmv/plus_affect_surprise_dis), outputs `angry`, `happy`, `neutral`, and `surprise`; its `angry` also covers scrunched, disgusted faces. On held-out faces it scores 89.2% on the FER+ test set and 81.3% on in-the-wild AffectNet photos, and reads 79% of scrunched faces as angry. [Metrics](openmv/plus_affect_surprise_dis/model_metrics.json)

## Run it in OpenMV IDE

1. Connect the Nicla Vision with a USB data cable. Click **Connect** in OpenMV IDE and install the latest OpenMV firmware if prompted. [Arduino's Nicla Vision setup guide](https://docs.arduino.cc/tutorials/nicla-vision/user-manual)
2. Copy [`expression.tflite`](openmv/plus_affect_surprise_dis/expression.tflite) and [`labels.txt`](openmv/plus_affect_surprise_dis/labels.txt) from `openmv/plus_affect_surprise_dis/` to the root of the drive that appears for the board. Keep the two files together. Any other model folder in `openmv/` works the same way.
3. Open [`openmv/main.py`](openmv/main.py) in OpenMV IDE and click **Run**. Center one face inside the square in the camera preview. The expression and model score appear over the preview and in the serial terminal. The script shows `unsure` below a score of 20% and weights the angry score by 1.5 (`SENSITIVITY` in the script), matching the camera test page.
4. For automatic startup, save the script as `main.py` in the root of the board's drive, eject the drive, and reset the board. For standalone display, copy [`ssd1306.py`](openmv/ssd1306.py) to the board alongside the script before ejecting. With the team's SSD1306 wired to J2 and the battery connected, turn on the glasses and center a face in the camera's view. The script updates the OLED about twice per second and also runs when the OLED is disconnected. [Team wiring and parts](Hardware/BOM.md)

The square is a fixed crop. For the demo, fill most of it with one front-facing face. The script classifies visible expressions and can produce a score for an empty frame or an expression outside the model's classes. The USB preview and serial terminal belong to OpenMV IDE. The recommended model is about 166 KB, roughly six times the compute of the original model; check its frame rate and memory on the board.

The OLED driver is from [MicroPython Lib](https://github.com/micropython/micropython-lib/blob/a08087249fda8a7994f7c54ccaad29fb9fcc448a/micropython/drivers/display/ssd1306/ssd1306.py) and is distributed under the [MIT license](openmv/SSD1306-LICENSE.txt).

## Try it on a laptop camera

[`web/camera.html`](web/camera.html) runs any model in `openmv/` on the laptop webcam with a small JavaScript INT8 interpreter that matches the official TFLite runtime. It adds face tracking and eye alignment, a sensitivity slider and a "Why this reading?" card. Frames never leave the browser. From this folder, run `py -m http.server 8000` and open `http://localhost:8000/web/camera.html`.

## Models

| Folder | Classes | Notes |
|---|---|---|
| `plus_affect_surprise_dis` | angry, happy, neutral, surprise | Recommended. FER+ plus AffectNet, cutout; disgust folded into angry |
| `plus_affect_surprise` | angry, happy, neutral, surprise | Same without the disgust merge; slightly better on pure angry |
| `plus_affect` | angry, happy, neutral, sad | Same recipe with sad; sad is hard to read live |
| `plus_cut`, `plus_wide`, `plus_soft`, `plus_eq` | angry, happy, neutral, sad | FER+ only: cutout, plain, soft labels, histogram-equalised |
| `mvp4`, `mvp3`, `neutral` | various | Original FER labels |
| `mnv2`, `plus_mnv2*` | angry, happy, neutral, sad | MobileNetV2 96×96 experiments; weaker on 48×48 faces |
| `expression.tflite` (root) | angry, happy, sad, surprise | The original small model |

The `plus_affect*` models were trained partly on an unofficial copy of AffectNet, which is licensed for non-commercial research only. They are hackathon prototypes; the images are not in this repository.

## Training

The models are small grayscale CNNs at the dataset's native 48×48 resolution, trained in TensorFlow and exported with INT8 weights and activations for OpenMV. They are separate from the 96×96 MobileNetV2 setup in Edge Impulse Studio.

- [`training/train.py`](training/train.py) trains and exports a model. Options: `--labels`, `--balance`, `--arch wide`, `--augment`, `--cutout`, `--soft`, `--equalize`.
- [`training/build_ferplus.py`](training/build_ferplus.py) and [`training/build_affectnet.py`](training/build_affectnet.py) build the FER+ and FER+ + AffectNet datasets.
- [`training/evaluate.py`](training/evaluate.py), [`training/mouth_test.py`](training/mouth_test.py) and [`training/tune_sensitivity.py`](training/tune_sensitivity.py) score models, measure reliance on the mouth, and measure per-class sensitivity trade-offs.

To reproduce the original model with Python 3.11, download the FER-2013 ZIP from Kaggle, then run these commands from this folder. [FER-2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013/data)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r training\requirements.txt
.\.venv\Scripts\python.exe training\train.py path\to\fer2013.zip
```
