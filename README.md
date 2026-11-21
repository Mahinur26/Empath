# Empath on Arduino Nicla Vision

The included TensorFlow Lite model classifies four facial expressions on the Nicla Vision. Its output labels are `angry`, `happy`, `sad`, and `surprise`.

## Run it in OpenMV IDE

1. Connect the Nicla Vision with a USB data cable. Click **Connect** in OpenMV IDE and install the latest OpenMV firmware if prompted. [Arduino's Nicla Vision setup guide](https://docs.arduino.cc/tutorials/nicla-vision/user-manual)
2. Copy [`expression.tflite`](openmv/expression.tflite) and [`labels.txt`](openmv/labels.txt) to the root of the drive that appears for the board. Keep the two files together.
3. Open [`openmv/main.py`](openmv/main.py) in OpenMV IDE and click **Run**. Center one face inside the square in the camera preview. The expression and model score appear over the preview and in the serial terminal. The script shows `unsure` below a score of 60%.
4. For automatic startup, save the script as `main.py` in the root of the board's drive, eject the drive, and reset the board. For standalone display, copy [`ssd1306.py`](openmv/ssd1306.py) to the board alongside the script before ejecting. With the team's SSD1306 wired to J2 and the battery connected, turn on the glasses and center a face in the camera's view. The script updates the OLED about twice per second and also runs when the OLED is disconnected. [Team wiring and parts](Hardware/BOM.md)

The square is a fixed crop. For the demo, fill most of it with one front-facing face. The script classifies visible expressions and can produce a score for an empty frame or an expression outside the four classes. The USB preview and serial terminal belong to OpenMV IDE.

The OLED driver is from [MicroPython Lib](https://github.com/micropython/micropython-lib/blob/a08087249fda8a7994f7c54ccaad29fb9fcc448a/micropython/drivers/display/ssd1306/ssd1306.py) and is distributed under the [MIT license](openmv/SSD1306-LICENSE.txt).

## Training

The bundled model is a small grayscale CNN that uses the dataset's native 48×48 resolution. It was trained locally in TensorFlow and exported with INT8 weights and activations for OpenMV. The supplied files form a separate deployment from the 96×96 MobileNetV2 setup in Edge Impulse Studio.

The training script uses a stratified validation subset from Kaggle's training split and measures the quantized model on the original test split. [FER-2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013/data) · [Held-out test metrics](model_metrics.json)

To reproduce training with Python 3.11, download the FER-2013 ZIP from Kaggle, then run these commands from this folder.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r training\requirements.txt
.\.venv\Scripts\python.exe training\train.py path\to\fer2013.zip
```
