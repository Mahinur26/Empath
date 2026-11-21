"""Show facial-expression predictions in OpenMV IDE and on the glasses OLED."""

import time

import ml
from ml.preprocessing import Normalization

try:
    import csi
except ImportError:
    csi = None
    import sensor


MODEL_PATH = "/flash/expression.tflite"
LABELS_PATH = "/flash/labels.txt"
MIN_SCORE = 0.60
GUIDE_SIZE = 112
# Everyday sad and angry are subtler than the mostly posed training photos, so their scores are weighted up
# before picking the winner (the camera page's "Angry/sad sensitivity" slider). Labels not in labels.txt are ignored.
SENSITIVITY = {"angry": 1.5, "sad": 2.0}


def start_camera():
    if csi is not None:
        camera = csi.CSI()
        camera.reset()
        camera.pixformat(csi.GRAYSCALE)
        camera.framesize(csi.QQVGA)
        camera.snapshot(time=1200)
        return camera

    sensor.reset()
    sensor.set_pixformat(sensor.GRAYSCALE)
    sensor.set_framesize(sensor.QQVGA)
    sensor.skip_frames(time=1200)
    return sensor


def start_display():
    try:
        from machine import I2C
        from ssd1306 import SSD1306_I2C

        bus = I2C(1, freq=400000)
        addresses = bus.scan()
        for address in (0x3C, 0x3D):
            if address in addresses:
                return SSD1306_I2C(128, 64, bus, addr=address)
    except (ImportError, OSError) as error:
        print("OLED unavailable:", error)
    return None


def main():
    with open(LABELS_PATH) as label_file:
        labels = [line.strip() for line in label_file if line.strip()]
    if len(labels) < 2:
        raise ValueError("Copy the labels.txt supplied with the model to /flash")

    weights = [SENSITIVITY.get(label, 1.0) for label in labels]
    model = ml.Model(MODEL_PATH)
    if model.input_shape != [(1, 48, 48, 1)] or model.output_shape != [(1, len(labels))]:
        raise ValueError("Use the supplied expression.tflite and labels.txt together")

    camera = start_camera()
    display = start_display()
    normalizer = None
    last_output = time.ticks_ms()
    print("Center one face inside the square in the frame buffer.")

    while True:
        frame = camera.snapshot()
        if normalizer is None:
            side = min(GUIDE_SIZE, frame.width(), frame.height())
            roi = ((frame.width() - side) // 2, (frame.height() - side) // 2, side, side)
            normalizer = Normalization(roi=roi)

        scores = model.predict([normalizer(frame)])[0].flatten()
        weighted = [float(scores[i]) * weights[i] for i in range(len(labels))]
        total = sum(weighted) or 1.0
        scores = [value / total for value in weighted]
        best = max(range(len(labels)), key=lambda index: scores[index])
        score = float(scores[best])
        label = labels[best] if score >= MIN_SCORE else "unsure"
        caption = "%s %d%%" % (label, round(100 * score))

        frame.draw_rectangle(roi, color=255)
        frame.draw_string(2, 2, caption, color=0, scale=2)
        frame.draw_string(1, 1, caption, color=255, scale=2)
        now = time.ticks_ms()
        if time.ticks_diff(now, last_output) >= 500:
            print(caption)
            if display is not None:
                try:
                    display.fill(0)
                    display.text("EXPRESSION", 4, 12)
                    display.text(caption, 4, 34)
                    display.show()
                except OSError as error:
                    print("OLED disconnected:", error)
                    display = None
            last_output = now


main()
