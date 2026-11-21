"""Show facial-expression predictions in OpenMV IDE and on the glasses OLED."""

import os
import time

import framebuf
import ml
from machine import I2C
from ml.preprocessing import Normalization

try:
    import csi
except ImportError:
    csi = None
    import sensor


# Tools > ROM File System in OpenMV IDE; otherwise the files are read from the board's drive.
ROMFS_DIR = "/rom"
FLASH_DIR = "/flash"
MODEL_FILE = "expression.tflite"
LABELS_FILE = "labels.txt"
MIN_SCORE = 0.20
GUIDE_SIZE = 112
# Everyday sad and angry are subtler than the mostly posed training photos, so their scores are weighted up
# before picking the winner (the camera page's "Angry/sad sensitivity" slider). Labels not in labels.txt are ignored.
SENSITIVITY = {"angry": 1.5, "sad": 2.0}

OLED_WIDTH = 128
OLED_HEIGHT = 64
OLED_ADDRESSES = (0x3C, 0x3D)
# Only redraw the OLED when the label changes or the score moves this many percentage points.
REDRAW_STEP_PCT = 5


# Minimal SSD1306 I2C driver adapted from MicroPython Lib, MIT license (see SSD1306-LICENSE.txt).
class SSD1306_I2C(framebuf.FrameBuffer):
    def __init__(self, width, height, i2c, addr=0x3C):
        self.width = width
        self.height = height
        self.pages = height // 8
        self.i2c = i2c
        self.addr = addr
        self.cmd_buf = bytearray(2)
        self.buffer = bytearray(self.pages * width)
        super().__init__(self.buffer, width, height, framebuf.MONO_VLSB)
        for cmd in (
            0xAE,  # display off
            0x20, 0x00,  # horizontal addressing mode
            0x40,  # start line 0
            0xA1,  # column 127 mapped to SEG0
            0xA8, height - 1,  # multiplex ratio
            0xC8,  # scan COM[N] to COM0
            0xD3, 0x00,  # display offset 0
            0xDA, 0x12 if height == 64 else 0x02,  # COM pin config
            0xD5, 0x80,  # clock divide
            0xD9, 0xF1,  # pre-charge (internal VCC)
            0xDB, 0x30,  # VCOMH deselect
            0x81, 0xFF,  # contrast
            0xA4,  # output follows RAM
            0xA6,  # not inverted
            0x8D, 0x14,  # enable charge pump
            0xAF,  # display on
        ):
            self.write_cmd(cmd)
        self.fill(0)
        self.show()

    def write_cmd(self, cmd):
        self.cmd_buf[0] = 0x80
        self.cmd_buf[1] = cmd
        self.i2c.writeto(self.addr, self.cmd_buf)

    def show(self):
        for cmd in (0x21, 0, self.width - 1, 0x22, 0, self.pages - 1):
            self.write_cmd(cmd)
        self.i2c.writevto(self.addr, (b"\x40", self.buffer))


def start_display():
    try:
        bus = I2C(1, freq=400000)
        found = bus.scan()
        for address in OLED_ADDRESSES:
            if address in found:
                return SSD1306_I2C(OLED_WIDTH, OLED_HEIGHT, bus, addr=address)
        print("No OLED at 0x3C/0x3D. Continuing without display.")
    except OSError as error:
        print("OLED unavailable:", error)
    return None


def draw_big_text(oled, text, y, max_scale=3):
    # The built-in font is 8x8, so render it into a scratch buffer and blow each pixel up.
    width = len(text) * 8
    scale = max(1, min(max_scale, OLED_WIDTH // width))
    buf = bytearray(width)
    glyphs = framebuf.FrameBuffer(buf, width, 8, framebuf.MONO_VLSB)
    glyphs.text(text, 0, 0, 1)
    x0 = (OLED_WIDTH - width * scale) // 2
    for sx in range(width):
        column = buf[sx]
        if not column:
            continue
        for sy in range(8):
            if column & (1 << sy):
                oled.fill_rect(x0 + sx * scale, y + sy * scale, scale, scale, 1)


def draw_centered_text(oled, text, y):
    oled.text(text, (OLED_WIDTH - len(text) * 8) // 2, y)


def show_message(oled, line1, line2=""):
    oled.fill(0)
    draw_centered_text(oled, line1, 24)
    if line2:
        draw_centered_text(oled, line2, 36)
    oled.show()


def show_expression(oled, label, score):
    pct = round(100 * score)
    oled.fill(0)
    if score >= MIN_SCORE:
        draw_big_text(oled, label, 4)
        draw_centered_text(oled, "%d%%" % pct, 36)
    else:
        draw_big_text(oled, "?", 4)
        draw_centered_text(oled, "%s %d%%" % (label, pct), 36)
    bar_width = OLED_WIDTH - 8
    oled.rect(4, 50, bar_width, 10, 1)
    oled.fill_rect(6, 52, (bar_width - 4) * pct // 100, 6, 1)
    oled.show()


def exists(path):
    try:
        os.stat(path)
        return True
    except OSError:
        return False


def load_model():
    rom_model = ROMFS_DIR + "/" + MODEL_FILE
    if exists(rom_model):
        model = ml.Model(rom_model)
        labels_path = ROMFS_DIR + "/" + LABELS_FILE
    else:
        model = ml.Model(FLASH_DIR + "/" + MODEL_FILE)
        labels_path = FLASH_DIR + "/" + LABELS_FILE
    with open(labels_path) as label_file:
        labels = [line.strip() for line in label_file if line.strip()]
    if tuple(model.input_shape) != ((1, 48, 48, 1),) or tuple(model.output_shape) != ((1, len(labels)),):
        raise ValueError("Use the supplied expression.tflite and labels.txt together")
    return model, labels


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


def main():
    display = start_display()
    if display is not None:
        show_message(display, "Loading...")

    camera = start_camera()
    model, labels = load_model()
    weights = [SENSITIVITY.get(label, 1.0) for label in labels]
    print(model)
    if display is not None:
        show_message(display, "Ready", "look at a face")
    print("Center one face inside the square in the frame buffer.")

    normalizer = None
    shown_label = None
    shown_pct = -100
    last_print = time.ticks_ms()
    clock = time.clock()

    while True:
        clock.tick()
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
        label = labels[best]
        score = float(scores[best])
        pct = round(100 * score)
        caption = "%s %d%%" % (label if score >= MIN_SCORE else "unsure", pct)

        frame.draw_rectangle(roi, color=255)
        frame.draw_string((2, 2), caption, color=0, scale=2)
        frame.draw_string((1, 1), caption, color=255, scale=2)

        if display is not None and (label != shown_label or abs(pct - shown_pct) >= REDRAW_STEP_PCT):
            try:
                show_expression(display, label, score)
                shown_label, shown_pct = label, pct
            except OSError as error:
                print("OLED disconnected:", error)
                display = None

        now = time.ticks_ms()
        if time.ticks_diff(now, last_print) >= 500:
            print("%.1f fps\t%s" % (clock.fps(), caption))
            last_print = now


main()
