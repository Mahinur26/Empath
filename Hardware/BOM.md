# Bill of Materials — Emotion-Detection Glasses


Wiring reference: [Cirkit Designer schematic](https://app.cirkitdesigner.com/project/69ebc1e3-7c25-4586-92e4-6b42f8a82b8d)

<img width="1364" height="926" alt="image" src="https://github.com/user-attachments/assets/f5827699-80cc-4ae7-b24c-a90063b3911a" />

---

## Electronics

| Qty | Part | Spec / notes | Source |
| --- | --- | --- | --- |
| 1 | Arduino Nicla Vision | ABX00051. 22.86 × 22.86 mm. Camera forward on right temple | [Arduino store](https://store.arduino.cc/products/nicla-vision) |
| 1 | 0.96" SSD1306 OLED | 128 × 64, 4-pin I2C (`VCC GND SCL SDA` or `GND VCC SCL SDA` — check silkscreen). Addr 0x3C typical | [Winstar-style](https://www.winstar.com.tw/products/oled-module/graphic-oled-display/4-pin-oled.html) / Amazon clone |
| 1 | LiPo battery | EEMB LP653042, 3.7 V **820 mAh**, ~30.5 × 44 × 6.8 mm, JST PH 2.0 (red +, black −) | [Amazon B082152887](https://www.amazon.com/dp/B082152887) · [EEMB](https://www.eemb.com/product-216) |
| 1 | Slide switch | Mini SPST/SPDT, SS12D00-type. ≥ 0.5 A. Lever fits 7 × 3 mm window in battery bay | Digi-Key / Amazon / AliExpress |
| 4 | Female Dupont housings | 2.54 mm, for OLED male header. Straight or right-angle pins OK | Standard jumper kit |


### Optional / convenience

| Qty | Part | Spec / notes | Source |
| --- | --- | --- | --- |
| 1 | JST PH 2.0 2-pin socket | Mating connector if you keep the battery plug instead of cutting it | Digi-Key / Adafruit |
| — | Heat-shrink / Kapton | Strain relief and insulation at solder joints | — |

---

## Wiring (soldering map)

Battery through inline switch → Nicla VBAT. OLED on J2 I2C.

| From | To | Wire |
| --- | --- | --- |
| Battery red (+) | Switch common → Nicla **J3 VBAT** (pad nearest USB) | Red |
| Battery black (−) | Nicla **J2-6 GND** (or J4 pin 3 GND) | Black |
| OLED VCC | Nicla **J2-7 VDDIO_EXT** (3.3 V) | Red |
| OLED GND | Nicla **J2-6 GND** | Black |
| OLED SCL | Nicla **J2-2 SCL** | — |
| OLED SDA | Nicla **J2-1 SDA** | — |
| Battery NTC | Leave unconnected | — |

**Do not** reverse battery polarity (no reverse-polarity protection). **Do not** use J3 and J4 at the same time. Switch must be **ON** for USB charging.

---

## Mechanical 

| Qty | Part | Spec / notes | Source |
| --- | --- | --- | --- |
| 1 set | 3D-printed mounts | `nicla_mount.stl`, `display_battery_holder.stl`, `clip_test_comb.stl` (print comb first) | `stl/` via `cad/glasses_parts.py` |
| — | PETG filament | Recommended for snap clips (PLA creeps/breaks). 0.2 mm layers, ≥ 3 walls, no supports | — |

---

## Consumables & tools (not product BOM)

- Soldering iron, flux, solder
- Multimeter (verify battery polarity before soldering)
- Calipers (measure OLED PCB, Nicla thickness, temple straight length before final CAD)
- Micro-USB cable (program / charge Nicla)

---

## Estimated runtime

~105 mA (Nicla capture) + ~15 mA (OLED) ≈ **120 mA**. 820 mAh ÷ 120 mA ≈ 6.8 h theoretical → expect **5–6 h** continuous. Confirm with onboard MAX17262 fuel gauge.

---

## Open purchase decisions

- Exact SS12D00 (or equivalent) switch SKU for the 7 × 3 mm lever window
- OLED vendor (pin order and PCB size vary; measure before locking CAD)
- Keep JST PH plug vs. cut-and-solder battery leads
