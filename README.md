# MotionPlay

**Sense → Ingest → Process → Store → Analyze → (ML) → Feedback**

```
MPU6050 + button ──ESP32──WiFi/UDP 50 Hz──> laptop: receiver → gesture engine → game → SQLite → analyze / ML
                      ▲                                                    │
                      └── UDP 4211: buzzer · vibration · LED · OLED  ◄─────┘   (feedback loop)
```

## 1. Features

| Area | What it does |
|---|---|
| Control | tilt to move the paddle, plus **FLICK** (pulse), **SHAKE** (bomb), **HOLD STILL** (shield) and an optional button |
| Game | good / gold / **red** orbs, lives, combo ×5, levels, **boss wave** every 5th level, **adaptive difficulty** |
| Players | solo or **2-player split screen** (two ESP32s) |
| Hardware | ESP32 + MPU6050, plus optional button, buzzer, vibration motor, OLED and RGB LED (the game talks back to the controller) |
| ML | **gesture classifier**, **next-10 s performance (fatigue) predictor**, **player identification** |
| Analytics | 9 charts + a live dashboard while playing |

## 2. Hardware (all add-ons optional)

| Part | Pin | Notes |
|---|---|---|
| MPU6050 | SDA **GPIO21**, SCL **GPIO22**, VCC 3V3, GND | required |
| OLED 0.96" SSD1306 **I²C** (128×64, addr 0x3C) | same SDA/SCL as MPU6050 | shows score / lives / level / combo. Check the pin order printed on your module (VCC/GND order differs between brands) |
| Push button | **GPIO27** ↔ GND | internal pull-up, no resistor needed. Press = pulse; also start / restart |
| Active buzzer | **GPIO26** (+), GND (−) | must be an **active** buzzer (makes a tone by itself) |
| Vibration motor **module** | **GPIO25** → IN, VCC 3V3, GND | use a module with a driver. A bare motor needs an NPN transistor + 1 kΩ + diode; never drive a motor straight from a pin |
| WS2812 / NeoPixel | **GPIO4** → DIN via 330 Ω, VCC, GND | 1 pixel is enough (`RGB_COUNT` in the .ino) |

Diagram: `docs/wiring.png`. **Power:** USB cable or a power bank (a bare LiPo cannot power the board correctly without a boost converter).

**Sharing SDA/SCL between MPU6050 and OLED:** the ESP32 only has one pair, so both boards connect to the same two wires. Easiest: a small breadboard as a hub: plug the MPU6050 and OLED headers in, then join matching pins (VCC/GND/SDA/SCL) with jumpers. A typical ESP32 dev board is too wide to leave free holes on a half-size breadboard, so keep it off the breadboard and use wires to it.

**Feedback the game sends back**

| Code | When | Buzzer | Motor | LED |
|---|---|---|---|---|
| H | caught an orb | 40 ms | 40 ms | green |
| M | missed an orb | – | 60 ms | amber |
| X | red orb hit (life lost) | 250 ms | 300 ms | red |
| G | gesture used | 25 ms | – | blue |
| S | every 0.5 s | – | – | OLED: score, lives, level, combo |
| E | game over | 400 ms | – | purple, OLED "GAME OVER" |

## 3. Flash the firmware

1. Arduino IDE → open `firmware/motionplay/motionplay.ino`.
2. Set `WIFI_SSID`, `WIFI_PASS`, `LAPTOP_IP`, `PLAYER_ID` (1, or 2 for the second controller).
3. Turn on only what you wired: `#define USE_BUTTON 1`, `USE_BUZZER`, `USE_MOTOR`, `USE_OLED`, `USE_RGB`.
4. Install libraries for the flags you enabled (Library Manager): OLED → *Adafruit SSD1306* + *Adafruit GFX*; RGB → *Adafruit NeoPixel*.
5. Board *ESP32 Dev Module* → Upload → Serial Monitor 115200 → `Connected, ESP32 IP: ...`.

Add parts one at a time and re-flash after each: easier to find a wiring mistake. With all flags at 0 the sketch only streams sensor data.

## 4. Run (laptop)

```bash
pip install -r requirements.txt
cd laptop
python selftest.py                                # ~20 s, checks the whole pipeline, touches nothing of yours
```

| Goal | Command |
|---|---|
| Try with **no hardware** | `python game.py --player me --mouse` (keys **Z X C** = flick / shake / still) |
| Fake controller | terminal 1 `python simulator.py` → terminal 2 `python game.py --player me` |
| **Play with the ESP32** | `python game.py --player shreya` |
| 2 players | `python game.py --player1 shreya --player2 friend` (ESP32 #1 and #2). Same orb sequence for both, split screen |
| ML gesture engine | `python game.py --player shreya --engine ml` |
| Fixed difficulty | add `--no-adapt` |
| Live charts while playing | second terminal: `python live_dashboard.py` |
| Tune gestures for your hand | `python gesture_monitor.py` |
| Analyze | `python analyze.py --player shreya` → `analysis_out/dashboard.png` |
| Demo data (3 bot players) | `python demo_data.py` then `python analyze.py --db motionplay_demo.db` |

Data goes to `laptop/motionplay.db` (created on first run). **SPACE or the button restarts** after game over; **ESC** quits.

### Game rules

| Orb | Effect |
|---|---|
| cyan (good) | +10 × combo multiplier (×1 … ×5, +1 per 5 combo) |
| gold (fast) | +50 × multiplier, no penalty if missed |
| red (bad) | **catching it costs a life** (5 lives). Avoid it |

| Gesture | Power-up |
|---|---|
| **FLICK** the controller sharply (or press the button) | **pulse**: destroys every red orb on screen (4 s cooldown) |
| **SHAKE** | **bomb**: catches every good orb and destroys red ones. One bomb is earned per 10 combo (max 3) |
| **HOLD STILL** 1 s | **shield**: absorbs one red-orb hit (10 s cooldown) |

A new level every 12 orbs (+8 % speed, more red orbs); **boss wave** (a snake of 10 orbs) when you reach levels 5, 10, ….
**Adaptive difficulty:** a feedback controller watches your rolling catch rate (last 12 orbs). Above 85 % the game speeds up, below 65 % it slows down, so everyone plays near 75 %. Every orb's difficulty is logged.

## 5. ML models (`laptop/train_model.py`)

| Model | Input → output | Evaluation | Result on shipped data |
|---|---|---|---|
| **Gesture classifier** (RF / HistGB / SVM / LogReg, best by CV) | 0.6 s raw IMU window → tilt / flick / shake / still | stratified 5-fold (grouped by recording trial for real data) | ~100 % on **synthetic** windows. This only proves the pipeline; it is **not** a claim about your hand. Record your own (below) |
| **Performance / fatigue predictor** | 10 s window (jerk, gyro RMS, paddle speed, catch rate, reaction, difficulty, gestures, elapsed time) → catch rate of the **next** 10 s | 5-fold **grouped by session** | MAE 0.154 vs 0.183 (predict-the-mean) and 0.190 ("next = now"); R² +0.20. Modest but real on demo data |
| **Player identification** | 10 s movement-style window → who is playing | 5-fold grouped by session | 3 demo players: 65-67 % per window, **78-89 % per session** (majority vote) vs 33 % chance |

Shipped models in `models/` were trained on **synthetic / bot data**, so retrain on your own:

```bash
python collect_gestures.py --name shreya --reps 15      # ~4 min: tilt, flick, shake, still, random order
python train_model.py gesture                           # uses every CSV in data/gestures/
python game.py --player shreya --engine ml

# after ~5+ real sessions (player id needs 2 players x 3 sessions):
python train_model.py fatigue
python train_model.py player
```

Charts: `models/gesture_confusion.png`, `fatigue_eval.png` (actual vs predicted + feature importance), `player_confusion.png`.
If scikit-learn warns that a `.joblib` was built with another version, just retrain (one command).

## 6. Gesture detection: two engines

* **rules** (default): thresholds in `config.py`: `FLICK_GYRO` 300 deg/s, shake = ≥3 direction changes in 0.8 s, still = gyro RMS < 5 deg/s for 1 s. Flick fires ~0.25 s after the peak (it waits to be sure it is not the start of a shake).
* **ml**: random forest on 34 window features (`features.py`), re-armed after each gesture, flick held 0.2 s so a shake isn't mistaken for a flick.

On synthetic streams the rule engine found 78/78 flicks, 87/87 shakes, 80/90 stills with ~1 false flick (it also fires `still` in any calm second, which is the intended shield trigger); the ML engine found all of them with some false flicks at shake onsets (14). **Real hands differ from my simulator**, so run `gesture_monitor.py` and adjust `FLICK_GYRO` / `STILL_GYRO` until flicks trigger and normal tilting does not. If clipping occurs on hard flicks, set the MPU gyro range to ±500 (comment in the .ino).

## 7. Analytics (`analyze.py`)

Learning curve (regression slope + p-value) · score / session · reaction time · smoothness (RMS jerk) · **fatigue** (first vs last third: reaction, tremor, catch rate) · playstyle clusters (k-means on 10 s windows) · gesture usage · adaptive difficulty over time · catch rate vs red-orb mistakes. Outputs `dashboard.png`, `session_summary.csv`, `report.json`.

## 8. Database (laptop/motionplay.db)

```
sessions(id, player, started, ended, mode, slot, engine, adaptive, score, level, lives_left)
samples (session_id, seq, t_dev_ms, t_host, ax..gz, btn)                         ~50 rows/s
events  (session_id, idx, kind, t_spawn, orb_x, paddle_x_spawn, t_resolve, outcome, ok,
         paddle_x_resolve, level, speed_mult, combo, score, lives)               one row per orb
frames  (session_id, t, paddle_x)
gestures(session_id, t, gesture, source, conf, applied)                          source = rules | ml | key | button
```

## 9. Viva talking points

* Closed loop: the system doesn't only observe, it **acts** (adaptive difficulty + haptic/LED/OLED feedback).
* UDP for the 50 Hz stream (a late packet is useless), CSV payload for debuggability, two clocks (device and host).
* Gesture detection is a **time-series classification** problem: rules baseline vs learned model, compared by cross-validation.
* **Leakage-aware evaluation:** grouped by session/trial, so windows from one session never appear in both train and test. Naive splitting would overstate accuracy.
* Honest results: baselines (mean, persistence) reported next to the model; synthetic-data limits stated.
* Pure-logic `arena.py` is shared by the game, the bots that generate demo data and the self-test: one source of truth.

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| "waiting for controller..." | wrong `LAPTOP_IP`, different WiFi, firewall (allow Python on UDP 4210), campus WiFi isolation → use phone hotspot |
| Feedback (buzzer / OLED) does nothing | the game replies to the controller's IP on UDP 4211. Check the Serial Monitor shows `Connected`, that `USE_*` flags are 1, and that the network has no client isolation (phone hotspot fixes it) |
| OLED blank | wrong address (try 0x3C vs 0x3D in the .ino), SDA/SCL swapped, library not installed, `USE_OLED` still 0 |
| Gestures fire while just tilting | raise `FLICK_GYRO` / `SHAKE_MIN_GYRO` |
| Flicks never trigger | lower `FLICK_GYRO`, or move the MPU gyro to ±500 and set `GYR_LSB = 65.5` |
| Shield triggers too easily / never | `STILL_GYRO` up / down |
| Paddle reversed or not moving | `TILT_SIGN = -1` or `TILT_AXIS = 0` in `config.py` |
| `MPU read failed` | SDA/SCL wiring, power from 3V3, loose header solder |
| Second controller hijacks first | both ESP32s must have different `PLAYER_ID` |

## 11. Files

```
firmware/motionplay/motionplay.ino         controller firmware (flags for each add-on)
laptop/arena.py        game logic (no pygame)            laptop/game.py          pygame game, solo + versus
laptop/gestures.py     rule + ML engines                 laptop/features.py      window features
laptop/receiver.py     UDP in + feedback out             laptop/db.py            SQLite
laptop/synth.py        synthetic IMU                     laptop/simulator.py     fake ESP32
laptop/demo_data.py    bot players                       laptop/collect_gestures.py  record real gestures
laptop/analysis_lib.py + analyze.py                      laptop/train_model.py   3 ML models
laptop/live_dashboard.py   laptop/gesture_monitor.py      laptop/selftest.py
laptop/config.py       every tunable number
models/  docs/wiring.png
```
