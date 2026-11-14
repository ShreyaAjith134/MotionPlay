"""Record YOUR real gestures to train the classifier (this is what makes the ML model actually good).

    python collect_gestures.py --name shreya --reps 15

It prompts you in the terminal: get ready -> NOW! -> do the gesture for ~1.2 s.
Classes: tilt (normal playing movements), flick, shake, still.
Output:  data/gestures/<name>_<time>.csv   (label,trial,t,ax,ay,az,gx,gy,gz)
Then:    python train_model.py gesture          # uses every CSV in data/gestures/
Needs the ESP32 (or simulator.py) streaming on UDP.
"""
import argparse
import csv
import random
import sys
import time

from config import *
from receiver import Receiver

PROMPTS = {
    "tilt": "TILT left and right normally, like playing the game",
    "flick": "FLICK your wrist ONCE, sharply, at 'NOW'",
    "shake": "SHAKE the controller side to side, fast",
    "still": "HOLD STILL (rest it on your hand / keep it steady)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="me")
    ap.add_argument("--reps", type=int, default=15, help="trials per gesture")
    ap.add_argument("--pid", type=int, default=1)
    ap.add_argument("--port", type=int, default=UDP_PORT)
    ap.add_argument("--seconds", type=float, default=1.4)
    a = ap.parse_args()

    rx = Receiver(a.port)
    print("waiting for controller data...")
    while not rx.total[a.pid]:
        time.sleep(0.1)
    GESTURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = GESTURE_DATA_DIR / f"{a.name}_{int(time.time())}.csv"
    order = [c for c in GESTURE_CLASSES for _ in range(a.reps)]
    random.shuffle(order)                                 # random order so you cannot anticipate / drift
    print(f"{len(order)} trials -> {path}\n")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "trial", "t", "ax", "ay", "az", "gx", "gy", "gz"])
        for n, label in enumerate(order, 1):
            print(f"[{n}/{len(order)}]  {label.upper():6s} - {PROMPTS[label]}")
            for c in (3, 2, 1):
                print(f"   {c}...", end="", flush=True)
                time.sleep(0.6)
            rx.drain(a.pid)
            print("  NOW!\a", flush=True)
            t0, rows = time.time(), []
            while time.time() - t0 < a.seconds:
                rows += rx.drain(a.pid)
                time.sleep(0.01)
            for r in rows:
                w.writerow([label, n, f"{r[0] - t0:.4f}"] + [f"{v:.4f}" for v in r[3:9]])
            fh.flush()
            time.sleep(0.5)
    print(f"\nsaved {path}\nnow run:  python train_model.py gesture")
    rx.close()


if __name__ == "__main__":
    sys.exit(main())
