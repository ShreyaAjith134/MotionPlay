"""Tune gestures on YOUR hardware: prints a live bar of gyro strength and every gesture the engine detects.

    python gesture_monitor.py                 # rule engine
    python gesture_monitor.py --engine ml
Try: tilt normally, flick, shake, hold still. Watch the 'peak' column:
  - your flicks should peak well above FLICK_GYRO (config.py, default 300 deg/s)  -> lower it if they do not trigger
  - normal tilting should stay well below it                                       -> raise it if tilting triggers flicks
  - holding still should read below STILL_GYRO (default 5 deg/s RMS)               -> raise it if shield never triggers
"""
import argparse
import math
import time

from config import *
from gestures import make_engine
from receiver import Receiver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["rules", "ml"], default="rules")
    ap.add_argument("--pid", type=int, default=1)
    ap.add_argument("--port", type=int, default=UDP_PORT)
    a = ap.parse_args()
    rx, eng = Receiver(a.port), make_engine(a.engine)
    print(f"engine: {eng.name}   (Ctrl+C to stop)\nwaiting for controller data...")
    peak, rms_acc, n, last = 0.0, 0.0, 0, time.time()
    try:
        while True:
            rows = rx.drain(a.pid)
            for r in rows:
                gm = math.sqrt(r[6] ** 2 + r[7] ** 2 + r[8] ** 2)
                peak, rms_acc, n = max(peak, gm), rms_acc + gm * gm, n + 1
            for t, g, src, conf in eng.update(rows):
                print(f"   >>> {g.upper():6s} detected  (conf {conf:.2f})")
            if time.time() - last >= 0.5 and n:
                rms = math.sqrt(rms_acc / n)
                bar = "#" * min(60, int(peak / 15))
                print(f"peak {peak:6.0f} deg/s   rms {rms:6.1f}   {bar}")
                peak, rms_acc, n, last = 0.0, 0.0, 0, time.time()
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
