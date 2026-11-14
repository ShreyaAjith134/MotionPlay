"""Gesture detection from the IMU stream.

rows are the tuples produced by receiver.py:
    (t_host, seq, t_dev, ax, ay, az, gx, gy, gz, btn)
update(rows) returns a list of (t, gesture, source, confidence).

RuleEngine : thresholds from config.py (works out of the box, tune FLICK_GYRO etc. for your hand)
MLEngine   : random-forest / SVM trained by train_model.py gesture  (models/gesture_model.joblib)
"""
import collections
import math

import numpy as np

from config import *
from features import reversal_count, window_features


class RuleEngine:
    name = "rules"

    def __init__(self):
        n = int(SAMPLE_HZ * max(SHAKE_WINDOW, STILL_SEC)) + 5
        self.buf = collections.deque(maxlen=n)        # (t, gx, gy, gz, gm)
        self.cand = None                              # pending flick: (t_start, peak)
        self.flick_cd = self.shake_cd = 0.0
        self.still_armed = True

    def _recent(self, t, seconds):
        return [b for b in self.buf if b[0] >= t - seconds]

    def _rev(self, t, seconds, thr=60.0):
        r = self._recent(t, seconds)
        return reversal_count([b[1:4] for b in r], thr) if len(r) > 3 else 0

    def update(self, rows):
        out = []
        for r in rows:
            t = r[0]
            gx, gy, gz = r[6], r[7], r[8]
            gm = math.sqrt(gx * gx + gy * gy + gz * gz)
            self.buf.append((t, gx, gy, gz, gm))

            # ---- FLICK: sharp peak that is NOT followed by more back-and-forth
            if self.cand is None:
                if gm > FLICK_GYRO and t >= self.flick_cd:
                    self.cand = (t, gm)
            else:
                self.cand = (self.cand[0], max(self.cand[1], gm))
                age = t - self.cand[0]
                if age >= FLICK_CONFIRM:
                    calm = max(b[4] for b in self._recent(t, 0.08)) < 0.6 * FLICK_GYRO
                    if calm and self._rev(t, 0.6) < FLICK_MAX_REV:
                        out.append((t, "flick", "rules", min(1.0, 0.5 + self.cand[1] / (4 * FLICK_GYRO))))
                        self.flick_cd = t + FLICK_COOLDOWN
                        self.cand = None
                    elif age > 0.6:                      # still ringing -> it was a shake / tilt, not a flick
                        self.cand = None

            # ---- SHAKE: repeated direction changes
            if t >= self.shake_cd and self._rev(t, SHAKE_WINDOW, SHAKE_MIN_GYRO) >= SHAKE_REVERSALS:
                out.append((t, "shake", "rules", 0.9))
                self.shake_cd = t + SHAKE_COOLDOWN
                self.flick_cd = t + 0.7
                self.cand = None

            # ---- STILL: calm for STILL_SEC (fires once, re-arms after you move again)
            rec = self._recent(t, STILL_SEC)
            if len(rec) >= int(0.9 * STILL_SEC * SAMPLE_HZ):
                g = np.array([b[4] for b in rec])
                if self.still_armed and float(np.sqrt(np.mean(g ** 2))) < STILL_GYRO and g.max() < 3 * STILL_GYRO:
                    out.append((t, "still", "rules", 0.9))
                    self.still_armed = False
            if not self.still_armed and gm > STILL_REARM:
                self.still_armed = True
        return out


class MLEngine:
    name = "ml"

    def __init__(self, path):
        import joblib
        d = joblib.load(path)
        self.model = d["model"]
        self.classes = list(d["classes"])
        self.win = int(d.get("window", GESTURE_WINDOW))
        self.buf = collections.deque(maxlen=self.win)
        self.count = 0
        self.pending, self.streak = None, 0
        self.cd = {"flick": 0.0, "shake": 0.0, "still": 0.0}
        self.cd_len = {"flick": FLICK_COOLDOWN, "shake": SHAKE_COOLDOWN, "still": 0.0}
        self.armed = {"flick": True, "shake": True, "still": True}   # a gesture re-arms once the window shows something else
        self.flick_due = None                                          # flick waits briefly: a shake starts like a flick

    def update(self, rows):
        out = []
        for r in rows:
            self.buf.append(r[3:9])
            self.count += 1
            if self.flick_due is not None and r[0] >= self.flick_due:
                out.append((r[0], "flick", "ml", 0.9))
                self.flick_due = None
            if len(self.buf) < self.win or self.count % GESTURE_STRIDE:
                continue
            p = self.model.predict_proba(window_features(np.array(self.buf))[None, :])[0]
            k = int(np.argmax(p))
            cls, prob, t = self.classes[k], float(p[k]), r[0]
            if prob > 0.6:
                for g in self.armed:
                    if g != cls:
                        self.armed[g] = True
            if cls in self.armed and prob >= ML_MIN_PROB:
                self.streak = self.streak + 1 if cls == self.pending else 1
                self.pending = cls
                if cls == "shake":
                    self.flick_due = None                              # it was the start of a shake, not a flick
                if self.streak >= 2 and self.armed[cls] and t >= self.cd[cls]:
                    self.armed[cls] = False
                    self.cd[cls] = t + self.cd_len[cls]
                    if cls == "flick":
                        self.flick_due = t + ML_FLICK_HOLD
                    else:
                        out.append((t, cls, "ml", prob))
            else:
                self.pending, self.streak = None, 0
        return out


def make_engine(kind="rules"):
    if kind == "ml":
        path = MODEL_DIR / "gesture_model.joblib"
        if path.exists():
            return MLEngine(path)
        print(f"[gestures] {path} not found -> falling back to rule engine (run: python train_model.py gesture)")
    return RuleEngine()
