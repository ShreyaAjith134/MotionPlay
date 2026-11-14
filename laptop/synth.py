"""Synthetic IMU generator.

Used for: the fake ESP32 (simulator.py), demo data (demo_data.py) and bootstrapping the gesture
classifier before you have collected real recordings. Real data (collect_gestures.py) always beats this.
"""
import numpy as np

from config import *


def smooth_tilt(ts, rng, amp=None, f=None):
    amp = amp if amp is not None else rng.uniform(0.2, 0.5)
    f = f if f is not None else rng.uniform(0.15, 1.0)
    return (amp * np.sin(2 * np.pi * f * ts + rng.uniform(0, 6.28))
            + 0.12 * amp * np.sin(2 * np.pi * 2.3 * f * ts + rng.uniform(0, 6.28)))


def imu_from_tilt(ts, tilt, rng, tremor=0.01, gestures=(), axis=None):
    """ts (n,) seconds, tilt (n,) in g.  gestures = [(t, 'flick'|'shake'|'still'), ...]
    flick: gyro pulse AT t;  shake / still: oscillation / calm in the window ENDING at t."""
    axis = TILT_AXIS if axis is None else axis
    n = len(ts)
    tl = np.array(tilt, float)
    tremor = np.broadcast_to(np.asarray(tremor, float), (n,)).copy()
    noise_acc = 0.004 + tremor
    noise_gyr = 1.0 + 40 * tremor
    orig = tl.copy()
    for tg, kind in gestures:
        if kind == "still":
            m = (ts >= tg - STILL_SEC - 0.3) & (ts <= tg)
            if m.any():
                i0 = int(np.argmax(m))
                held = orig[i0] + (orig[m] - orig[i0]) * np.exp(-(ts[m] - ts[i0]) / 0.08)
                d_end = held[-1] - orig[m][-1]
                tl[m] = held
                after = (ts > tg) & (ts <= tg + 1.0)           # glide back, no step -> no fake gyro spike
                tl[after] = orig[after] + d_end * np.exp(-(ts[after] - tg) / 0.15)
                noise_acc[m] = 0.003
                noise_gyr[m] = 0.8
    gyro_tilt = np.gradient(tl, ts) * 57.3 * 0.5
    acc = np.zeros((n, 3))
    gyr = np.zeros((n, 3))
    acc[:, 2] = 0.98
    acc[:, axis] += tl
    acc += rng.normal(0, noise_acc[:, None], (n, 3))
    gyr[:, 0] = gyro_tilt
    gyr += rng.normal(0, noise_gyr[:, None], (n, 3))
    for tg, kind in gestures:
        if kind == "flick":
            k = int(rng.integers(0, 3))
            amp = rng.uniform(450, 900) * rng.choice([-1, 1])
            pulse = np.exp(-((ts - tg) / 0.045) ** 2)
            rebound = np.exp(-((ts - tg - 0.09) / 0.05) ** 2)
            gyr[:, k] += amp * pulse - 0.35 * amp * rebound
            acc[:, int(rng.integers(0, 3))] += rng.uniform(0.5, 1.5) * rng.choice([-1, 1]) * pulse
        elif kind == "shake":
            m = (ts >= tg - 0.8) & (ts <= tg)
            f = rng.uniform(3.0, 6.0)
            osc = np.sin(2 * np.pi * f * (ts - tg + 0.8))
            gyr[m, int(rng.integers(0, 3))] += rng.uniform(250, 600) * osc[m]
            acc[m, int(rng.integers(0, 3))] += rng.uniform(0.4, 1.2) * osc[m]
    return np.hstack([acc, gyr])


def make_window(label, rng, n=GESTURE_WINDOW, fs=SAMPLE_HZ):
    """One labelled training window (n, 6)."""
    pad = 30
    ts = np.arange(-pad, n) / fs
    tilt = smooth_tilt(ts, rng, amp=rng.uniform(0.05, 0.55), f=rng.uniform(0.1, 1.3))
    tremor = rng.uniform(0.003, 0.03)
    T = ts[-1]
    g = []
    if label == "flick":
        g = [(rng.uniform(0.15 * n / fs, 0.85 * n / fs), "flick")]
    elif label == "shake":
        # half the windows: shake already going; half: shake just started (onset 0.35-0.7 s ago) like in live play
        s_on = 0.8 if rng.random() < 0.5 else rng.uniform(0.35, 0.7)
        g = [(T + (0.8 - s_on) + (rng.uniform(0, 0.1) if s_on == 0.8 else 0.0), "shake")]
    elif label == "still":
        g = [(T, "still")]
    return imu_from_tilt(ts, tilt, rng, tremor, g)[-n:]


def make_dataset(n_per_class, rng):
    X, y = [], []
    for label in GESTURE_CLASSES:
        for _ in range(n_per_class):
            X.append(make_window(label, rng))
            y.append(label)
    return X, y
