"""Window features shared by the gesture classifier (and anything else that looks at raw IMU windows)."""
import numpy as np

from config import SAMPLE_HZ


def reversal_count(gyro, thr=60.0):
    """Direction changes of the dominant gyro axis (ignoring samples slower than thr deg/s)."""
    gyro = np.asarray(gyro, float)
    if len(gyro) < 3:
        return 0
    k = int(np.argmax(gyro.std(axis=0)))
    x = gyro[:, k]
    x = x[np.abs(x) > thr]
    if len(x) < 2:
        return 0
    s = np.sign(x)
    return int(np.sum(s[1:] != s[:-1]))


FEATURE_NAMES = [f"{c}_{s}" for c in ("ax", "ay", "az", "gx", "gy", "gz") for s in ("mean", "std", "min", "max")] + [
    "gm_mean", "gm_max", "gm_rms", "acc_dev_rms", "acc_dev_max", "peaks150", "reversals", "jerk_rms",
    "dom_freq", "dom_ratio"]


def window_features(w, fs=SAMPLE_HZ):
    """w: (n, 6) array [ax ay az gx gy gz] -> feature vector (len(FEATURE_NAMES))."""
    w = np.asarray(w, float)
    f = []
    for c in range(6):
        col = w[:, c]
        f += [col.mean(), col.std(), col.min(), col.max()]
    acc, gyr = w[:, :3], w[:, 3:]
    gm = np.linalg.norm(gyr, axis=1)
    dev = np.abs(np.linalg.norm(acc, axis=1) - 1.0)
    above = gm > 150
    peaks = int(np.sum(above[1:] & ~above[:-1])) + int(above[0])
    jerk = np.linalg.norm(np.diff(acc, axis=0), axis=1) * fs if len(w) > 1 else np.zeros(1)
    spec = np.abs(np.fft.rfft(gm - gm.mean()))
    if len(spec) > 1:
        k = int(np.argmax(spec[1:])) + 1
        dom_freq, dom_ratio = k * fs / len(w), spec[k] / (spec[1:].sum() + 1e-9)
    else:
        dom_freq = dom_ratio = 0.0
    f += [gm.mean(), gm.max(), float(np.sqrt(np.mean(gm ** 2))), float(np.sqrt(np.mean(dev ** 2))), dev.max(),
          peaks, reversal_count(gyr), float(np.sqrt(np.mean(jerk ** 2))), dom_freq, dom_ratio]
    return np.array(f, float)


def trial_windows(label, arr, win, stride=5, jitter=(-8, -4, 0, 4, 8)):
    """Cut a recorded trial (n,6) into labelled training windows.
    flick trials -> windows centred on the gyro peak (so every window really contains the flick);
    other labels  -> sliding windows."""
    arr = np.asarray(arr, float)
    out = []
    if len(arr) < win:
        return out
    if label == "flick":
        c = int(np.argmax(np.linalg.norm(arr[:, 3:], axis=1)))
        for j in jitter:
            s = c - win // 2 + j
            if 0 <= s <= len(arr) - win:
                out.append(arr[s:s + win])
    else:
        for s in range(0, len(arr) - win + 1, stride):
            out.append(arr[s:s + win])
    return out
