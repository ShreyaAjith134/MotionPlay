"""Quick end-to-end check (30-60 s). Uses a temp folder, does NOT touch your database or shipped models.
    python selftest.py
"""
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

here = pathlib.Path(__file__).resolve().parent
tmp = pathlib.Path(tempfile.mkdtemp())
env = dict(os.environ, MOTIONPLAY_MODEL_DIR=str(tmp / "models"), SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy")
fails = []


def step(name, args, expect=None):
    r = subprocess.run([sys.executable, *args], cwd=here, env=env, capture_output=True, text=True)
    ok = r.returncode == 0 and (expect is None or expect in r.stdout)
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        fails.append(name)
        print(r.stdout[-600:], r.stderr[-900:])


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


db = str(tmp / "t.db")
step("demo data (2 players x 4 sessions)", ["demo_data.py", "--db", db, "--sessions", "4", "--players", "alice,bob"], "session")
step("analysis dashboard", ["analyze.py", "--db", db, "--out", str(tmp / "out")], "Saved")
check("dashboard.png written", (tmp / "out" / "dashboard.png").exists())
step("train gesture model (synthetic)", ["train_model.py", "gesture", "--synthetic", "--n", "250"], "saved models/gesture_model")
step("train fatigue model", ["train_model.py", "fatigue", "--db", db], "saved models/fatigue_model")
step("train player model", ["train_model.py", "player", "--db", db], "saved models/player_model")
step("live dashboard frame", ["live_dashboard.py", "--db", db, "--save-frame", str(tmp / "frame.png")], "saved")

sys.path.insert(0, str(here))
from arena import Arena
from gestures import RuleEngine
from synth import imu_from_tilt, smooth_tilt

A = Arena(800, seed=1)
A.reset(0.0)
for i in range(60 * 30):
    A.update(i / 60, 1 / 60, 400)
check("arena runs 30 s and spawns orbs", A.spawned > 15)
ok, _ = A.gesture("flick", 31.0)
check("flick gesture applies", ok)

rng = np.random.default_rng(1)
ts = np.arange(0, 20, 0.02)
imu = imu_from_tilt(ts, smooth_tilt(ts, rng), rng, 0.01, [(5, "flick"), (11.8, "shake"), (17, "still")])
ev = RuleEngine().update([(ts[i], i, 0, *imu[i], 0) for i in range(len(ts))])
got = {e[1] for e in ev}
check("rule engine finds flick+shake+still", {"flick", "shake", "still"} <= got)
print("\nALL GOOD" if not fails else f"\n{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
