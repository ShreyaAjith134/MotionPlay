"""Generate demo data: three virtual players (bots) play the REAL game logic (arena.py) and their sensor
streams are synthesised. Lets you run analyze.py / train_model.py without playing.

    python demo_data.py                           # 3 players x 6 sessions -> motionplay_demo.db
    python demo_data.py --sessions 4 --db my.db

alice = smooth / careful, bob = aggressive / shaky, cara = slow / cautious.
Each bot improves with practice (learning curve) and gets sloppier within a session (fatigue).
"""
import argparse
import math
import random

import numpy as np

import db
from arena import Arena
from config import *
from synth import imu_from_tilt

PERSONAS = {
    "alice": dict(gain=5.0, vmax=520, tremor=0.006, rt=0.38, aim=18, dodge=0.85, g_rt=0.45, pulse=0.9, bomb=0.8, shield=0.8),
    "bob": dict(gain=14.0, vmax=1100, tremor=0.028, rt=0.30, aim=45, dodge=0.55, g_rt=0.30, pulse=0.7, bomb=1.0, shield=0.2),
    "cara": dict(gain=3.0, vmax=330, tremor=0.010, rt=0.55, aim=12, dodge=0.90, g_rt=0.80, pulse=0.6, bomb=0.5, shield=1.0),
}
SPAN = ARENA_W_SOLO / 2 - PADDLE_W / 2


def play(p, skill, rng, seed):
    A = Arena(ARENA_W_SOLO, seed=seed, adaptive=True)
    A.reset(0.0)
    x, vx, dt, t = A.w / 2, 0.0, 1 / 60, 0.0
    info, events, gests, frames = {}, [], [], []
    while t < GAME_SECONDS and not A.over:
        fat = 1 + 0.6 * t / GAME_SECONDS
        for o in A.orbs:
            if o.idx not in info:
                rt = max(0.15, rng.gauss(p["rt"] * (1.5 - 0.8 * skill) * fat, 0.05))
                info[o.idx] = (t + rt, rng.gauss(0, p["aim"] * (1.9 - 1.3 * skill) * fat), rng.random() < p["dodge"] * (0.6 + 0.4 * skill))
        # --- gestures (reaction-delayed)
        react = lambda k: (1 - math.exp(-dt / (p["g_rt"] * (1.3 - 0.4 * skill) * fat))) * k
        todo = []
        if any(o.kind == "bad" and o.y > ARENA_H * 0.35 for o in A.orbs) and t >= A.pulse_ready and rng.random() < react(p["pulse"]):
            todo.append("flick")
        if A.bombs > 0 and sum(o.kind != "bad" for o in A.orbs) >= 3 and rng.random() < react(p["bomb"]):
            todo.append("shake")
        if t >= A.shield_ready and not A.shield_active and rng.random() < dt * 0.2 * p["shield"]:
            todo.append("still")
        for g in todo:
            ok, evs = A.gesture(g, t)
            gests.append((t, g, "bot", 1.0, int(ok)))
            events += evs
        # --- movement
        desired = x
        chase = [o for o in A.orbs if o.kind != "bad" and t >= info[o.idx][0]]
        if chase:
            o = max(chase, key=lambda o: o.y)
            desired = o.x + info[o.idx][1]
        for o in A.orbs:
            if o.kind == "bad" and o.y > ARENA_H * 0.45 and t >= info[o.idx][0] and info[o.idx][2] \
                    and abs(o.x - x) < PADDLE_W / 2 + 40:
                desired = x + (-1 if o.x > x else 1) * (PADDLE_W / 2 + 70)
        want = max(-p["vmax"], min(p["vmax"], (desired - x) * p["gain"]))
        vx += (want - vx) * 0.25
        x = max(PADDLE_W / 2, min(A.w - PADDLE_W / 2, x + vx * dt))
        events += A.update(t, dt, x)
        frames.append((t, x))
        t += dt
    return A, events, gests, frames, t


def store(conn, name, p, skill, A, events, gests, frames, t_end, t0, rng):
    sid = db.new_session(conn, name, t0, "solo", 0, "bot", 1)
    ft = np.array([f[0] for f in frames])
    fx = np.array([f[1] for f in frames])
    ts = np.arange(0, t_end, 1 / SAMPLE_HZ)
    tilt = np.interp(ts, ft, (fx - A.w / 2) / SPAN * TILT_RANGE_G) * TILT_SIGN
    tremor = p["tremor"] * (1.3 - 0.5 * skill) * (1 + 0.6 * ts / GAME_SECONDS)
    nprng = np.random.default_rng(rng.randrange(1 << 30))
    imu = imu_from_tilt(ts, tilt, nprng, tremor, [(g[0], g[1]) for g in gests])
    db.insert_samples(conn, [(sid, i, int(ts[i] * 1000), t0 + ts[i], *map(float, imu[i]), 0) for i in range(len(ts))])
    db.insert_events(conn, [db.event_row(sid, e, t0) for e in events])
    db.insert_frames(conn, [(sid, t0 + f[0], f[1]) for f in frames])
    db.insert_gestures(conn, [(sid, t0 + g[0], g[1], g[2], g[3], g[4]) for g in gests])
    db.end_session(conn, sid, t0 + t_end, A.score, A.level, A.lives)
    return sid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(db.DB_PATH.with_name("motionplay_demo.db")))
    ap.add_argument("--sessions", type=int, default=6, help="sessions per player")
    ap.add_argument("--players", default="alice,bob,cara")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    conn = db.connect(a.db)
    base = 1_760_000_000.0
    names = [n.strip() for n in a.players.split(",")]
    print(f"writing {a.sessions} sessions x {len(names)} players to {a.db}")
    for s in range(a.sessions):
        skill = s / max(1, a.sessions - 1)
        for k, name in enumerate(names):
            p = PERSONAS[name]
            A, ev, gs, fr, tend = play(p, skill, rng, rng.randrange(1 << 30))
            sid = store(conn, name, p, skill, A, ev, gs, fr, tend, base + (s * 10 + k) * 3600, rng)
            print(f"  session {sid:2d} {name:6s} skill={skill:.2f}  score={A.score:5d}  level={A.level:2d}  lives={A.lives}  "
                  f"caught={A.stats['caught']:3d} missed={A.stats['missed']:3d}  gestures={len(gs)}")
    conn.close()


if __name__ == "__main__":
    main()
