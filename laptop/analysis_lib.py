"""Shared data loading + feature engineering for analyze.py and train_model.py."""
import numpy as np
import pandas as pd

from config import *

SKILL_KINDS = ("good", "gold")
SKILL_OUTCOMES = ("caught", "missed")        # bombs / pulses are not "skill" events


# ------------------------------------------------------------------ loading
def load(conn, player=None):
    q, params = "SELECT * FROM sessions WHERE ended IS NOT NULL", ()
    if player:
        q, params = q + " AND player=?", (player,)
    sessions = pd.read_sql(q + " ORDER BY started, id", conn, params=params)
    if sessions.empty:
        raise SystemExit("No finished sessions found. Play a game first (or run demo_data.py).")
    ids = ",".join(str(i) for i in sessions.id)
    rd = lambda t, o: pd.read_sql(f"SELECT * FROM {t} WHERE session_id IN ({ids}) ORDER BY {o}", conn)
    d = dict(sessions=sessions, samples=rd("samples", "session_id, t_host"), events=rd("events", "session_id, t_spawn"),
             frames=rd("frames", "session_id, t"), gestures=rd("gestures", "session_id, t"))
    return prep(d)


def prep(d):
    d["events"] = add_reaction_times(d["events"], d["frames"])
    d["samples"] = add_motion_features(d["samples"])
    return d


def add_reaction_times(events, frames):
    """Reaction time = spawn -> paddle first moves MOVE_PX toward a good/gold orb."""
    rts = np.full(len(events), np.nan)
    pos = {sid: g for sid, g in frames.groupby("session_id")}
    for n, r in enumerate(events.itertuples()):
        if r.kind not in SKILL_KINDS or r.outcome not in SKILL_OUTCOMES:
            continue
        g = pos.get(r.session_id)
        if g is None:
            continue
        d = r.orb_x - r.paddle_x_spawn
        if abs(d) < NEAR_PX:
            continue
        t, x = g.t.values, g.paddle_x.values
        i0, i1 = np.searchsorted(t, r.t_spawn), np.searchsorted(t, r.t_spawn + RT_MAX)
        if i0 >= len(t):
            continue
        j = np.searchsorted(t, r.t_spawn - 0.15)
        if t[i0] > t[j] and abs(x[i0] - x[j]) / (t[i0] - t[j]) > PRE_SPEED_PX_S:
            continue                                   # paddle was already moving (chasing the previous orb)
        moved = np.where((x[i0:i1] - r.paddle_x_spawn) * np.sign(d) > MOVE_PX)[0]
        if len(moved):
            rt = t[i0 + moved[0]] - r.t_spawn
            if rt >= MIN_RT:
                rts[n] = rt
    events = events.copy()
    events["rt"] = rts
    return events


def add_motion_features(samples):
    """jerk = RMS derivative of lightly filtered acceleration (g/s), gmag = gyro magnitude (deg/s)."""
    out = []
    for sid, g in samples.groupby("session_id"):
        g = g.sort_values("t_dev_ms").copy()
        a = g[["ax", "ay", "az"]].rolling(5, center=True, min_periods=1).mean().values
        t = g.t_dev_ms.values / 1000.0
        dt = np.diff(t)
        ok = (dt > 0) & (dt < 0.2)
        jerk = np.full(len(g), np.nan)
        jerk[1:][ok] = np.linalg.norm(np.diff(a, axis=0)[ok], axis=1) / dt[ok]
        g["jerk"] = jerk
        g["gmag"] = np.linalg.norm(g[["gx", "gy", "gz"]].values, axis=1)
        out.append(g)
    return pd.concat(out) if out else samples.assign(jerk=np.nan, gmag=np.nan)


def skill_events(ev):
    return ev[ev.kind.isin(SKILL_KINDS) & ev.outcome.isin(SKILL_OUTCOMES)]


def _rms(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return float(np.sqrt(np.mean(v ** 2))) if len(v) else np.nan


def _thirds(t, t0, t1):
    return np.clip(np.digitize(t, np.linspace(t0, t1, 4)[1:-1]), 0, 2)


# ------------------------------------------------------------------ per-session table
def session_table(d):
    rows = []
    for s in d["sessions"].itertuples():
        ev = d["events"][d["events"].session_id == s.id].copy()
        sm = d["samples"][d["samples"].session_id == s.id]
        gs = d["gestures"][d["gestures"].session_id == s.id]
        sk = skill_events(ev)
        if len(sk) < 5 or sm.empty:
            continue
        t0, t1 = ev.t_spawn.min(), ev.t_resolve.max()
        sk = sk.assign(third=_thirds(sk.t_spawn.values, t0, t1))
        sm = sm.assign(third=_thirds(sm.t_host.values, t0, t1))
        rt3 = sk.groupby("third").rt.median().reindex([0, 1, 2])
        ok3 = sk.groupby("third").ok.mean().reindex([0, 1, 2])
        tr3 = sm.groupby("third").gmag.apply(_rms).reindex([0, 1, 2])
        bad = ev[ev.kind == "bad"]
        applied = gs[gs.applied == 1].gesture.value_counts() if len(gs) else pd.Series(dtype=int)
        pct = lambda a, b: (a - b) / b * 100 if b else np.nan
        rows.append(dict(
            session=s.id, player=s.player, started=s.started, mode=s.mode, score=s.score, level=s.level,
            lives_left=s.lives_left, orbs=len(ev), catch_rate=sk.ok.mean(),
            bad_hit_rate=float((bad.outcome == "caught").mean()) if len(bad) else np.nan,
            rt_median=sk.rt.median(), jerk_rms=_rms(sm.jerk), gyro_rms=_rms(sm.gmag),
            diff_mean=ev.speed_mult.mean(), best_combo=int(ev.combo.max()),
            flicks=int(applied.get("flick", 0)), shakes=int(applied.get("shake", 0)), stills=int(applied.get("still", 0)),
            fatigue_rt_pct=pct(rt3[2], rt3[0]), fatigue_catch_pp=(ok3[2] - ok3[0]) * 100,
            fatigue_tremor_pct=pct(tr3[2], tr3[0])))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 10-second windows
WINDOW_FEATURES = ["jerk_rms", "gyro_rms", "paddle_speed", "catch_rate", "rt_mean", "diff_mean", "level_mean", "n_gest"]


def window_table(d):
    """One row per WINDOW_SEC window per session: movement features + performance (used for clustering,
    player identification and the fatigue/performance predictor)."""
    wins = []
    for s in d["sessions"].itertuples():
        ev = d["events"][d["events"].session_id == s.id]
        sm = d["samples"][d["samples"].session_id == s.id]
        fr = d["frames"][d["frames"].session_id == s.id].copy()
        gs = d["gestures"][d["gestures"].session_id == s.id]
        sk = skill_events(ev)
        if len(sk) < 5 or sm.empty:
            continue
        t0 = ev.t_spawn.min()
        fr["speed"] = np.abs(fr.paddle_x.diff()) / fr.t.diff()
        W = lambda t: ((t - t0) // WINDOW_SEC).astype(int)
        ev = ev.assign(w=W(ev.t_spawn))
        sk = sk.assign(w=W(sk.t_spawn))
        sm = sm.assign(w=W(sm.t_host))
        fr = fr.assign(w=W(fr.t))
        gw = W(gs.t) if len(gs) else pd.Series(dtype=int)
        for w in sorted(set(sk.w) & set(sm.w)):
            e, m, f = sk[sk.w == w], sm[sm.w == w], fr[fr.w == w]
            if len(e) < 3 or len(m) < 20:
                continue
            wins.append(dict(session=s.id, player=s.player, win=int(w), elapsed=w * WINDOW_SEC,
                             jerk_rms=_rms(m.jerk), gyro_rms=_rms(m.gmag), paddle_speed=float(np.nanmean(f.speed)),
                             catch_rate=e.ok.mean(), rt_mean=e.rt.mean(), n_skill=len(e),
                             diff_mean=ev[ev.w == w].speed_mult.mean(), level_mean=ev[ev.w == w].level.mean(),
                             n_gest=int((gw == w).sum()) if len(gs) else 0))
    wt = pd.DataFrame(wins)
    if wt.empty:
        return wt
    # fill missing reaction times with the session median so every window has the feature
    wt["rt_mean"] = wt.groupby("session").rt_mean.transform(lambda v: v.fillna(v.median()))
    wt["rt_mean"] = wt.rt_mean.fillna(wt.rt_mean.median())
    return wt
