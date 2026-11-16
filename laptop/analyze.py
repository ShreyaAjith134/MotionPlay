"""ANALYZE stage: learning curve, smoothness, reaction time, fatigue, gesture usage, playstyle clusters, difficulty.

    python analyze.py                                    # every finished session in motionplay.db
    python analyze.py --db motionplay_demo.db            # synthetic bots
    python analyze.py --player shreya --out analysis_out

Outputs: <out>/dashboard.png, session_summary.csv, report.json
"""
import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import db
from analysis_lib import *
from config import *

PAL = ["#6c5ce7", "#00b894", "#e17055", "#0984e3", "#fdcb6e", "#d63031"]


def cluster_playstyles(w, k=3):
    feats = ["jerk_rms", "gyro_rms", "paddle_speed", "catch_rate", "rt_mean"]
    w = w.dropna(subset=feats).copy()
    k = min(k, len(w))
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(StandardScaler().fit_transform(w[feats]))
    w["cluster"] = km.labels_
    order = w.groupby("cluster").jerk_rms.mean().sort_values().index.tolist()      # calm -> wild
    names = {1: ["Smooth / Precise"], 2: ["Smooth / Precise", "Aggressive"]}.get(k, ["Smooth / Precise", "Balanced", "Aggressive"])
    w["playstyle"] = w.cluster.map({c: names[i] for i, c in enumerate(order)})
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(db.DB_PATH))
    ap.add_argument("--player", default=None)
    ap.add_argument("--out", default="analysis_out")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(exist_ok=True)

    d = load(db.connect(a.db), a.player)
    summ = session_table(d)
    wins = window_table(d)
    summ["n"] = range(1, len(summ) + 1)
    players = list(dict.fromkeys(summ.player))
    pcol = {p: PAL[i % len(PAL)] for i, p in enumerate(players)}

    # ---- learning curve (chronological, skill events only)
    sk = skill_events(d["events"]).sort_values("t_spawn").reset_index(drop=True)
    sk["n"] = np.arange(1, len(sk) + 1)
    sk["roll"] = sk.ok.rolling(20, min_periods=5).mean()
    lr = stats.linregress(sk.n, sk.ok)
    slope100 = lr.slope * 100 * 100

    clusters = None
    if len(wins) >= 3:
        clusters = cluster_playstyles(wins)
        dom = clusters.groupby("session").playstyle.agg(lambda s: s.value_counts().idxmax())
        summ["dominant_style"] = summ.session.map(dom)
    summ.round(3).to_csv(out / "session_summary.csv", index=False)

    # ---- dashboard
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    fig, ax = plt.subplots(3, 3, figsize=(18, 13))
    fig.suptitle("MotionPlay - player insights", fontsize=17, fontweight="bold")

    for p in players:
        g = sk[sk.session_id.isin(summ[summ.player == p].session)] if len(players) > 1 else sk
        ax[0, 0].plot(g.n, g.roll * 100, lw=1.6, color=pcol[p], label=p)
    ax[0, 0].plot(sk.n, (lr.intercept + lr.slope * sk.n) * 100, "--", color="#2d3436")
    ax[0, 0].set(title=f"Learning curve  ({slope100:+.1f} pp / 100 orbs, p={lr.pvalue:.3g})", xlabel="orb #",
                 ylabel="rolling catch rate % (20)")
    if len(players) > 1:
        ax[0, 0].legend(frameon=False, fontsize=8)

    ax[0, 1].bar(summ.n, summ.score, color=[pcol[p] for p in summ.player])
    ax[0, 1].set(title="Score per session", xlabel="session #", ylabel="points")
    ax[0, 2].bar(summ.n, summ.rt_median * 1000, color=[pcol[p] for p in summ.player])
    ax[0, 2].set(title="Reaction time (median)", xlabel="session #", ylabel="ms")

    ax[1, 0].bar(summ.n - 0.2, summ.jerk_rms, 0.4, color="#e17055", label="jerk (g/s)")
    ax2 = ax[1, 0].twinx()
    ax2.bar(summ.n + 0.2, summ.gyro_rms, 0.4, color="#fdcb6e", label="gyro (deg/s)")
    ax2.spines["top"].set_visible(False)
    ax[1, 0].set(title="Control smoothness (lower = smoother)", xlabel="session #", ylabel="jerk g/s")
    ax2.set_ylabel("gyro deg/s")

    fv = [summ.fatigue_rt_pct.mean(), summ.fatigue_tremor_pct.mean(), -summ.fatigue_catch_pp.mean()]
    ax[1, 1].bar(["reaction time\n(% slower)", "hand tremor\n(% more)", "catch rate\n(pp drop)"], fv, color=["#a29bfe", "#d63031", "#00b894"])
    ax[1, 1].axhline(0, color="#636e72", lw=1)
    ax[1, 1].set(title="Fatigue: last third vs first third of a session", ylabel="change (positive = fatigue)")

    if clusters is not None:
        pc = {"Smooth / Precise": "#00b894", "Balanced": "#0984e3", "Aggressive": "#d63031"}
        for name, g in clusters.groupby("playstyle"):
            ax[1, 2].scatter(g.jerk_rms, g.paddle_speed, s=45, alpha=.8, color=pc.get(name), label=f"{name} (n={len(g)})")
        ax[1, 2].legend(frameon=False)
    ax[1, 2].set(title="Playstyle clusters (10 s windows, k-means)", xlabel="jerk RMS (g/s)", ylabel="paddle speed px/s")

    ax[2, 0].bar(summ.n, summ.flicks, color="#a29bfe", label="flick = pulse")
    ax[2, 0].bar(summ.n, summ.shakes, bottom=summ.flicks, color="#ff8c42", label="shake = bomb")
    ax[2, 0].bar(summ.n, summ.stills, bottom=summ.flicks + summ.shakes, color="#6eaaff", label="still = shield")
    ax[2, 0].set(title="Gesture power-ups used (applied)", xlabel="session #", ylabel="count")
    ax[2, 0].legend(frameon=False, fontsize=8)

    ev = d["events"].sort_values("t_spawn")
    for p in players:
        g = ev[ev.session_id.isin(summ[summ.player == p].session)]
        ax[2, 1].plot(np.arange(len(g)), g.speed_mult.rolling(15, min_periods=1).mean(), color=pcol[p], lw=1.4, label=p)
    ax[2, 1].set(title="Adaptive difficulty multiplier over time", xlabel="orb #", ylabel="speed x")
    if len(players) > 1:
        ax[2, 1].legend(frameon=False, fontsize=8)

    b = summ[["bad_hit_rate"]].copy()
    ax[2, 2].bar(summ.n, summ.catch_rate * 100, color="#00b894", label="good/gold caught %")
    ax[2, 2].plot(summ.n, b.bad_hit_rate * 100, "o-", color="#d63031", label="red orbs hit %")
    ax[2, 2].set(title="Catch rate vs red-orb mistakes", xlabel="session #", ylabel="%")
    ax[2, 2].legend(frameon=False, fontsize=8)

    for a_ in ax.flat:
        if "session" in a_.get_xlabel():
            a_.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(out / "dashboard.png", dpi=120)

    report = dict(
        sessions=len(summ), players=players, orbs=int(len(d["events"])),
        overall_catch_rate=round(float(sk.ok.mean()), 3),
        learning_slope_pp_per_100_orbs=round(float(slope100), 2), learning_p_value=round(float(lr.pvalue), 4),
        reaction_time_median_ms=round(float(skill_events(d["events"]).rt.median() * 1000), 1),
        smoothness_jerk_rms_mean=round(float(summ.jerk_rms.mean()), 2),
        fatigue_rt_change_pct=round(float(summ.fatigue_rt_pct.mean()), 1),
        fatigue_catch_change_pp=round(float(summ.fatigue_catch_pp.mean()), 1),
        fatigue_tremor_change_pct=round(float(summ.fatigue_tremor_pct.mean()), 1),
        gestures_applied=dict(flick=int(summ.flicks.sum()), shake=int(summ.shakes.sum()), still=int(summ.stills.sum())),
        playstyle_windows=clusters.playstyle.value_counts().to_dict() if clusters is not None else {})
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nSaved: {out / 'dashboard.png'}, {out / 'session_summary.csv'}, {out / 'report.json'}")


if __name__ == "__main__":
    main()
