"""Live dashboard: run it in a second terminal WHILE you play. Refreshes every second from the SQLite file.

    python live_dashboard.py                       # follows the newest session (2 in versus mode)
    python live_dashboard.py --db motionplay_demo.db --save-frame frame.png     # one frame to a PNG (no window)
"""
import argparse

import matplotlib
import numpy as np
import pandas as pd

import db

PAL = ["#6c5ce7", "#e17055"]


def latest_sessions(conn):
    s = pd.read_sql("SELECT * FROM sessions ORDER BY id DESC LIMIT 2", conn)
    if s.empty:
        return s
    if s.iloc[0].mode != "versus":
        s = s.iloc[:1]
    return s.sort_values("slot")


def draw(fig, axs, conn):
    for a in axs.flat:
        a.clear()
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    s = latest_sessions(conn)
    if s.empty:
        axs[0, 0].text(0.1, 0.5, "waiting for a session...", transform=axs[0, 0].transAxes)
        return
    gnames = ["flick", "shake", "still"]
    for k, row in enumerate(s.itertuples()):
        ev = pd.read_sql("SELECT * FROM events WHERE session_id=? ORDER BY t_spawn", conn, params=(row.id,))
        gs = pd.read_sql("SELECT * FROM gestures WHERE session_id=? AND applied=1", conn, params=(row.id,))
        col = PAL[k % 2]
        sk = ev[ev.kind.isin(["good", "gold"]) & ev.outcome.isin(["caught", "missed"])]
        if len(sk):
            axs[0, 0].plot(np.arange(len(sk)), sk.ok.rolling(10, min_periods=3).mean() * 100, color=col, lw=2, label=row.player)
        if len(ev):
            axs[0, 1].plot(np.arange(len(ev)), ev.speed_mult, color=col, lw=2, label=row.player)
            axs[1, 0].step(np.arange(len(ev)), ev.score, color=col, lw=2, label=row.player)
        counts = [int((gs.gesture == g).sum()) if len(gs) else 0 for g in gnames]
        axs[1, 1].bar(np.arange(3) + k * 0.38, counts, 0.36, color=col, label=row.player)
        last = ev.iloc[-1] if len(ev) else None
        txt = (f"{row.player}: score {int(last.score)}  lives {int(last.lives)}  level {int(last.level)}  combo {int(last.combo)}"
               if last is not None else f"{row.player}: starting...")
        fig.text(0.02, 0.965 - k * 0.03, txt + ("   (running)" if pd.isna(row.ended) else "   (finished)"), color=col,
                 fontsize=12, fontweight="bold")
    axs[0, 0].set(title="Rolling catch rate % (last 10 orbs)", ylim=(0, 105), xlabel="orb #")
    axs[0, 1].set(title="Adaptive difficulty (speed x)", ylim=(0.5, 1.9), xlabel="orb #")
    axs[1, 0].set(title="Score", xlabel="orb #")
    axs[1, 1].set(title="Power-ups used", xticks=np.arange(3) + (len(s) - 1) * 0.19, xticklabels=["flick\n(pulse)", "shake\n(bomb)", "still\n(shield)"])
    for a in (axs[0, 0], axs[0, 1], axs[1, 0]):
        if len(s) > 1:
            a.legend(frameon=False, fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(db.DB_PATH))
    ap.add_argument("--save-frame", default=None)
    a = ap.parse_args()
    if a.save_frame:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    conn = db.connect(a.db)
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5))
    fig.subplots_adjust(top=0.86, hspace=0.35, wspace=0.25)
    if a.save_frame:
        draw(fig, axs, conn)
        fig.savefig(a.save_frame, dpi=110)
        print("saved", a.save_frame)
        return
    ani = FuncAnimation(fig, lambda _: (fig.texts.clear(), draw(fig, axs, conn)), interval=1000, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
