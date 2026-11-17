"""Train the three ML models of MotionPlay.

    python train_model.py gesture                  # CSVs in data/gestures/ if present, else synthetic data
    python train_model.py gesture --synthetic      # force synthetic
    python train_model.py fatigue --db motionplay_demo.db
    python train_model.py player  --db motionplay_demo.db
    python train_model.py all     --db motionplay_demo.db

1. gesture : window of raw IMU (0.6 s) -> tilt | flick | shake | still        (used live with: game.py --engine ml)
2. fatigue : this 10 s window -> catch rate of the NEXT 10 s window            (performance / fatigue predictor)
3. player  : 10 s window of movement style -> which player is it              (needs >= 2 players in the DB)

Evaluation is honest: cross-validation groups by trial (gesture) or by SESSION (fatigue, player), so windows of
the same session never appear in both train and test.
Outputs: models/*.joblib, models/*_report.json, models/*.png
"""
import argparse
import glob
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score, mean_absolute_error,
                             r2_score)
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import db
from analysis_lib import load, window_table
from config import *
from features import FEATURE_NAMES, trial_windows, window_features
from synth import make_dataset

warnings.filterwarnings("ignore")
MODEL_DIR.mkdir(exist_ok=True)


def save_conf(cm, classes, title, path):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=30)
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(title=title, xlabel="predicted", ylabel="true")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------ 1. gesture classifier
def train_gesture(a):
    files = [] if a.synthetic else sorted(glob.glob(str(GESTURE_DATA_DIR / "*.csv")))
    if files:
        X, y, groups = [], [], []
        for fp in files:
            df = pd.read_csv(fp)
            for (label, trial), g in df.groupby(["label", "trial"]):
                for w in trial_windows(label, g[["ax", "ay", "az", "gx", "gy", "gz"]].values, GESTURE_WINDOW):
                    X.append(window_features(w))
                    y.append(label)
                    groups.append(f"{fp}:{trial}")
        X, y, groups, source = np.array(X), np.array(y), np.array(groups), f"REAL recordings ({len(files)} files)"
        cv = GroupKFold(n_splits=5)
    else:
        rng = np.random.default_rng(0)
        Xw, y = make_dataset(a.n, rng)
        X, y, groups = np.array([window_features(w) for w in Xw]), np.array(y), None
        source = "SYNTHETIC windows (record real ones with collect_gestures.py for a model that fits YOUR hand)"
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    print(f"gesture data: {len(y)} windows from {source}; classes {({k: int(v) for k, v in pd.Series(y).value_counts().items()})}")

    cands = {
        "random_forest": RandomForestClassifier(n_estimators=150, min_samples_leaf=2, random_state=0, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=0),
        "svm_rbf": make_pipeline(StandardScaler(), SVC(C=3, probability=True, random_state=0)),
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
    }
    res, preds = {}, {}
    for name, m in cands.items():
        p = cross_val_predict(m, X, y, cv=cv, groups=groups)
        preds[name] = p
        res[name] = dict(accuracy=round(accuracy_score(y, p), 4), macro_f1=round(f1_score(y, p, average="macro"), 4))
        print(f"  {name:24s} acc {res[name]['accuracy']:.3f}  macro-F1 {res[name]['macro_f1']:.3f}")
    best = max(res, key=lambda k: res[k]["macro_f1"])
    model = cands[best].fit(X, y)
    if hasattr(model, "set_params") and best == "random_forest":
        model.set_params(n_jobs=1)                      # live inference runs in the game loop
    classes = list(model.classes_)
    cm = confusion_matrix(y, preds[best], labels=classes)
    save_conf(cm, classes, f"gesture: {best} (CV)", MODEL_DIR / "gesture_confusion.png")
    print(f"best: {best}\n" + classification_report(y, preds[best], digits=3))
    joblib.dump(dict(model=model, classes=classes, window=GESTURE_WINDOW, features=FEATURE_NAMES, kind=best,
                     trained_on=source, cv=res), MODEL_DIR / "gesture_model.joblib")
    (MODEL_DIR / "gesture_report.json").write_text(json.dumps(dict(best=best, trained_on=source, cv=res), indent=2))
    print("saved models/gesture_model.joblib")


# ------------------------------------------------------------------ 2. fatigue / next-window performance
FATIGUE_FEATURES = ["elapsed", "jerk_rms", "gyro_rms", "paddle_speed", "catch_rate", "rt_mean", "diff_mean", "level_mean", "n_gest"]


def train_fatigue(a):
    wt = window_table(load(db.connect(a.db)))
    wt = wt.sort_values(["session", "win"])
    wt["next_catch"] = wt.groupby("session").catch_rate.shift(-1)
    wt["next_win"] = wt.groupby("session").win.shift(-1)
    wt = wt[(wt.next_win == wt.win + 1)].dropna(subset=FATIGUE_FEATURES + ["next_catch"])
    if wt.session.nunique() < 4:
        print("fatigue model: need at least 4 sessions - play more (or run demo_data.py)")
        return
    X, y, g = wt[FATIGUE_FEATURES].values, wt.next_catch.values, wt.session.values
    cv = GroupKFold(n_splits=min(5, wt.session.nunique()))
    cands = {
        "baseline: mean": DummyRegressor(),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=3.0)),
        "random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, random_state=0),
    }
    res, preds = {}, {}
    persist = wt.catch_rate.values                                   # "next window = this window"
    res["baseline: persistence"] = dict(mae=round(mean_absolute_error(y, persist), 4), r2=round(r2_score(y, persist), 4))
    for n, m in cands.items():
        p = cross_val_predict(m, X, y, cv=cv, groups=g)
        preds[n] = p
        res[n] = dict(mae=round(mean_absolute_error(y, p), 4), r2=round(r2_score(y, p), 4))
    for n, r in res.items():
        print(f"  {n:26s} MAE {r['mae']:.3f}  R2 {r['r2']:+.3f}")
    best = min((k for k in cands if not k.startswith("baseline")), key=lambda k: res[k]["mae"])
    model = cands[best].fit(X, y)
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=-1).fit(X, y)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].scatter(y * 100, preds[best] * 100, s=14, alpha=.6, color="#6c5ce7")
    ax[0].plot([0, 100], [0, 100], "--", color="#636e72")
    ax[0].set(title=f"Next-window catch rate: {best} (session-held-out CV)\nMAE {res[best]['mae']*100:.1f} pp, R2 {res[best]['r2']:+.2f}",
              xlabel="actual %", ylabel="predicted %")
    order = np.argsort(rf.feature_importances_)
    ax[1].barh(np.array(FATIGUE_FEATURES)[order], rf.feature_importances_[order], color="#00b894")
    ax[1].set(title="What predicts the next 10 s? (random-forest importance)")
    for a_ in ax:
        a_.spines["top"].set_visible(False)
        a_.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "fatigue_eval.png", dpi=120)
    plt.close(fig)
    joblib.dump(dict(model=model, features=FATIGUE_FEATURES, target="catch rate of the next 10 s window", kind=best, cv=res),
                MODEL_DIR / "fatigue_model.joblib")
    (MODEL_DIR / "fatigue_report.json").write_text(json.dumps(dict(best=best, cv=res, windows=int(len(wt)),
                                                                   sessions=int(wt.session.nunique())), indent=2))
    print(f"best: {best}   saved models/fatigue_model.joblib")


# ------------------------------------------------------------------ 3. player identification
PLAYER_FEATURES = ["jerk_rms", "gyro_rms", "paddle_speed", "rt_mean", "n_gest"]


def train_player(a):
    wt = window_table(load(db.connect(a.db))).dropna(subset=PLAYER_FEATURES)
    per = wt.groupby("player").session.nunique()
    keep = per[per >= 3].index
    wt = wt[wt.player.isin(keep)]
    if len(keep) < 2:
        print("player model: need >= 2 players with >= 3 sessions each in the DB (run demo_data.py or play with a friend)")
        return
    X, y, g = wt[PLAYER_FEATURES].values, wt.player.values, wt.session.values
    cv = GroupKFold(n_splits=min(5, wt.session.nunique()))
    cands = {
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=-1),
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
    }
    res, preds = {}, {}
    for n, m in cands.items():
        p = cross_val_predict(m, X, y, cv=cv, groups=g)
        preds[n] = p
        df = pd.DataFrame(dict(s=g, y=y, p=p))
        vote = df.groupby("s").agg(y=("y", "first"), p=("p", lambda v: v.value_counts().idxmax()))
        res[n] = dict(window_acc=round(accuracy_score(y, p), 4), session_acc=round(accuracy_score(vote.y, vote.p), 4))
        print(f"  {n:22s} window acc {res[n]['window_acc']:.3f}   session (majority vote) acc {res[n]['session_acc']:.3f}")
    best = max(res, key=lambda k: res[k]["session_acc"] + res[k]["window_acc"])
    model = cands[best].fit(X, y)
    classes = list(model.classes_)
    save_conf(confusion_matrix(y, preds[best], labels=classes), classes, f"player id: {best} (CV)", MODEL_DIR / "player_confusion.png")
    joblib.dump(dict(model=model, classes=classes, features=PLAYER_FEATURES, kind=best, cv=res), MODEL_DIR / "player_model.joblib")
    (MODEL_DIR / "player_report.json").write_text(json.dumps(dict(best=best, cv=res, players=classes), indent=2))
    print(f"best: {best}   chance = {1/len(classes):.2f}   saved models/player_model.joblib")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["gesture", "fatigue", "player", "all"])
    ap.add_argument("--db", default=str(db.DB_PATH))
    ap.add_argument("--synthetic", action="store_true", help="gesture: ignore recordings, use synthetic windows")
    ap.add_argument("--n", type=int, default=800, help="gesture: synthetic windows per class")
    a = ap.parse_args()
    if a.what in ("gesture", "all"):
        print("== gesture classifier ==")
        train_gesture(a)
    if a.what in ("fatigue", "all"):
        print("\n== fatigue / next-window performance ==")
        train_fatigue(a)
    if a.what in ("player", "all"):
        print("\n== player identification ==")
        train_player(a)


if __name__ == "__main__":
    main()
