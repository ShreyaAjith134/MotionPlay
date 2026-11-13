"""MotionPlay - shared settings (laptop side). Tune things here, not in the code."""
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_DIR = pathlib.Path(os.environ.get("MOTIONPLAY_MODEL_DIR", ROOT / "models"))
GESTURE_DATA_DIR = ROOT / "data" / "gestures"

# ---------------- network ----------------
UDP_PORT = 4210            # ESP32 -> laptop  (sensor stream)
CMD_PORT = 4211            # laptop -> ESP32  (feedback: buzzer / motor / LED / OLED)
SAMPLE_HZ = 50

# ---------------- arena / game rules ----------------
ARENA_W_SOLO = 800
ARENA_W_VERSUS = 600       # per player in 2-player mode
ARENA_H = 600
HUD_H = 64
GAP = 10                   # px between arenas in versus mode
PADDLE_W, PADDLE_H = 110, 16
PADDLE_Y = ARENA_H - 50
ORB_R = 14
BASE_SPEED = 240.0         # px/s at level 1
SPAWN_MIN, SPAWN_MAX = 0.9, 1.5
GAME_SECONDS = 90
LIVES = 5
LEVEL_ORBS = 12            # a new level every N orbs
LEVEL_SPEED_GAIN = 0.08    # +8 % speed per level
BOSS_EVERY = 5             # boss wave when reaching level 5, 10, ...
P_BAD_BASE, P_BAD_PER_LEVEL, P_BAD_MAX = 0.15, 0.02, 0.32
P_GOLD = 0.10
GOLD_SPEED = 1.5

# power-ups (triggered by gestures)
PULSE_CD = 4.0             # FLICK : destroy every red orb on screen
SHIELD_CD = 10.0           # STILL : shield absorbs one red-orb hit
SHIELD_TTL = 8.0
BOMB_EVERY = 10            # SHAKE : bomb, earned every 10 combo
BOMB_MAX = 3

# adaptive difficulty (feedback controller on rolling catch rate)
ADAPT_TARGET = 0.75
ADAPT_BAND = 0.10
ADAPT_STEP = 0.04
ADAPT_WINDOW = 12
ADAPT_MIN_N = 6
ADAPT_MIN_MULT, ADAPT_MAX_MULT = 0.6, 1.8

# ---------------- controller mapping ----------------
TILT_AXIS = 1              # 0=ax 1=ay 2=az : accel axis that changes when you tilt left/right
TILT_SIGN = 1              # -1 if left/right is reversed
TILT_RANGE_G = 0.6         # tilt (g) that maps to the screen edge
SMOOTH_ALPHA = 0.25
CALIBRATION_SECONDS = 1.5

# ---------------- gesture detection (rule engine) ----------------
FLICK_GYRO = 300.0         # deg/s : sharp wrist flick
FLICK_CONFIRM = 0.20       # s    : wait this long after the peak to be sure it is not a shake
FLICK_COOLDOWN = 0.5
FLICK_MAX_REV = 3
SHAKE_REVERSALS = 3        # direction changes within SHAKE_WINDOW
SHAKE_WINDOW = 0.8
SHAKE_MIN_GYRO = 120.0
SHAKE_COOLDOWN = 1.5
STILL_SEC = 1.0            # hold still this long -> shield
STILL_GYRO = 5.0           # deg/s RMS
STILL_REARM = 30.0         # must move above this before STILL can fire again

# ---------------- gesture detection (ML engine) ----------------
GESTURE_CLASSES = ["tilt", "flick", "shake", "still"]
GESTURE_WINDOW = 30        # samples (0.6 s)
GESTURE_STRIDE = 5         # predict every 0.1 s
ML_MIN_PROB = 0.75
ML_FLICK_HOLD = 0.2        # s: wait before firing a flick, so the start of a shake is not mistaken for one

# ---------------- analysis ----------------
WINDOW_SEC = 10
NEAR_PX = 40
MOVE_PX = 15
RT_MAX = 1.5
PRE_SPEED_PX_S = 80
MIN_RT = 0.12
