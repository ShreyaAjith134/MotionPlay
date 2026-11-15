"""Pure game logic (no pygame, no wall clock).

The same Arena is driven by the real game (game.py), by the bot in demo_data.py and by the self-tests.

Orbs   : good (cyan, +10)  gold (yellow, fast, +50)  bad (red, catching it costs a life)
Powers : FLICK -> pulse (destroys every red orb, 4 s cooldown)
         SHAKE -> bomb  (auto-catches every good orb, destroys red ones; one bomb earned per 10 combo)
         STILL -> shield (absorbs one red-orb hit, 10 s cooldown)
Rules  : combo multiplier up to x5, levels every 12 orbs, boss wave every 5th level,
         adaptive difficulty keeps the rolling catch rate near ADAPT_TARGET.
"""
import math
import random
from collections import deque
from dataclasses import dataclass

from config import *

C_GOOD = (0, 229, 255)
C_GOLD = (255, 196, 61)
C_BAD = (255, 71, 87)
C_SHIELD = (110, 170, 255)
C_BOMB = (255, 140, 66)
C_PULSE = (190, 120, 255)


@dataclass
class Orb:
    idx: int
    kind: str          # 'good' | 'bad' | 'gold'
    x: float
    y: float
    vy: float
    t_spawn: float
    px: float          # paddle x when it spawned
    mult: float        # difficulty multiplier when it spawned
    level: int


class Adaptive:
    """Feedback controller: too easy -> faster, too hard -> slower."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.mult = 1.0
        self.hist = deque(maxlen=ADAPT_WINDOW)

    def record(self, ok):
        self.hist.append(1 if ok else 0)
        if not self.enabled or len(self.hist) < ADAPT_MIN_N:
            return
        rate = sum(self.hist) / len(self.hist)
        if rate > ADAPT_TARGET + ADAPT_BAND:
            self.mult += ADAPT_STEP
        elif rate < ADAPT_TARGET - ADAPT_BAND:
            self.mult -= ADAPT_STEP
        self.mult = min(ADAPT_MAX_MULT, max(ADAPT_MIN_MULT, self.mult))


class Arena:
    def __init__(self, width, seed=None, adaptive=True):
        self.w = width
        self.seed = seed
        self.adaptive_on = adaptive
        self.reset(0.0)

    # ------------------------------------------------------------ state
    def reset(self, t0):
        self.rng = random.Random(self.seed)
        self.t0 = t0
        self.now = t0
        self.score = 0
        self.lives = LIVES
        self.level = 1
        self.combo = 0
        self.best_combo = 0
        self.bombs = 0
        self.orbs = []
        self.idx = 0
        self.spawned = 0
        self.next_spawn = t0 + 1.0
        self.wave = []
        self.pulse_ready = t0
        self.shield_ready = t0
        self.shield_until = 0.0
        self.last_paddle = self.w / 2
        self.adapt = Adaptive(self.adaptive_on)
        self.over = False
        self.toasts = []
        self.fx = []           # hardware feedback codes (H hit, M miss, X bad hit, G gesture); consumer clears
        self.stats = {"caught": 0, "missed": 0, "bad_hit": 0, "bad_avoided": 0, "gestures": 0}

    @property
    def shield_active(self):
        return self.shield_until > self.now

    @property
    def speed(self):
        return BASE_SPEED * (1 + LEVEL_SPEED_GAIN * (self.level - 1)) * self.adapt.mult

    def pulse_frac(self):
        return 1.0 - max(0.0, self.pulse_ready - self.now) / PULSE_CD

    def shield_frac(self):
        return 1.0 - max(0.0, self.shield_ready - self.now) / SHIELD_CD

    def toast(self, text, color):
        self.toasts = [x for x in self.toasts if x[2] > self.now][-4:]
        self.toasts.append((text, color, self.now + 1.3))

    # ------------------------------------------------------------ spawning
    def _p_bad(self):
        return min(P_BAD_MAX, P_BAD_BASE + P_BAD_PER_LEVEL * (self.level - 1))

    def _spawn(self, t, kind, x, paddle_x):
        vy = self.speed * (GOLD_SPEED if kind == "gold" else 1.0)
        self.orbs.append(Orb(self.idx, kind, x, -ORB_R, vy, t, paddle_x, self.adapt.mult, self.level))
        self.idx += 1
        self.spawned += 1
        if self.spawned % LEVEL_ORBS == 0:
            self._level_up(t)

    def _level_up(self, t):
        self.level += 1
        self.toast(f"LEVEL {self.level}", (255, 255, 255))
        if self.level % BOSS_EVERY == 0:
            self._queue_wave(t)

    def _queue_wave(self, t):
        n, amp, phase = 10, self.w * 0.32, self.rng.uniform(0, 6.28)
        for i in range(n):
            kind = "bad" if i in (3, 7) else ("gold" if i == 9 else "good")
            self.wave.append((t + 1.2 + i * 0.4, kind, self.w / 2 + amp * math.sin(phase + i * 0.9)))
        self.next_spawn = t + 1.2 + n * 0.4 + 1.0
        self.toast("BOSS WAVE", C_BAD)

    # ------------------------------------------------------------ resolution
    def _resolve(self, o, t, px, outcome):
        ok = 0
        if o.kind in ("good", "gold"):
            if outcome in ("caught", "bombed"):
                self.combo += 1
                self.best_combo = max(self.best_combo, self.combo)
                self.score += (50 if o.kind == "gold" else 10) * min(5, 1 + self.combo // 5)
                if self.combo % BOMB_EVERY == 0 and self.bombs < BOMB_MAX:
                    self.bombs += 1
                    self.toast("BOMB READY", C_BOMB)
                ok = 1
                self.stats["caught"] += 1
                self.fx.append("H")
            else:
                if o.kind == "good":
                    self.combo = 0
                self.stats["missed"] += 1
                self.fx.append("M")
            self.adapt.record(ok)
        else:
            if outcome == "caught":
                if self.shield_active:
                    outcome, ok, self.shield_until = "shielded", 1, 0.0
                    self.toast("SHIELD!", C_SHIELD)
                    self.fx.append("G")
                else:
                    self.lives -= 1
                    self.combo = 0
                    self.stats["bad_hit"] += 1
                    self.fx.append("X")
            else:
                outcome = "avoided" if outcome == "missed" else outcome
                ok = 1
                self.stats["bad_avoided"] += 1
        return dict(idx=o.idx, kind=o.kind, t_spawn=o.t_spawn, orb_x=o.x, px_spawn=o.px, t_resolve=t,
                    outcome=outcome, ok=ok, px_resolve=px, level=o.level, mult=o.mult,
                    combo=self.combo, score=self.score, lives=self.lives)

    # ------------------------------------------------------------ main step
    def update(self, t, dt, paddle_x):
        """Advance the game to time t. Returns the list of orb-resolution events (dicts)."""
        self.now = t
        self.last_paddle = paddle_x
        events = []
        if self.over:
            return events
        while self.wave and self.wave[0][0] <= t:
            _, kind, x = self.wave.pop(0)
            self._spawn(t, kind, x, paddle_x)
        if not self.wave and t >= self.next_spawn:
            r, pb = self.rng.random(), self._p_bad()
            kind = "bad" if r < pb else ("gold" if r < pb + P_GOLD else "good")
            x = self.rng.uniform(ORB_R + 10, self.w - ORB_R - 10)
            self.next_spawn = t + self.rng.uniform(SPAWN_MIN, SPAWN_MAX) * max(0.55, 1 - 0.05 * (self.level - 1))
            self._spawn(t, kind, x, paddle_x)      # may overwrite next_spawn (boss wave)
        line = PADDLE_Y - PADDLE_H / 2
        for o in self.orbs[:]:
            o.y += o.vy * dt
            if o.y >= line:
                self.orbs.remove(o)
                hit = abs(o.x - paddle_x) <= PADDLE_W / 2 + ORB_R
                events.append(self._resolve(o, t, paddle_x, "caught" if hit else "missed"))
        if self.lives <= 0:
            self.over = True
        return events

    def gesture(self, g, t):
        """Apply a gesture. Returns (applied, events)."""
        self.now = t
        events = []
        if self.over:
            return False, events
        if g == "flick":
            if t < self.pulse_ready:
                return False, events
            self.pulse_ready = t + PULSE_CD
            for o in [o for o in self.orbs if o.kind == "bad"]:
                self.orbs.remove(o)
                events.append(self._resolve(o, t, self.last_paddle, "destroyed"))
            self.toast("PULSE", C_PULSE)
        elif g == "shake":
            if self.bombs <= 0:
                return False, events
            self.bombs -= 1
            for o in self.orbs[:]:
                self.orbs.remove(o)
                events.append(self._resolve(o, t, self.last_paddle, "destroyed" if o.kind == "bad" else "bombed"))
            self.toast("BOMB!", C_BOMB)
        elif g == "still":
            if t < self.shield_ready or self.shield_active:
                return False, events
            self.shield_ready = t + SHIELD_CD
            self.shield_until = t + SHIELD_TTL
            self.toast("SHIELD UP", C_SHIELD)
        else:
            return False, events
        self.stats["gestures"] += 1
        self.fx.append("G")
        return True, events
