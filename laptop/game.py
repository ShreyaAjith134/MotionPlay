"""MotionPlay - gesture game, logs everything to SQLite, sends feedback to the controller.

    python game.py --player shreya                      # solo, ESP32 (or simulator.py)
    python game.py --player shreya --mouse              # solo, mouse + keys Z X C  (no hardware)
    python game.py --player1 shreya --player2 friend    # 2-player split screen (ESP32 #1 and #2)
    python game.py --player shreya --engine ml          # ML gesture classifier instead of rules
    python game.py --player shreya --no-adapt           # fixed difficulty

Controls:  tilt = move | FLICK = pulse (kills red orbs) | SHAKE = bomb | HOLD STILL = shield
           button (GPIO27) = pulse, and start/restart.  Keys: P1 Z X C = flick shake still, P2 J K L, ESC quit, SPACE restart
"""
import argparse
import math
import os
import random
import time

import pygame

import cloud
import db
from arena import *
from config import *
from gestures import make_engine
from receiver import Receiver

BG, HUD_BG, TEXT, DIM, PADDLE = (12, 14, 28), (22, 26, 52), (235, 238, 255), (130, 140, 175), (120, 255, 160)
KEYMAP = {pygame.K_z: (0, "flick"), pygame.K_x: (0, "shake"), pygame.K_c: (0, "still"),
          pygame.K_j: (1, "flick"), pygame.K_k: (1, "shake"), pygame.K_l: (1, "still")}


class Slot:
    """One player: arena + controller state + buffered DB rows."""

    def __init__(self, idx, name, width, x0, args):
        self.idx, self.name, self.pid, self.w, self.x0 = idx, name, idx + 1, width, x0
        self.arena = Arena(width, seed=args.seed, adaptive=not args.no_adapt)
        self.engine = make_engine(args.engine)
        self.mouse = args.mouse and idx == 0
        self.paddle_x = width / 2
        self.tilt0 = self.tilt_f = 0.0
        self.calib = []
        self.btn_prev = 0
        self.sid = None
        self.done = False
        self.buf = dict(samples=[], events=[], frames=[], gest=[])
        self.last_status = 0.0

    @property
    def ready(self):
        return self.mouse or len(self.calib) > 10


def log_events(slot, evs):
    for e in evs:
        slot.buf["events"].append(db.event_row(slot.sid, e))


def use_gesture(slot, rx, g, src, conf, now):
    applied, evs = slot.arena.gesture(g, now)
    slot.buf["gest"].append((slot.sid, now, g, src, conf, int(applied)))
    log_events(slot, evs)


def flush(conn, slots):
    for s in slots:
        if s.buf["samples"]:
            db.insert_samples(conn, s.buf["samples"])
        if s.buf["events"]:
            db.insert_events(conn, s.buf["events"])
        if s.buf["frames"]:
            db.insert_frames(conn, s.buf["frames"])
        if s.buf["gest"]:
            db.insert_gestures(conn, s.buf["gest"])
        s.buf = dict(samples=[], events=[], frames=[], gest=[])


# ------------------------------------------------------------------ drawing
def draw_slot(screen, fonts, s, phase, now, duration, play_start, versus):
    f_s, f_m, f_b = fonts
    A, x0 = s.arena, s.x0
    pygame.draw.rect(screen, HUD_BG, (x0, 0, s.w, HUD_H))
    pygame.draw.rect(screen, BG, (x0, HUD_H, s.w, ARENA_H))
    for gx in range(100, s.w, 100):
        pygame.draw.line(screen, (20, 23, 44), (x0 + gx, HUD_H), (x0 + gx, HUD_H + ARENA_H))
    # --- HUD
    screen.blit(f_s.render(s.name.upper(), True, DIM), (x0 + 10, 6))
    screen.blit(f_b.render(f"{A.score}", True, TEXT), (x0 + 10, 22))
    for i in range(LIVES):
        pygame.draw.circle(screen, C_BAD if i < A.lives else (60, 50, 70), (x0 + s.w - 20 - i * 22, 16), 7)
    mult = min(5, 1 + A.combo // 5)
    screen.blit(f_s.render(f"LV {A.level}   x{mult}  combo {A.combo}   bombs {A.bombs}   speed {A.adapt.mult:.2f}",
                           True, TEXT), (x0 + 150, 6 + 0))
    left = max(0.0, duration - (now - play_start)) if phase == "playing" else duration
    screen.blit(f_s.render(f"{left:4.0f}s", True, DIM), (x0 + s.w - 60, 36))
    for k, (lab, frac, col) in enumerate((("PULSE", A.pulse_frac(), C_PULSE), ("SHIELD", A.shield_frac(), C_SHIELD))):
        bx = x0 + 150 + k * 150
        screen.blit(f_s.render(lab, True, DIM), (bx, 38))
        pygame.draw.rect(screen, (45, 50, 80), (bx + 62, 42, 70, 8))
        pygame.draw.rect(screen, col, (bx + 62, 42, int(70 * frac), 8))
    # --- orbs
    for o in A.orbs:
        c = {"good": C_GOOD, "gold": C_GOLD, "bad": C_BAD}[o.kind]
        p = (int(x0 + o.x), int(HUD_H + o.y))
        pygame.draw.circle(screen, c, p, ORB_R)
        if o.kind == "gold":
            pygame.draw.circle(screen, (255, 235, 160), p, ORB_R + 4, 2)
        if o.kind == "bad":
            pygame.draw.line(screen, BG, (p[0] - 6, p[1] - 6), (p[0] + 6, p[1] + 6), 3)
            pygame.draw.line(screen, BG, (p[0] - 6, p[1] + 6), (p[0] + 6, p[1] - 6), 3)
    # --- paddle
    px, py = x0 + s.paddle_x, HUD_H + PADDLE_Y
    pygame.draw.rect(screen, PADDLE, (px - PADDLE_W / 2, py - PADDLE_H / 2, PADDLE_W, PADDLE_H), border_radius=8)
    if A.shield_active:
        pygame.draw.ellipse(screen, C_SHIELD, (px - PADDLE_W / 2 - 10, py - 26, PADDLE_W + 20, 50), 3)
    # --- toasts
    for i, (txt, col, exp) in enumerate([t for t in A.toasts if t[2] > now][-3:]):
        img = f_m.render(txt, True, col)
        screen.blit(img, (x0 + s.w / 2 - img.get_width() / 2, HUD_H + 90 + i * 34))
    # --- overlays
    if phase == "calibrating":
        msg = "mouse mode" if s.mouse else ("hold flat & still..." if s.ready else "waiting for controller...")
        screen.blit(f_b.render("CALIBRATING", True, TEXT), (x0 + s.w / 2 - 110, HUD_H + 230))
        screen.blit(f_m.render(msg, True, DIM), (x0 + s.w / 2 - 120, HUD_H + 280))
    elif phase == "done":
        screen.blit(f_b.render("GAME OVER", True, TEXT), (x0 + s.w / 2 - 100, HUD_H + 210))
        screen.blit(f_m.render(f"score {A.score}   level {A.level}", True, TEXT), (x0 + s.w / 2 - 105, HUD_H + 260))
        screen.blit(f_s.render("SPACE or button = play again", True, DIM), (x0 + s.w / 2 - 105, HUD_H + 300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", default="player1")
    ap.add_argument("--player1", default=None)
    ap.add_argument("--player2", default=None, help="giving a second name turns on 2-player split screen")
    ap.add_argument("--port", type=int, default=UDP_PORT)
    ap.add_argument("--duration", type=float, default=GAME_SECONDS)
    ap.add_argument("--engine", choices=["rules", "ml"], default="rules")
    ap.add_argument("--mouse", action="store_true")
    ap.add_argument("--no-adapt", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--db", default=str(db.DB_PATH))
    args = ap.parse_args()
    if args.seed is None:
        args.seed = random.randrange(1 << 30)           # same orb sequence for both players = fair
    names = [args.player1 or args.player] + ([args.player2] if args.player2 else [])
    versus = len(names) == 2
    aw = ARENA_W_VERSUS if versus else ARENA_W_SOLO

    conn = db.connect(args.db)
    rx = Receiver(args.port)
    up = cloud.Uploader()
    slots = [Slot(i, n, aw, i * (aw + GAP), args) for i, n in enumerate(names)]
    mode = "versus" if versus else "solo"

    pygame.init()
    screen = pygame.display.set_mode((aw * len(slots) + GAP * (len(slots) - 1), HUD_H + ARENA_H))
    pygame.display.set_caption("MotionPlay")
    mono = "consolas,menlo,dejavusansmono,monospace"
    fonts = (pygame.font.SysFont(mono, 15), pygame.font.SysFont(mono, 24, bold=True), pygame.font.SysFont(mono, 36, bold=True))
    clock = pygame.time.Clock()
    headless = os.environ.get("SDL_VIDEODRIVER") == "dummy"

    phase, calib_start, play_start = "calibrating", time.time(), 0.0
    last_flush, running = time.time(), True
    print(f"[{mode}] players: {', '.join(names)} | engine: {slots[0].engine.name} | "
          f"adaptive: {not args.no_adapt} | cloud: {'ThingSpeak' if up.enabled else 'off'} | "
          f"waiting for controller(s) on UDP {args.port}")

    while running:
        dt = clock.tick(60) / 1000.0
        now = time.time()
        restart = False
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE and phase == "done":
                    restart = True
                elif e.key in KEYMAP and phase == "playing" and KEYMAP[e.key][0] < len(slots):
                    use_gesture(slots[KEYMAP[e.key][0]], rx, KEYMAP[e.key][1], "key", 1.0, now)

        # ---------------- INGEST + PROCESS (per player)
        for s in slots:
            rows = [] if s.mouse else rx.drain(s.pid)
            for r in rows:
                if r[9] and not s.btn_prev:                     # button rising edge
                    if phase == "playing":
                        use_gesture(s, rx, "flick", "button", 1.0, now)
                    elif phase == "done":
                        restart = True
                s.btn_prev = r[9]
                raw = r[3 + TILT_AXIS]
                if phase == "calibrating":
                    s.calib.append(raw)
                elif phase == "playing":
                    s.tilt_f += SMOOTH_ALPHA * ((raw - s.tilt0) - s.tilt_f)
                    s.buf["samples"].append((s.sid, r[1], r[2], r[0]) + r[3:9] + (r[9],))
            if phase == "playing" and rows and not s.done:
                for t, g, src, conf in s.engine.update(rows):
                    use_gesture(s, rx, g, src, conf, now)
            # paddle
            if s.mouse:
                target = pygame.mouse.get_pos()[0] - s.x0
            else:
                norm = max(-1.0, min(1.0, TILT_SIGN * s.tilt_f / TILT_RANGE_G))
                target = s.w / 2 + norm * (s.w / 2 - PADDLE_W / 2)
            s.paddle_x += (target - s.paddle_x) * 0.35
            s.paddle_x = max(PADDLE_W / 2, min(s.w - PADDLE_W / 2, s.paddle_x))

        # ---------------- phase machine
        if restart or (phase == "calibrating" and all(s.ready for s in slots)
                       and (all(s.mouse for s in slots) or now - calib_start > CALIBRATION_SECONDS)):
            if restart:
                phase, calib_start = "calibrating", now
                for s in slots:
                    s.calib, s.done, s.tilt_f = [], False, 0.0
                    s.arena.reset(now)
            else:
                phase, play_start = "playing", now
                for s in slots:
                    if s.calib:
                        s.tilt0 = sum(s.calib) / len(s.calib)
                    s.arena.reset(now)
                    s.sid = db.new_session(conn, s.name, now, mode, s.idx, s.engine.name, not args.no_adapt)
                    rx.send(s.pid, f"S,0,{LIVES},1,0")
                    print(f"session {s.sid} started: {s.name} (tilt0={s.tilt0:.3f} g)")

        # ---------------- GAME
        if phase == "playing":
            for s in slots:
                if s.done:
                    continue
                s.buf["frames"].append((s.sid, now, s.paddle_x))
                log_events(s, s.arena.update(now, dt, s.paddle_x))
                for code in s.arena.fx:
                    rx.send(s.pid, code)
                s.arena.fx.clear()
                if now - s.last_status > 0.5:
                    A = s.arena
                    rx.send(s.pid, f"S,{A.score},{A.lives},{A.level},{A.combo}")
                    s.last_status = now
                if s.arena.over or now - play_start >= args.duration:
                    s.done = True
                    s.arena.over = True
                    db.end_session(conn, s.sid, now, s.arena.score, s.arena.level, s.arena.lives)
                    flush(conn, slots)
                    up.upload(conn, s.sid)
                    rx.send(s.pid, f"E,{s.arena.score}")
                    print(f"{s.name}: score {s.arena.score}, level {s.arena.level}, lives {s.arena.lives}")
            if all(s.done for s in slots):
                phase = "done"
        if now - last_flush > 1.0 or phase == "done":
            flush(conn, slots)
            last_flush = now

        # ---------------- render
        screen.fill((5, 6, 14))
        for s in slots:
            draw_slot(screen, fonts, s, phase, now, args.duration, play_start, versus)
        if versus and phase == "done":
            w = max(slots, key=lambda s: s.arena.score)
            tie = slots[0].arena.score == slots[1].arena.score
            txt = "DRAW" if tie else f"{w.name.upper()} WINS"
            img = fonts[2].render(txt, True, (255, 255, 255))
            screen.blit(img, (screen.get_width() / 2 - img.get_width() / 2, HUD_H + 120))
        pygame.display.flip()
        if phase == "done" and headless:
            break

    flush(conn, slots)
    for s in slots:
        if s.sid and not s.done:
            db.end_session(conn, s.sid, time.time(), s.arena.score, s.arena.level, s.arena.lives)
    rx.close()
    pygame.quit()
    up.finish()
    print("packets received:", dict(rx.total))


if __name__ == "__main__":
    main()
