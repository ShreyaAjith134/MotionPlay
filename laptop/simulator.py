"""Fake ESP32: streams IMU packets over UDP (random tilting + occasional flick / shake / hold-still)
and prints any feedback commands the game sends back (H / M / X / G / S / E).

    python simulator.py                      # player 1 -> localhost
    python simulator.py --pid 2              # second fake controller (versus mode)
    python simulator.py --host 192.168.1.50 --seconds 30
"""
import argparse
import socket
import threading
import time

import numpy as np

from config import *
from synth import imu_from_tilt, smooth_tilt


def make_chunk(rng, secs, hz):
    ts = np.arange(0, secs, 1.0 / hz)
    tilt = (0.45 * np.sin(2 * np.pi * 0.20 * ts + rng.uniform(0, 6)) + 0.15 * np.sin(2 * np.pi * 0.9 * ts)
            + 0.05 * smooth_tilt(ts, rng))
    gl, t = [], 3.0
    while t < secs - 3:
        kind = str(rng.choice(["flick", "flick", "shake", "still"]))
        gl.append((t + (1.0 if kind == "still" else 0.8 if kind == "shake" else 0.0), kind))
        t += rng.uniform(4, 8)
    return ts, imu_from_tilt(ts, tilt, rng, tremor=0.01, gestures=gl)


def listen_feedback(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind(("0.0.0.0", port))
    except OSError:
        return
    s.settimeout(0.5)

    def loop():
        while True:
            try:
                data, _ = s.recvfrom(64)
                print("   <- feedback:", data.decode())
            except socket.timeout:
                pass
            except OSError:
                break
    threading.Thread(target=loop, daemon=True).start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=UDP_PORT)
    ap.add_argument("--pid", type=int, default=1)
    ap.add_argument("--hz", type=float, default=SAMPLE_HZ)
    ap.add_argument("--seconds", type=float, default=0, help="0 = run forever")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--quiet", action="store_true", help="do not print feedback commands")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if not a.quiet and a.pid == 1:
        listen_feedback(CMD_PORT)
    print(f"Simulating ESP32 #{a.pid} -> {a.host}:{a.port} at {a.hz:.0f} Hz (Ctrl+C to stop)")
    t0, seq, offset = time.time(), 0, 0.0
    try:
        while True:
            ts, imu = make_chunk(rng, 60.0, a.hz)
            for i in range(len(ts)):
                target = t0 + offset + ts[i]
                time.sleep(max(0.0, target - time.time()))
                if a.seconds and target - t0 > a.seconds:
                    raise KeyboardInterrupt
                r = imu[i]
                msg = "%d,%d,%d,%.4f,%.4f,%.4f,%.2f,%.2f,%.2f,0" % (
                    a.pid, seq, int((target - t0) * 1000), r[0], r[1], r[2], r[3], r[4], r[5])
                sock.sendto(msg.encode(), (a.host, a.port))
                seq += 1
            offset += 60.0
    except KeyboardInterrupt:
        pass
    print(f"sent {seq} packets")


if __name__ == "__main__":
    main()
