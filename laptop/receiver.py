"""UDP link to the ESP32(s).

in  : sensor packets: pid,seq,t_ms,ax,ay,az,gx,gy,gz,btn     (an 8-field packet without pid/btn is also accepted -> pid 1)
out : feedback commands to the sender's IP on CMD_PORT
        H hit | M miss | X bad-orb hit | G gesture used | S,score,lives,level,combo (OLED) | E,score (game over)
Row tuple: (t_host, seq, t_dev_ms, ax, ay, az, gx, gy, gz, btn)
"""
import collections
import socket
import threading
import time

from config import CMD_PORT, UDP_PORT


def parse(data):
    try:
        p = data.decode().strip().split(",")
        now = time.time()
        if len(p) == 8:                      # no pid / button field
            return 1, (now, int(p[0]), int(p[1])) + tuple(float(x) for x in p[2:8]) + (0,)
        if len(p) == 10:
            return int(p[0]), (now, int(p[1]), int(p[2])) + tuple(float(x) for x in p[3:9]) + (int(p[9]),)
    except (ValueError, UnicodeDecodeError):
        pass
    return None


class Receiver:
    def __init__(self, port=UDP_PORT, cmd_port=CMD_PORT):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", port))
        self.sock.settimeout(0.2)
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cmd_port = cmd_port
        self.queues = {}
        self.addr = {}
        self.total = collections.Counter()
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(256)
            except socket.timeout:
                continue
            except OSError:
                break
            parsed = parse(data)
            if parsed is None:
                continue
            pid, row = parsed
            self.queues.setdefault(pid, collections.deque()).append(row)
            self.addr[pid] = addr[0]
            self.total[pid] += 1

    def drain(self, pid):
        q = self.queues.get(pid)
        out = []
        while q:
            out.append(q.popleft())
        return out

    def send(self, pid, text):
        ip = self.addr.get(pid)
        if ip:
            try:
                self.tx.sendto(text.encode(), (ip, self.cmd_port))
            except OSError:
                pass

    def close(self):
        self.running = False
        self.sock.close()
        self.tx.close()
