"""ThingSpeak cloud upload: one summary row per finished game session.

Setup (once): copy thingspeak.example.json -> thingspeak.json and paste your channel ID + Write API key.
(or set env vars THINGSPEAK_CHANNEL_ID / THINGSPEAK_WRITE_KEY). No key = cloud upload is simply off.

Channel fields (name them like this in ThingSpeak -> Channel Settings):
    field1 Score | field2 Level | field3 Catch rate % | field4 Lives left
    field5 Red orbs hit | field6 Power-ups used | field7 Final speed x | field8 Duration s
    status = "player=<name> mode=<solo|versus> engine=<rules|ml>"

Uploads run on a background thread, so the game never waits for the network. If there is no internet the
game carries on and everything is still in SQLite; `python cloud_sync.py --all` can upload it later.
Free ThingSpeak accepts one update per 15 s per channel, so the worker spaces uploads out.
"""
import json
import os
import pathlib
import queue
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

KEY_FILE = pathlib.Path(__file__).with_name("thingspeak.json")
UPDATE_URL = "https://api.thingspeak.com/update.json"
BULK_URL = "https://api.thingspeak.com/channels/{}/bulk_update.json"
MIN_GAP = 16.0      # s between updates (free plan limit is 15 s)

FIELD_NAMES = ["Score", "Level", "Catch rate %", "Lives left", "Red orbs hit", "Power-ups used", "Final speed x",
               "Duration s"]


def load_keys():
    ch = os.environ.get("THINGSPEAK_CHANNEL_ID", "")
    key = os.environ.get("THINGSPEAK_WRITE_KEY", "")
    if KEY_FILE.exists() and not key:
        try:
            d = json.loads(KEY_FILE.read_text())
            ch, key = str(d.get("channel_id", "")).strip(), str(d.get("write_key", "")).strip()
        except (OSError, ValueError) as e:
            print(f"[cloud] could not read {KEY_FILE.name}: {e}")
    if not key or "PASTE" in key:
        return None, None
    return ch or None, key


def load_read_key():
    """Read API key (only needed to read a private channel)."""
    if KEY_FILE.exists():
        try:
            return str(json.loads(KEY_FILE.read_text()).get("read_key", "")).strip() or None
        except (OSError, ValueError):
            pass
    return os.environ.get("THINGSPEAK_READ_KEY") or None


def session_summary(conn, sid):
    """Return the ThingSpeak payload (fields + status + created_at) for one session."""
    s = conn.execute("SELECT player, started, ended, mode, engine, score, level, lives_left FROM sessions WHERE id=?",
                     (sid,)).fetchone()
    if s is None:
        return None
    player, started, ended, mode, engine, score, level, lives = s
    good = conn.execute("SELECT COUNT(*), COALESCE(SUM(ok),0) FROM events WHERE session_id=? AND kind IN ('good','gold')",
                        (sid,)).fetchone()
    red_hit = conn.execute("SELECT COUNT(*) FROM events WHERE session_id=? AND kind='bad' AND outcome='caught'",
                           (sid,)).fetchone()[0]
    powerups = conn.execute("SELECT COUNT(*) FROM gestures WHERE session_id=? AND applied=1", (sid,)).fetchone()[0]
    speed = conn.execute("SELECT speed_mult FROM events WHERE session_id=? ORDER BY t_spawn DESC LIMIT 1",
                         (sid,)).fetchone()
    vals = [score or 0, level or 1, round(100.0 * good[1] / good[0], 1) if good[0] else 0, lives if lives is not None else 0,
            red_hit, powerups, round(speed[0], 2) if speed else 1.0, round((ended or started) - started, 1)]
    d = {f"field{i + 1}": v for i, v in enumerate(vals)}
    d["status"] = f"player={player} mode={mode} engine={engine}"
    d["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started))
    return d


def _post(url, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status, r.read().decode(errors="replace")


def send_one(key, summary):
    """Upload one summary now. Returns (ok, message)."""
    payload = {k: v for k, v in summary.items() if k != "created_at"}
    payload["api_key"] = key
    try:
        status, body = _post(UPDATE_URL, payload)
    except urllib.error.HTTPError as e:
        hint = " (check write_key in thingspeak.json)" if e.code in (400, 401) else ""
        return False, f"HTTP {e.code}{hint}"
    except (urllib.error.URLError, OSError) as e:
        return False, f"network error: {e}"
    if body.strip() in ("0", "-1"):
        return False, "rejected by ThingSpeak (wrong key, or less than 15 s since the last update)"
    try:
        return True, f"entry {json.loads(body).get('entry_id')}"
    except ValueError:
        return True, body[:60]


def send_bulk(channel_id, key, summaries):
    """Upload many summaries in one request, keeping their original timestamps."""
    payload = {"write_api_key": key, "updates": summaries}
    try:
        status, body = _post(BULK_URL.format(channel_id), payload)
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}: {e.read().decode(errors='replace')[:200]}"
    except (urllib.error.URLError, OSError) as e:
        return False, f"network error: {e}"
    return True, body[:200]


class Uploader:
    """Background queue used by game.py. upload(conn, sid) returns immediately."""

    def __init__(self):
        self.channel_id, self.key = load_keys()
        self.enabled = self.key is not None
        self.q = queue.Queue()
        self.last = 0.0
        if self.enabled:
            threading.Thread(target=self._loop, daemon=True).start()

    def upload(self, conn, sid):
        if not self.enabled:
            return
        summary = session_summary(conn, sid)        # read SQLite on the game thread, network on the worker
        if summary:
            self.q.put(summary)

    def _loop(self):
        while True:
            summary = self.q.get()
            wait = MIN_GAP - (time.time() - self.last)
            if wait > 0:
                time.sleep(wait)
            ok, msg = send_one(self.key, summary)
            self.last = time.time()
            who = summary["status"].split()[0]
            print(f"[cloud] {'uploaded' if ok else 'upload FAILED'} {who} score {summary['field1']}: {msg}")
            self.q.task_done()

    def finish(self, timeout=40.0):
        """Called when the game closes: give pending uploads a chance to go out."""
        if not self.enabled or self.q.unfinished_tasks == 0:
            return
        print(f"[cloud] sending {self.q.unfinished_tasks} pending upload(s)...")
        end = time.time() + timeout
        while self.q.unfinished_tasks and time.time() < end:
            time.sleep(0.2)
        if self.q.unfinished_tasks:
            print("[cloud] gave up waiting. Run `python cloud_sync.py --last 2` later to upload them.")
