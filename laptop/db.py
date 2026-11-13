"""SQLite storage. Default file: laptop/motionplay.db."""
import pathlib
import sqlite3

DB_PATH = pathlib.Path(__file__).with_name("motionplay.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player TEXT NOT NULL, started REAL NOT NULL, ended REAL,
    mode TEXT, slot INTEGER, engine TEXT, adaptive INTEGER,
    score INTEGER, level INTEGER, lives_left INTEGER
);
CREATE TABLE IF NOT EXISTS samples(      -- raw IMU stream (~50 rows/s)
    session_id INTEGER NOT NULL, seq INTEGER, t_dev_ms INTEGER, t_host REAL,
    ax REAL, ay REAL, az REAL, gx REAL, gy REAL, gz REAL, btn INTEGER
);
CREATE TABLE IF NOT EXISTS events(       -- one row per orb
    session_id INTEGER NOT NULL, idx INTEGER, kind TEXT,        -- good | bad | gold
    t_spawn REAL, orb_x REAL, paddle_x_spawn REAL,
    t_resolve REAL, outcome TEXT, ok INTEGER, paddle_x_resolve REAL,
    level INTEGER, speed_mult REAL, combo INTEGER, score INTEGER, lives INTEGER
);
CREATE TABLE IF NOT EXISTS frames(       -- paddle position per rendered frame
    session_id INTEGER NOT NULL, t REAL, paddle_x REAL
);
CREATE TABLE IF NOT EXISTS gestures(     -- every detected gesture (applied = had an effect)
    session_id INTEGER NOT NULL, t REAL, gesture TEXT, source TEXT, conf REAL, applied INTEGER
);
CREATE INDEX IF NOT EXISTS ix_samples ON samples(session_id, t_host);
CREATE INDEX IF NOT EXISTS ix_events ON events(session_id, t_spawn);
CREATE INDEX IF NOT EXISTS ix_frames ON frames(session_id, t);
CREATE INDEX IF NOT EXISTS ix_gest ON gestures(session_id, t);
"""


def connect(path=DB_PATH):
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")      # lets live_dashboard.py read while the game writes
    conn.executescript(SCHEMA)
    return conn


def new_session(conn, player, t, mode="solo", slot=0, engine="rules", adaptive=1):
    cur = conn.execute("INSERT INTO sessions(player, started, mode, slot, engine, adaptive) VALUES (?,?,?,?,?,?)",
                       (player, t, mode, slot, engine, int(adaptive)))
    conn.commit()
    return cur.lastrowid


def end_session(conn, sid, t, score=None, level=None, lives_left=None):
    conn.execute("UPDATE sessions SET ended=?, score=?, level=?, lives_left=? WHERE id=?",
                 (t, score, level, lives_left, sid))
    conn.commit()


def event_row(sid, e, t_offset=0.0):
    return (sid, e["idx"], e["kind"], t_offset + e["t_spawn"], e["orb_x"], e["px_spawn"], t_offset + e["t_resolve"],
            e["outcome"], e["ok"], e["px_resolve"], e["level"], e["mult"], e["combo"], e["score"], e["lives"])


def insert_samples(conn, rows):
    conn.executemany("INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()


def insert_events(conn, rows):
    conn.executemany("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()


def insert_frames(conn, rows):
    conn.executemany("INSERT INTO frames VALUES (?,?,?)", rows)
    conn.commit()


def insert_gestures(conn, rows):
    conn.executemany("INSERT INTO gestures VALUES (?,?,?,?,?,?)", rows)
    conn.commit()
