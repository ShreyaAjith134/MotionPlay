"""Check the ThingSpeak connection, or upload sessions that are already in SQLite.

    python cloud_sync.py --test                       # send one test row (check your key works)
    python cloud_sync.py --show                       # print the newest rows stored in ThingSpeak
    python cloud_sync.py --last 3                     # upload the 3 newest sessions
    python cloud_sync.py --all                        # upload every finished session
    python cloud_sync.py --all --db motionplay_demo.db    # upload the bot demo data (fills the charts)

--last / --all use ThingSpeak's bulk update, which keeps each session's real start time,
so they need the channel ID as well as the Write API key.
"""
import argparse
import json
import time
import urllib.request

import cloud
import db


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--test", action="store_true")
    g.add_argument("--show", action="store_true")
    g.add_argument("--last", type=int)
    g.add_argument("--all", action="store_true")
    ap.add_argument("--db", default=str(db.DB_PATH))
    a = ap.parse_args()

    channel_id, key = cloud.load_keys()
    if not key:
        raise SystemExit(f"No ThingSpeak key. Copy thingspeak.example.json to {cloud.KEY_FILE.name} and fill it in.")

    if a.show:
        if not channel_id:
            raise SystemExit("--show needs channel_id in thingspeak.json.")
        url = f"https://api.thingspeak.com/channels/{channel_id}/feeds.json?results=8&status=true"
        rk = cloud.load_read_key()
        if rk:
            url += "&api_key=" + rk
        with urllib.request.urlopen(url, timeout=10) as r:
            d = json.load(r)
        names = [d["channel"].get(f"field{i}", f"field{i}") for i in range(1, 9)]
        print(f"ThingSpeak channel {channel_id}: {d['channel']['name']}  ({d['channel']['last_entry_id']} entries)\n")
        print(f"{'uploaded (UTC)':20} {'player':12} " + " ".join(f"{n[:12]:>12}" for n in names[:6]))
        for e in d["feeds"]:
            who = (e.get("status") or "").split()[0].replace("player=", "") if e.get("status") else "-"
            print(f"{e['created_at'][:19].replace('T', ' '):20} {who:12} " + " ".join(f"{(e.get(f'field{i}') or '-'):>12}" for i in range(1, 7)))
        return

    if a.test:
        row = {f"field{i + 1}": v for i, v in enumerate([0, 1, 0, 5, 0, 0, 1.0, 0])}
        row["status"] = "player=test mode=connection-check engine=-"
        ok, msg = cloud.send_one(key, row)
        print(("OK, " if ok else "FAILED, ") + msg)
        if ok and channel_id:
            print(f"Open https://thingspeak.com/channels/{channel_id} to see it.")
        return

    if not channel_id:
        raise SystemExit("--last / --all need channel_id in thingspeak.json as well.")
    conn = db.connect(a.db)
    q = "SELECT id FROM sessions WHERE ended IS NOT NULL ORDER BY started"
    ids = [r[0] for r in conn.execute(q)]
    if a.last:
        ids = ids[-a.last:]
    if not ids:
        raise SystemExit("No finished sessions in " + a.db)
    rows = [cloud.session_summary(conn, i) for i in ids]
    for chunk_start in range(0, len(rows), 900):            # free plan: 960 rows per bulk request
        if chunk_start:
            time.sleep(cloud.MIN_GAP)
        ok, msg = cloud.send_bulk(channel_id, key, rows[chunk_start:chunk_start + 900])
        print(("OK, " if ok else "FAILED, ") + msg)
        if not ok:
            raise SystemExit(1)
    print(f"Uploaded {len(rows)} session(s). Open https://thingspeak.com/channels/{channel_id}")


if __name__ == "__main__":
    main()
