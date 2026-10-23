import json
import sqlite3
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import LIVE_DIR, PROJECT_ROOT, move_legacy

DB_PATH = LIVE_DIR / "telemetry.sqlite3"
move_legacy(PROJECT_ROOT / "data" / "telemetry.sqlite3", DB_PATH)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    source TEXT NOT NULL,
    user_text TEXT NOT NULL,
    candidates TEXT NOT NULL,   -- JSON: [{"text": ..., "score": ...}]
    chosen TEXT NOT NULL,
    why TEXT NOT NULL,
    context TEXT NOT NULL,      -- JSON: {"persona", "demos", "history"}
    thought TEXT NOT NULL DEFAULT ''
)
"""


def _connect(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute(_SCHEMA)
    return con


def record(source, user_text, candidates, chosen, why, context, thought="",
           path=DB_PATH, now=None):
    row = (time.time() if now is None else now, source, user_text,
           json.dumps([{"text": t, "score": s} for t, s in candidates],
                      ensure_ascii=False),
           chosen, why, json.dumps(context, ensure_ascii=False), thought)
    con = _connect(path)
    try:
        with con:
            con.execute(
                "INSERT INTO turns(ts, source, user_text, candidates, chosen,"
                " why, context, thought) VALUES(?,?,?,?,?,?,?,?)", row)
        return con.execute("SELECT last_insert_rowid()").fetchone()[0]
    finally:
        con.close()


def rows(limit=20, source=None, path=DB_PATH):
    con = _connect(path)
    try:
        query = "SELECT id, ts, source, user_text, candidates, chosen, why," \
                " context, thought FROM turns"
        params = []
        if source:
            query += " WHERE source = ?"
            params.append(source)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        found = con.execute(query, params).fetchall()
    finally:
        con.close()
    out = []
    for row in reversed(found):
        item = dict(zip(("id", "ts", "source", "user_text", "candidates",
                         "chosen", "why", "context", "thought"), row))
        item["candidates"] = json.loads(item["candidates"])
        item["context"] = json.loads(item["context"])
        out.append(item)
    return out
