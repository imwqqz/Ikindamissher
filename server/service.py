"""24/7 HTTP daemon around the chatbot engine.

Run: uv run python server\\service.py --engine qwen (or --engine her)
Endpoints (poll transport only):
- POST /message  {"text": "..."}  -> {"reply": "..."}
- GET  /poll?since=N             -> {"messages": [{id, text, ts}], "since": N}

Durability: atomic session writes every turn, proactive messages in SQLite,
one bad scheduler tick logs without stopping the daemon.
"""
import json
import os
import sqlite3
import sys
import threading
import time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import engines.dispatch as engine
import client.repl as cli
from config import resolve_config_args
from data import speaker_names

SESSION_PATH = _ROOT / "data" / "chat_session.json"
QUEUE_PATH = _ROOT / "data" / "proactive.sqlite3"

_MAX_POLL = 50


def load_session(path=SESSION_PATH):
    if not Path(path).exists():
        return {"history": []}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        # a corrupt session moves aside instead of overwriting
        bad = Path(path).with_suffix(".corrupt.json")
        try:
            os.replace(path, bad)
            print(f"[session] unreadable, moved to {bad}: {exc}")
        except OSError:
            pass
        return {"history": []}
    if not isinstance(data, dict) or not isinstance(data.get("history"), list):
        return {"history": []}
    return data


def save_session(sess, path=SESSION_PATH):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(sess, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    os.replace(tmp, path)  # atomic on Windows (same volume)

def _init_queue(path=QUEUE_PATH):
    con = sqlite3.connect(path)
    try:
        with con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS proactive ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "text TEXT NOT NULL, ts REAL NOT NULL)")
    finally:
        con.close()


def _queue_push(text, ts, path=QUEUE_PATH):
    con = sqlite3.connect(path)
    try:
        with con:
            cur = con.execute(
                "INSERT INTO proactive(text, ts) VALUES(?, ?)", (text, ts))
            return cur.lastrowid
    finally:
        con.close()


def _queue_since(since, path=QUEUE_PATH):
    con = sqlite3.connect(path)
    try:
        rows = con.execute(
            "SELECT id, text, ts FROM proactive WHERE id > ? "
            "ORDER BY id LIMIT ?", (since, _MAX_POLL)).fetchall()
    finally:
        con.close()
    return [{"id": r[0], "text": r[1], "ts": r[2]} for r in rows]


def handle_message(text, S):
    """One user turn: reply, then persist the extended history atomically."""
    args = S["args"]
    user_name, her_name = speaker_names()
    with S["session_lock"]:
        history = list(load_session().get("history", []))
        with S["engine_lock"]:
            reply = engine.generate_reply(
                args, S["state"], S["model"], S["tokenizer"], text, history,
                S["demo_index"], S["device"])
            if not isinstance(reply, str):  # never trust the engine blindly
                reply = ""
            # remember the user's turn even on a silent reply
            history.append(f"{user_name}: "
                           f"{engine.clip_msg(text, 15, S['tokenizer'])}")
            if reply:
                history.append(f"{her_name}: "
                               f"{engine.clip_msg(reply, 15, S['tokenizer'])}")
            history = history[-16:]
            save_session({"history": history, "updated": time.time()})
    return reply


def _in_quiet_hours(hour, start, end):
    if start is None or end is None or start == end:
        return False
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end  # window wraps midnight


def _tick(S):
    args = S["args"]
    now = time.time()
    if args.proactive_every <= 0:
        return
    if now - S["last_proactive"] < args.proactive_every * 60:
        return
    if _in_quiet_hours(time.localtime().tm_hour, args.quiet_start,
                       args.quiet_end):
        return
    with S["engine_lock"]:
        text = engine.generate_proactive(
            args, S["state"], S["model"], S["tokenizer"], S["openers"],
            S["device"])
    if text:
        _queue_push(text, now)
        # the opener joins the session history too, like the REPL records it
        with S["session_lock"]:
            _, her_name = speaker_names()
            history = list(load_session().get("history", []))
            history.append(f"{her_name}: "
                           f"{engine.clip_msg(text, 15, S['tokenizer'])}")
            save_session({"history": history[-16:], "updated": now})
        S["last_proactive"] = now
        print(f"[scheduler] queued proactive: {text!r}")
    else:
        print("[scheduler] proactive generation empty; cooling down")
        S["last_proactive"] = now - args.proactive_every * 30  # retry in half


def _scheduler_loop(S, stop):
    while not stop.is_set():
        deadline = time.monotonic() + S["args"].interval
        try:
            _tick(S)
        except Exception:
            traceback.print_exc()  # a bad tick must never stop the daemon
        finally:
            stop.wait(max(0.0, deadline - time.monotonic()))


class Handler(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/poll":
            return self._json(404, {"error": "unknown endpoint"})
        q = urllib.parse.parse_qs(parsed.query)
        try:
            since = int(q.get("since", ["0"])[0] or 0)
        except ValueError:
            return self._json(400, {"error": "since must be an integer"})
        try:
            msgs = _queue_since(since)
            return self._json(200, {"messages": msgs,
                                    "since": max((m["id"] for m in msgs),
                                                 default=since)})
        except Exception:
            traceback.print_exc()
            return self._json(500, {"error": "poll failed"})

    def do_POST(self):
        if urllib.parse.urlparse(self.path).path != "/message":
            return self._json(404, {"error": "unknown endpoint"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            text = str(json.loads(self.rfile.read(n) or b"{}").get("text", ""))
            text = text.strip()
            if not text:
                return self._json(400, {"error": "text must not be empty"})
            reply = handle_message(text, self.server.S)
            return self._json(200, {"reply": reply})
        except Exception:
            traceback.print_exc()
            return self._json(500, {"error": "generation failed"})

    def log_message(self, fmt, *argv):  # quiet default access log
        if self.server.verbose:
            super().log_message(fmt, *argv)


def build_parser():
    ap = cli.build_parser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--interval", type=int, default=60,
                    help="scheduler tick period in seconds")
    ap.add_argument("--proactive-every", type=int, default=90,
                    help="minimum minutes between proactive messages "
                         "(0 disables)")
    ap.add_argument("--quiet-start", type=int, default=1, metavar="H",
                    help="no proactive messages from this hour (inclusive)")
    ap.add_argument("--quiet-end", type=int, default=9, metavar="H",
                    help="no proactive messages until this hour (exclusive)")
    ap.add_argument("--verbose", action="store_true")
    return ap


def main(argv=None):
    parser = build_parser()
    args = resolve_config_args(parser, argv, cli.CHAT_CONFIG)
    args = engine.resolve_max_reply(args)

    device = cli._setup_device(args.cpu)
    state, model, tokenizer = engine.load_assets(args, device)
    cfg = state["cfg"]
    cli._configure_from_checkpoint(args, cfg)

    corpus_text = engine.load_corpus(args, cfg)
    demo_index = engine.build_demo_index(corpus_text, tokenizer) \
        if corpus_text else None
    openers = engine.load_openers(args)

    _init_queue()
    S = {
        "args": args,
        "state": state,
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "demo_index": demo_index,
        "openers": openers,
        "engine_lock": threading.Lock(),   # one model, one generation at a time
        "session_lock": threading.Lock(),
        "last_proactive": 0.0,             # tick 1 fires if hours allow
        "interval": args.interval,
    }
    stop = threading.Event()
    threading.Thread(target=_scheduler_loop, args=(S, stop), daemon=True).start()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.S = S
    server.verbose = args.verbose
    print(f"serving on http://{args.host}:{args.port} "
          f"(POST /message, GET /poll?since=N)")
    print(f"session: {SESSION_PATH} | queue: {QUEUE_PATH}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        stop.set()
        server.server_close()


if __name__ == "__main__":
    main()

