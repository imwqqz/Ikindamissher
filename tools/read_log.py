import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import turnlog


def _fmt_ts(ts):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="print the turn audit log")
    ap.add_argument("--tail", type=int, default=20)
    ap.add_argument("--source", default=None, help="daemon or repl")
    ap.add_argument("--full", action="store_true",
                    help="also print the context she replied from and the thought")
    ap.add_argument("--json", action="store_true", help="raw rows as JSON")
    args = ap.parse_args(argv)

    found = turnlog.rows(limit=args.tail, source=args.source)
    if args.json:
        print(json.dumps(found, ensure_ascii=False, indent=1))
        return 0
    if not found:
        print("no turns logged yet")
        return 0
    for row in found:
        print(f"[{_fmt_ts(row['ts'])}] {row['source']} #{row['id']}")
        print(f"  you: {row['user_text'] or '(proactive)'}")
        print(f"  her: {row['chosen'] or '(silence)'}")
        print(f"  why: {row['why']}")
        for candidate in row["candidates"]:
            mark = "*" if candidate["text"] == row["chosen"] else " "
            print(f"   {mark} {candidate['score']:.3f}  {candidate['text']!r}")
        if args.full:
            context = row["context"]
            if context.get("demos"):
                print(f"  demos: {context['demos']}")
            if context.get("history"):
                print(f"  history: {context['history'][-4:]}")
            if row["thought"]:
                print(f"  thought: {row['thought']}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
