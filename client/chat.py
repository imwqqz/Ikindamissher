import argparse
import json
import sys
import urllib.error
import urllib.request


def _request(url, body=None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
        method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _drain(base, since):
    """Print her proactive openers queued since the last check."""
    try:
        payload = _request(f"{base}/poll?since={since}")
    except (urllib.error.URLError, OSError):
        return since
    for message in payload.get("messages", []):
        print(f"\nher: {message['text']}")
        since = max(since, message["id"])
    return since


def build_parser():
    ap = argparse.ArgumentParser(
        description="chat with the 24/7 daemon from a terminal")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8777)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    base = f"http://{args.host}:{args.port}"
    try:
        _request(f"{base}/poll?since=0")
    except (urllib.error.URLError, OSError) as exc:
        print(f"daemon not reachable at {base} ({exc})")
        print("start it with: uv run python server\\service.py")
        return 1
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"chatting with the daemon at {base} - type 'exit' to quit")
    since = 0
    while True:
        since = _drain(base, since)
        try:
            prompt = input("<you>: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt or prompt.lower() in ("exit", "quit"):
            break
        since = _drain(base, since)
        try:
            payload = _request(f"{base}/message", {"text": prompt})
        except (urllib.error.URLError, OSError) as exc:
            print(f"[error] {exc}")
            continue
        reply = payload.get("reply") or ""
        print(f"her: {reply}" if reply else "her: (silence)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
