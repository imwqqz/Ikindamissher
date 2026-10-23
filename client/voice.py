import argparse
import json
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


def _post(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def health(base, timeout=3):
    with urllib.request.urlopen(f"{base}/health", timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def listen(base, timeout=120):
    return json.loads(_post(f"{base}/listen", {}, timeout).decode("utf-8")) \
        .get("text", "")


def ask(daemon, text, timeout=300):
    return json.loads(_post(f"{daemon}/message", {"text": text}, timeout)
                      .decode("utf-8")).get("reply", "")


def fetch_speech(base, text, timeout=180):
    return _post(f"{base}/speak", {"text": text}, timeout)


def play(wav_bytes):
    try:
        import winsound
    except ImportError:
        print("[voice] playback needs Windows (winsound)")
        return
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
        fh.write(wav_bytes)
        path = fh.name
    try:
        winsound.PlaySound(path, winsound.SND_FILENAME)
    finally:
        Path(path).unlink(missing_ok=True)


def build_parser():
    ap = argparse.ArgumentParser(
        description="voice chat: talk -> daemon -> speaks")
    ap.add_argument("--sidecar", default="http://127.0.0.1:8779",
                    help="voice server (voice/server.py)")
    ap.add_argument("--daemon", default="http://127.0.0.1:8777",
                    help="chat daemon (server/service.py)")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        health(args.sidecar)
    except (urllib.error.URLError, OSError) as exc:
        print(f"voice sidecar not reachable at {args.sidecar} ({exc})")
        print("start it with: uv run --project voice python voice/server.py")
        return 1
    try:
        health(args.daemon + "/poll?since=0")
    except (urllib.error.URLError, OSError) as exc:
        print(f"chat daemon not reachable at {args.daemon} ({exc})")
        print("start it with: uv run python server\\service.py")
        return 1

    print("voice chat ready - Enter to talk, 'exit' to quit")
    while True:
        try:
            command = input("\n[Enter to talk] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if command in ("exit", "quit"):
            break
        try:
            text = listen(args.sidecar)
        except (urllib.error.URLError, OSError) as exc:
            print(f"[voice] listen failed: {exc}")
            continue
        if not text:
            print("(nothing heard, try again)")
            continue
        print(f"<you>: {text}")
        try:
            reply = ask(args.daemon, text)
        except (urllib.error.URLError, OSError) as exc:
            print(f"[daemon] {exc}")
            continue
        if not reply:
            print("her: (silence)")
            continue
        print(f"her: {reply}")
        try:
            play(fetch_speech(args.sidecar, reply))
        except (urllib.error.URLError, OSError) as exc:
            print(f"[voice] speak failed: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
