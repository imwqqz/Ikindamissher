import argparse
import json
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    import msvcrt
except ImportError:  # not Windows: the silence cutoff ends the turn
    msvcrt = None

CLEAR = "\r" + " " * 72 + "\r"
KEY_GRACE = 0.5  # an Enter autorepeat from the start key must not stop at once
_TTY = sys.stdout.isatty()


def _post(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _get(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def health(base, timeout=3):
    return _get(f"{base}/health", timeout)


def _drain_keys():
    if msvcrt is not None:
        while msvcrt.kbhit():
            msvcrt.getwch()


def _enter_pressed():
    if msvcrt is None:
        return False
    hit = False
    while msvcrt.kbhit():
        hit = hit or msvcrt.getwch() in ("\r", "\n")
    return hit


def _listen_call(base, out, timeout):
    try:
        out["text"] = json.loads(_post(f"{base}/listen", {}, timeout)
                                 .decode("utf-8")).get("text", "")
    except Exception as exc:  # noqa: BLE001 - re-raised on the caller's thread
        out["error"] = exc


def listen(base, timeout=120):
    # the POST blocks while the sidecar records; poll /health so the wait is
    # visible and so Enter can stop the recording early
    out = {}
    _drain_keys()
    started = time.time()
    worker = threading.Thread(target=_listen_call, args=(base, out, timeout),
                              daemon=True)
    worker.start()
    stopped = False
    while worker.is_alive():
        try:
            h = health(base, timeout=1)
        except (urllib.error.URLError, OSError):
            h = {}
        state = h.get("state", "recording")
        if state == "recording":
            line = (f"[rec {time.time() - h.get('since', started):4.1f}s "
                    f"rms {h.get('rms', 0.0):.3f}] Enter to stop")
            if not stopped and time.time() - started >= KEY_GRACE \
                    and _enter_pressed():
                stopped = True
                try:
                    _post(f"{base}/listen/stop", {}, 2)
                except (urllib.error.URLError, OSError):
                    pass
        else:
            line = f"[{state}…]"
        if _TTY:
            print("\r" + line.ljust(56), end="", flush=True)
        time.sleep(0.15)
    worker.join()
    if _TTY:
        print(CLEAR, end="", flush=True)
    if "error" in out:
        raise out["error"]
    return out.get("text", "")


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
        print("start it with: uv run python voice/server.py")
        return 1
    try:
        # /poll is side-effect free; POST /message would log a real "test" turn
        _get(args.daemon + "/poll?since=0", 5)
    except (urllib.error.URLError, OSError) as exc:
        print(f"chat daemon not reachable at {args.daemon} ({exc})")
        print("start it with: uv run python server\\service.py")
        return 1

    print("voice chat ready - Enter to talk, Enter again to stop, "
          "'exit' to quit")
    while True:
        _drain_keys()  # a paste during the last turn must not auto-submit
        try:
            command = input("\n[Enter to talk] ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if command.lower() in ("exit", "quit"):
            break
        if command:
            text = command  # typed or pasted line: text turn, no mic
        else:
            try:
                text = listen(args.sidecar)
            except (urllib.error.URLError, OSError) as exc:
                print(f"[voice] listen failed: {exc}")
                continue
        if not text:
            print("(nothing heard, try again)")
            continue
        print(f"<you>: {text}")
        print("… waiting for her", end="", flush=True)
        started = time.perf_counter()
        try:
            reply = ask(args.daemon, text)
        except (urllib.error.URLError, OSError) as exc:
            print(f"\r[daemon] {exc}")
            continue
        print(f"\r… she replied in {time.perf_counter() - started:.1f}s")
        if not reply:
            print("her: (silence)")
            continue
        print(f"her: {reply}")
        print("… speaking", end="", flush=True)
        try:
            play(fetch_speech(args.sidecar, reply))
        except (urllib.error.URLError, OSError) as exc:
            print(f"\r[voice] speak failed: {exc}")
            continue
        print()
        try:
            if health(args.sidecar, timeout=1).get("last_tts") == "piper":
                print("[voice] piper fallback - edge failed, see the "
                      "sidecar log")
        except (urllib.error.URLError, OSError):
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
