import threading
import time
from pathlib import Path

import sys

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import LIVE_DIR, PROJECT_ROOT, move_legacy
from data import CONVERSATION_GAP_MINUTES, speaker_names

LOG_PATH = LIVE_DIR / "live_chats.txt"
move_legacy(PROJECT_ROOT / "data" / "live_chats.txt", LOG_PATH)

_lock = threading.Lock()
# shortcut: the lock is per-process; the REPL and the daemon could in theory
# append at the same moment. Upgrade to a file lock if the two ever race.


def _one_line(text):
    # The corpus is one turn per line; a multi-line reply would parse as a new
    # (label-less) turn.
    return " ".join(line.strip() for line in text.splitlines() if line.strip())


def append_turn(user_text=None, her_text=None, path=LOG_PATH, now=None):
    """Append one exchange; a quiet gap starts a new conversation.

    Callers pass plain text; this module owns the format. The gap is measured
    from the file's mtime, so a daemon restart does not need extra state.
    """
    user_name, her_name = speaker_names()
    lines = []
    if user_text:
        lines.append(f"{user_name}: {_one_line(user_text)}")
    if her_text:
        lines.append(f"{her_name}: {_one_line(her_text)}")
    if not lines:
        return
    path = Path(path)
    now = time.time() if now is None else now
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        gap = path.exists() and \
            now - path.stat().st_mtime > CONVERSATION_GAP_MINUTES * 60
        with path.open("a", encoding="utf-8") as fh:
            if gap:
                fh.write("\n")
            fh.write("\n".join(lines) + "\n")
