import argparse
import json
import shutil
import sys
import time
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent
_ROOT = _TOOLS.parent
for _p in (str(_ROOT / "src"), str(_ROOT), str(_TOOLS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import export_sft
from config import CHAT_CONFIG, LIVE_DIR, PROJECT_ROOT, load_config
from data import configure_speakers, speaker_names

LIVE = LIVE_DIR / "live_chats.txt"
SFT = PROJECT_ROOT / "data" / "sft.json"
COMBINED = PROJECT_ROOT / "data" / "sft_combined.json"


def live_rows(text, user, her_name):
    turns = []
    for line in text.splitlines():
        name, sep, msg = line.partition(": ")
        if sep and msg.strip():
            turns.append((name.strip().lower(), msg.strip()))
    convs, _ = export_sft.group_conversations(turns, user, her_name)
    return [{"messages": conv} for conv in convs]


def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dedupe(rows):
    # retraining on an uncleared log must not multiply exchanges
    seen, out = set(), []
    for row in rows:
        key = json.dumps(row, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def write_rows(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def backup_adapter(adapter):
    if not adapter.exists():
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = adapter.parent / f"{adapter.name}.{stamp}.bak"
    shutil.copytree(adapter, dest)
    return dest


def build_parser():
    ap = argparse.ArgumentParser(
        description="fold live chats into the SFT set and retrain the adapter")
    ap.add_argument("--live", default=str(LIVE))
    ap.add_argument("--sft", default=str(SFT),
                    help="existing export (tools/export_sft.py --out data/sft.json)")
    ap.add_argument("--combined", default=str(COMBINED),
                    help="where to write export rows + live rows")
    ap.add_argument("--out", default=None,
                    help="adapter dir (default: data/her_qwen_adapter)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=None,
                    help="override the qwen_lm default learning rate")
    ap.add_argument("--no-backup", action="store_true")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = load_config(CHAT_CONFIG) if CHAT_CONFIG.exists() else {}
    configure_speakers(cfg.get("user_speaker"), cfg.get("her_speaker"))
    user, her_name = speaker_names()
    if not user or not her_name:
        print(f"set user_speaker and her_speaker in {CHAT_CONFIG} first")
        return 1

    live = Path(args.live)
    text = live.read_text(encoding="utf-8") if live.exists() else ""
    if not text.strip():
        print(f"no live chats yet ({live}); nothing to train")
        return 1

    rows = live_rows(text, user, her_name)
    if not rows:
        print(f"{live} has no complete user->her exchanges; nothing to train")
        return 1
    old = read_rows(Path(args.sft))
    combined = dedupe(old + rows)
    write_rows(combined, Path(args.combined))
    print(f"live rows: {len(rows)}")
    print(f"combined: {len(combined)} rows "
          f"({len(old)} export + {len(rows)} live) -> {args.combined}")

    import engines.qwen.qwen_lm as qwen_lm
    adapter = Path(args.out) if args.out else qwen_lm.ADAPTER_DIR
    if not args.no_backup:
        backup = backup_adapter(adapter)
        if backup:
            print(f"adapter backup: {backup}")

    train_args = qwen_lm.build_parser().parse_args(
        ["--data", str(args.combined), "--out", str(adapter),
         "--epochs", str(args.epochs)])
    if args.lr is not None:
        train_args.lr = args.lr
    qwen_lm.train(train_args)
    print("done. restart the daemon to serve the new adapter.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
