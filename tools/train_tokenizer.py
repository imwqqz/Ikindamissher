import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import (CHAT_CONFIG, DEFAULT_BPE, DEFAULT_DATA, DEFAULT_DESC,
                    load_config)
from data import (configure_speakers, detect_user, parse_wa, persona_lines,
                  speaker_names)
from tokenizer import BytePairEncoder


def build_parser():
    ap = argparse.ArgumentParser(
        description="build the retrieval BPE tokenizer")
    ap.add_argument("--data", default=str(DEFAULT_DATA),
                    help="WhatsApp export to train on")
    ap.add_argument("--description", default=str(DEFAULT_DESC),
                    help="persona fact sheet, appended to the corpus")
    ap.add_argument("--out", default=str(DEFAULT_BPE))
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--user", default=None,
                    help="your speaker name in the export (default: auto-detect)")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = load_config(CHAT_CONFIG) if CHAT_CONFIG.exists() else {}
    configure_speakers(cfg.get("user_speaker"), cfg.get("her_speaker"))
    user_name, her_name = speaker_names()
    if not user_name or not her_name:
        print(f"no speaker names configured; set user_speaker and her_speaker "
              f"in {CHAT_CONFIG}")
        return 1

    raw = Path(args.data).read_text(encoding="utf-8")
    user = args.user or detect_user(raw)
    corpus = parse_wa(raw, aliases={user.lower(): user_name})
    desc = Path(args.description)
    if desc.exists():
        corpus += "\n".join(persona_lines(desc.read_text(encoding="utf-8"))) + "\n"

    tok = BytePairEncoder(n_sentinels=100)
    tok.train(corpus, args.vocab_size)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(out)
    print(f"tokenizer saved: {out} ({tok.vocab_size:,} tokens)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
