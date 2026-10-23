import argparse
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

from config import (
    CHAT_CONFIG,
    DEFAULT_DATA,
    DEFAULT_DESC,
    request_cpu_fallback,
    resolve_config_args,
)
from data import configure_speakers, speaker_names
from chatlog import append_turn
from engines.dispatch import (
    build_demo_index,
    clip_msg,
    generate_proactive,
    generate_reply,
    load_assets,
    load_corpus,
    load_openers,
    resolve_max_reply,
)

BOT_LABEL = "her"


def build_parser():
    ap = argparse.ArgumentParser(
        description="repl: talk to her using the qwen adapter")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--description", default=str(DEFAULT_DESC))
    ap.add_argument("--user", default=None,
                    help="your speaker name in the export (default: auto-detect)")
    ap.add_argument("--user-speaker", default=None,
                    help="canonical name for you, as used when training")
    ap.add_argument("--her-speaker", default=None,
                    help="canonical name for the persona; must match training, "
                         "and is kept out of the source on purpose")
    ap.add_argument("--adapter", default=None,
                    help="adapter dir (default: data/her_qwen_adapter)")
    ap.add_argument("--shots", type=int, default=2,
                    help="real user/persona exchanges matched to your prompt")
    ap.add_argument("--demo-len", type=int, default=20,
                    help="max tokens per demo message")
    ap.add_argument("--max-reply", type=int, default=40)
    ap.add_argument("--candidates", type=int, default=3,
                    help="sample this many replies and keep the one that best "
                         "matches her retrieved turns (all are logged)")
    ap.add_argument("--think", action="store_true",
                    help="enable Qwen3 reasoning: slower, and the thought is "
                         "logged only (never spoken)")
    ap.add_argument("--turn-log", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="print every turn: incoming message, sampled "
                         "candidates with scores, chosen reply")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repeat-penalty", type=float, default=1.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true",
                    help="run on CPU when CUDA is unavailable, skipping the "
                         "interactive prompt")
    ap.add_argument("--proactive", action="store_true",
                    help="she speaks first at startup, like her openers in the "
                         "export")
    ap.add_argument("--proactive-only", action="store_true",
                    help="print one opener and exit (for an external scheduler)")
    ap.add_argument("--proactive-hour", type=int, default=None,
                    help="hour 0-23 to imitate; default: the current hour")
    ap.add_argument("--proactive-window", type=int, default=3,
                    help="hours around --proactive-hour to draw openers from")
    ap.add_argument("--proactive-gap", type=int, default=30,
                    help="silence in minutes that counts as her opening a "
                         "conversation")
    ap.add_argument("--config", default=None, metavar="FILE",
                    help="JSON file of defaults; explicit flags still win "
                         f"(default: {CHAT_CONFIG.name} if it exists)")
    ap.add_argument("--dump-config", default=None, metavar="FILE",
                    help="write every option and its default to FILE, then exit")
    return ap


def _init_stdout():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass


def _setup_device(force_cpu=False):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif force_cpu or request_cpu_fallback():
        print("[Warning] CUDA is not available; running on CPU (slow).")
        device = torch.device("cpu")
    else:
        print("CUDA is not available. Exiting.")
        sys.exit(1)
    print(f"device: {device}")
    return device


def _configure_speakers(args):
    # labels must match what the adapter was trained with; never guess
    configure_speakers(args.user_speaker, args.her_speaker)
    user_name, her_name = speaker_names()
    if not user_name or not her_name:
        raise SystemExit(
            "no speaker names configured. Set user_speaker and her_speaker in "
            f"{CHAT_CONFIG.name}, or pass them:\n"
            '  uv run python client/repl.py --user-speaker "<name>" '
            '--her-speaker "<name>"')
    return user_name, her_name


def _chat_loop(args, state, model, tokenizer, history, demo_index, device,
               user_name, her_name):
    while True:
        try:
            prompt = input(f"<{user_name}>: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt or prompt.lower() in ("exit", "quit"):
            break
        reply = generate_reply(args, state, model, tokenizer, prompt,
                               history, demo_index, device, source="repl")
        if reply:
            print(f"{BOT_LABEL}: {reply}")
            history.append(f"{user_name}: {clip_msg(prompt, 15, tokenizer)}")
            history.append(f"{her_name}: {clip_msg(reply, 15, tokenizer)}")
            history = history[-16:]
            append_turn(prompt, reply)


def main(argv=None):
    _init_stdout()

    args = resolve_config_args(build_parser(), argv, CHAT_CONFIG)
    args = resolve_max_reply(args)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = _setup_device(args.cpu)

    state, model, tokenizer = load_assets(args, device)
    user_name, her_name = _configure_speakers(args)

    corpus_text = load_corpus(args)
    demo_index = build_demo_index(corpus_text, tokenizer) if corpus_text else None

    history = []
    print(f"chatting with her as {her_name!r} - type 'exit' to quit")
    if args.proactive or args.proactive_only:
        opener = generate_proactive(args, state, model, tokenizer,
                                    load_openers(args), device, source="repl")
        if opener:
            print(f"{BOT_LABEL}: {opener}")
            history.append(f"{her_name}: {clip_msg(opener, 15, tokenizer)}")
            append_turn(None, opener)
        if args.proactive_only:
            return
    _chat_loop(args, state, model, tokenizer, history, demo_index, device,
               user_name, her_name)


if __name__ == "__main__":
    main()
