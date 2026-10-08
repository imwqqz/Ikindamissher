import argparse
import random
import re
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

from config import (
    DEFAULT_CKPT,
    DEFAULT_DATA,
    DEFAULT_DESC,
    PROJECT_ROOT,
    request_cpu_fallback,
    resolve_config_args,
)
from data import (
    _split_turns,
    configure_speakers,
    detect_user,
    first_turns,
    openers_near_hour,
    parse_wa,
    persona_lines,
    speaker_names,
)
from model import build_model
from tokenizer import bpe_from_state

CHAT_CONFIG = PROJECT_ROOT / "configs" / "chatbot.config.json"

BOT_LABEL = "her"


def clip_msg(msg, max_tokens, tokenizer):
    # Re-encode at a character boundary: slicing mid-sequence yields U+FFFD.
    msg = msg.strip()
    toks = tokenizer.encode(msg)
    if len(toks) <= max_tokens:
        return msg
    text = tokenizer.decode(toks[:max_tokens])
    # decode() already replaced the broken tail; drop it and anything partial.
    text = text.replace("\ufffd", "")
    if text and not text[-1].isspace():
        text += "..."
    return text.strip()


def chat_pairs(text):
    # Consecutive user->persona exchanges from a normalized corpus.
    user_name, her_name = speaker_names()
    turn = re.compile(rf"^({re.escape(user_name)}|{re.escape(her_name)}): (.*)$")
    turns = []
    for line in text.splitlines():
        m = turn.match(line)
        if m:
            turns.append((m.group(1), m.group(2).strip()))
    return [(turns[i - 1][1], turns[i][1]) for i in range(1, len(turns))
            if turns[i][0] == her_name and turns[i - 1][0] == user_name]


def pick_demos(pairs, prompt, shots, tokenizer):
    # Rank candidate demos by token overlap with the prompt
    # (T5, section 2.1 https://arxiv.org/abs/2005.14165)
    prompt_tokens = tokenizer.encode(prompt) if prompt else []
    if not prompt_tokens or shots <= 0:
        return []

    def overlap(question):
        q = tokenizer.encode(question)
        if not q:
            return 0.0
        common = sum(1 for t in prompt_tokens if t in q)
        return common / (len(prompt_tokens) ** 0.5 * len(q) ** 0.5)

    scored = sorted(((overlap(q), (q, a)) for q, a in pairs),
                    key=lambda item: item[0], reverse=True)
    return [pair for score, pair in scored if score > 0][:shots]


def few_shot_pairs(text, shots, demo_len, tokenizer, prompt=""):
    # Retrieve similar user1<=>her exchanges as in-context demos
    # (T5, section 2.1 https://arxiv.org/abs/2005.14165)
    user_name, her_name = speaker_names()
    pairs = chat_pairs(text)
    if not pairs:
        return "", 0
    selected = pick_demos(pairs, prompt, shots, tokenizer)
    context = "\n".join(
        f"{user_name}: {clip_msg(q, demo_len, tokenizer)}\n"
        f"{her_name}: {clip_msg(a, demo_len, tokenizer)}"
        for q, a in selected
    )
    return context, len(selected)


def _read_description(args):
    # Path("") resolves to ".", which exists, so an unset --description used to
    # read_text() a directory and die with PermissionError.
    if args.description and Path(args.description).exists():
        return "\n".join(persona_lines(
            Path(args.description).read_text(encoding="utf-8")))
    return ""


def _select_demos(corpus_text, args, prompt, tokenizer):
    if corpus_text and args.shots > 0:
        return few_shot_pairs(corpus_text, args.shots,
                              args.demo_len, tokenizer, prompt)
    return "", 0


def _trim_prompt(ids, max_ctx, tokenizer):
    if len(ids) <= max_ctx:
        return ids
    # Keep the tail but snap forward to a line break, so the prompt never
    # starts mid-subword.
    ids = ids[-max_ctx:]
    nl = set(tokenizer.encode("\n"))
    floor = max_ctx // 2
    starts = [i + 1 for i, tok in enumerate(ids)
              if tok in nl and len(ids) - (i + 1) >= floor]
    if starts:
        ids = ids[starts[0]:]
    return ids


def build_prompt(args, state, tokenizer, prompt, history, corpus_text):
    # Text-to-text prompt: persona + demos + history + her hint
    # (T5, section 2.1 https://arxiv.org/abs/2005.14165)
    user_name, her_name = speaker_names()
    description = _read_description(args)
    demos, n_shots = _select_demos(corpus_text, args, prompt, tokenizer)

    # Headroom must cover the reply, or RoPE overflows past max_len.
    max_ctx = state["cfg"]["max_len"] - args.max_reply - 1
    if max_ctx < 1:
        raise SystemExit(
            f"--max-reply {args.max_reply} leaves no room in a context of "
            f"{state['cfg']['max_len']}; lower it")
    prefix = "\n".join(p for p in (description, demos) if p)
    hist = "\n".join(history)
    full = "\n".join(p for p in (prefix, hist, f"{user_name}: {prompt}") if p)
    ids = tokenizer.encode(full + f"\n{her_name}: ")
    return _trim_prompt(ids, max_ctx, tokenizer), n_shots


def load_assets(args, device):
    # Load checkpoint, model, and tokenizer; exits if no checkpoint exists
    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        print(f"checkpoint not found at {ckpt}")
        print("train the memory first with: python her.py")
        raise SystemExit(1)

    state = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = state["cfg"]
    model = build_model(cfg).to(device).eval()
    if state.get("ema") and not args.raw_weights:
        model.load_state_dict(state["ema"])
        print("weights: using the checkpoint's EMA average")
    else:
        model.load_state_dict(state["model"])
    tokenizer = bpe_from_state(state["tokenizer"])
    note = f", trained {state['step']:,} steps" if state.get("step") else ""
    print(f"loaded checkpoint: {ckpt} (loss {state['loss']:.4f}{note})")
    print(f"objective: {cfg['objective']} | lora: {cfg['lora']} | "
          f"quant: {cfg.get('quant', 'none')}")
    return state, model, tokenizer


def load_corpus(args, cfg):
    # Normalize the export into canonical user/persona turns for demo retrieval.
    # The tail held out for validation during training is dropped here too, so a
    # demo can never quote text the model was not asked to learn from.
    user_name, _ = speaker_names()
    if not Path(args.data).exists():
        return ""
    raw = Path(args.data).read_text(encoding="utf-8")
    user = args.user or detect_user(raw)
    aliases = {user.lower(): user_name} if user.lower() != user_name else None
    corpus_text = parse_wa(raw, aliases=aliases)
    val_fraction = float(cfg.get("val_fraction", 0.0) or 0.0)
    if val_fraction > 0:
        corpus_text, held_out = _split_turns(corpus_text, val_fraction)
        if args.shots > 0 and held_out:
            print("demos: holding out the validation tail from retrieval")
    if args.shots > 0:
        print(f"few-shot retrieval on {args.data} (user: {user})")
    return corpus_text


def generate_reply(args, state, model, tokenizer, prompt, history, corpus_text, device):
    user_name, _ = speaker_names()
    ids, _ = build_prompt(args, state, tokenizer, prompt, history, corpus_text)
    gen = model.generate(
        torch.tensor([ids], device=device),
        max_new=args.max_reply,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repeat_penalty=args.repeat_penalty,
        stop_ids=tokenizer.stop_ids(),
    )
    reply = re.sub(r"^[:\s]+", "", tokenizer.decode(gen).strip())
    # Keep the newlines inside her turn (they separate her messages); cut only
    # if generation ran on into a user turn, which the sentinel should prevent.
    return re.split(rf"\n{re.escape(user_name)}:", reply)[0].strip()


def load_openers(args):
    # Her first-of-conversation messages, straight from the raw export, so a
    # proactive line can imitate how and when she actually opens a chat.
    if not Path(args.data).exists():
        return []
    raw = Path(args.data).read_text(encoding="utf-8")
    user_name, her_name = speaker_names()
    user = args.user or detect_user(raw)
    aliases = {user.lower(): user_name} if user.lower() != user_name else None
    return first_turns(raw, label=her_name, aliases=aliases,
                       gap_minutes=args.proactive_gap)


def _opener_demos(candidates, args, tokenizer):
    return [clip_msg(opener["text"], args.demo_len, tokenizer)
            for opener in candidates[:max(0, args.shots)]]


def generate_proactive(args, state, model, tokenizer, openers, device):
    """One unprompted message, biased toward how she opens chats at this hour."""
    _, her_name = speaker_names()
    off_hour = args.proactive_hour if args.proactive_hour is not None \
        else time.localtime().tm_hour
    candidates = openers_near_hour(openers, off_hour, args.proactive_window)
    demos = _opener_demos(candidates, args, tokenizer)
    head = _read_description(args)
    lines = ([head] if head else []) + [f"{her_name}: {d}" for d in demos]
    max_ctx = max(1, state["cfg"]["max_len"] - args.max_reply - 1)
    ids = _trim_prompt(tokenizer.encode("\n".join(lines) + f"\n{her_name}: "),
                       max_ctx, tokenizer)
    gen = model.generate(
        torch.tensor([ids], device=device),
        max_new=args.max_reply,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repeat_penalty=args.repeat_penalty,
        stop_ids=tokenizer.stop_ids(),
    )
    user_name, _ = speaker_names()
    reply = re.sub(r"^[:\s]+", "", tokenizer.decode(gen).strip())
    reply = re.split(rf"\n{re.escape(user_name)}:", reply)[0].strip()
    if reply:
        return reply
    # Sampling can return nothing (immediate stop); a real opener is faithful.
    return candidates[0]["text"] if candidates else ""


def build_parser():
    ap = argparse.ArgumentParser(
        description="chatbot: talk to her using the memory trained by her.py")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--description", default=str(DEFAULT_DESC))
    ap.add_argument("--user", default=None,
                    help="your speaker name in the export (default: auto-detect)")
    ap.add_argument("--user-speaker", default=None,
                    help="canonical name for you, as used when training")
    ap.add_argument("--her-speaker", default=None,
                    help="canonical name for the persona; must match training, "
                         "and is kept out of the source on purpose")
    ap.add_argument("--shots", type=int, default=2,
                    help="real user/persona exchanges matched to your prompt")
    ap.add_argument("--demo-len", type=int, default=20,
                    help="max tokens per demo message")
    ap.add_argument("--max-reply", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repeat-penalty", type=float, default=1.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--raw-weights", action="store_true",
                    help="ignore the checkpoint's EMA average and load the "
                         "raw weights instead")
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


def _pick_speakers(args, cfg):
    return args.user_speaker or cfg.get("user_speaker"), \
        args.her_speaker or cfg.get("her_speaker")


def _configure_from_checkpoint(args, cfg):
    # The checkpoint is the only record of how turns were labelled; never guess.
    configure_speakers(*_pick_speakers(args, cfg))
    user_name, her_name = speaker_names()
    if not user_name or not her_name:
        raise SystemExit(
            "this checkpoint predates speaker persistence and no names were "
            'given. Pass the same values you trained with, e.g.\n'
            '  py chatbot.py --user-speaker "<name>" --her-speaker "<name>"\n'
            "Retraining stores them in the checkpoint, so this is only "
            "needed for older files.")
    if (cfg.get("her_speaker") and her_name != cfg["her_speaker"]) or \
       (cfg.get("user_speaker") and user_name != cfg["user_speaker"]):
        print(f"note: overriding checkpoint labels (trained as "
              f"{cfg.get('user_speaker')!r}/{cfg.get('her_speaker')!r})")
    return user_name, her_name


def _chat_loop(args, state, model, tokenizer, history, corpus_text, device,
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
                               history, corpus_text, device)
        if reply:
            print(f"{BOT_LABEL}: {reply}")
            history.append(f"{user_name}: {clip_msg(prompt, 15, tokenizer)}")
            history.append(f"{her_name}: {clip_msg(reply, 15, tokenizer)}")
            history = history[-16:]


def main(argv=None):
    _init_stdout()

    args = resolve_config_args(build_parser(), argv, CHAT_CONFIG)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = _setup_device(args.cpu)

    state, model, tokenizer = load_assets(args, device)
    cfg = state["cfg"]
    user_name, her_name = _configure_from_checkpoint(args, cfg)

    corpus_text = load_corpus(args, cfg)

    history = []
    print(f"chatting with her as {her_name!r} - type 'exit' to quit")
    if args.proactive or args.proactive_only:
        opener = generate_proactive(args, state, model, tokenizer,
                                    load_openers(args), device)
        if opener:
            print(f"{BOT_LABEL}: {opener}")
            history.append(f"{her_name}: {clip_msg(opener, 15, tokenizer)}")
        if args.proactive_only:
            return
    _chat_loop(args, state, model, tokenizer, history, corpus_text, device,
               user_name, her_name)


if __name__ == "__main__":
    main()
