import argparse
import random
import re
import sys
from pathlib import Path

import torch

from her import (
    DEFAULT_CKPT,
    DEFAULT_DATA,
    DEFAULT_DESC,
    build_model,
    bpe_from_state,
    configure_speakers,
    detect_user,
    parse_wa,
    persona_lines,
    resolve_config_args,
    speaker_names,
)

CHAT_CONFIG = Path(__file__).parent / "chatbot.config.json"

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


def build_prompt(args, state, tokenizer, prompt, history, corpus_text):
    # Text-to-text prompt: persona + demos + history + her hint
    # (T5, section 2.1 https://arxiv.org/abs/2005.14165)
    user_name, her_name = speaker_names()
    description = ""
    # Path("") resolves to ".", which exists, so an unset --description used to
    # read_text() a directory and die with PermissionError.
    if args.description and Path(args.description).exists():
        description = "\n".join(persona_lines(
            Path(args.description).read_text(encoding="utf-8")))

    demos = ""
    n_shots = 0
    if corpus_text and args.shots > 0:
        demos, n_shots = few_shot_pairs(corpus_text, args.shots,
                                        args.demo_len, tokenizer, prompt)

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
    if len(ids) > max_ctx:
        # Keep the tail but snap forward to a line break, so the prompt never
        # starts mid-subword.
        ids = ids[-max_ctx:]
        nl = set(tokenizer.encode("\n"))
        floor = max_ctx // 2
        starts = [i + 1 for i, tok in enumerate(ids)
                  if tok in nl and len(ids) - (i + 1) >= floor]
        if starts:
            ids = ids[starts[0]:]
    return ids, n_shots


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
    model.load_state_dict(state["model"])
    tokenizer = bpe_from_state(state["tokenizer"])
    note = f", trained {state['step']:,} steps" if state.get("step") else ""
    print(f"loaded checkpoint: {ckpt} (loss {state['loss']:.4f}{note})")
    print(f"objective: {cfg['objective']} | lora: {cfg['lora']} | "
          f"quant: {cfg.get('quant', 'none')}")
    return state, model, tokenizer


def load_corpus(args):
    # Normalize the export into canonical user/persona turns for demo retrieval
    user_name, _ = speaker_names()
    if not Path(args.data).exists():
        return ""
    raw = Path(args.data).read_text(encoding="utf-8")
    user = args.user or detect_user(raw)
    aliases = {user.lower(): user_name} if user.lower() != user_name else None
    corpus_text = parse_wa(raw, aliases=aliases)
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
    return re.split(rf"(?:\n|{re.escape(user_name)}:)", reply)[0].strip()


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
    ap.add_argument("--config", default=None, metavar="FILE",
                    help="JSON file of defaults; explicit flags still win "
                         f"(default: {CHAT_CONFIG.name} if it exists)")
    ap.add_argument("--dump-config", default=None, metavar="FILE",
                    help="write every option and its default to FILE, then exit")
    return ap


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    args = resolve_config_args(build_parser(), argv, CHAT_CONFIG)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[Warning] CUDA is not available. Exiting.")
        sys.exit(1)
    print(f"device: {device}")

    state, model, tokenizer = load_assets(args, device)
    # The checkpoint is the only record of how turns were labelled; never guess.
    cfg = state["cfg"]
    configure_speakers(args.user_speaker or cfg.get("user_speaker"),
                       args.her_speaker or cfg.get("her_speaker"))
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

    corpus_text = load_corpus(args)

    history = []
    print(f"chatting with her as {her_name!r} - type 'exit' to quit")
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


if __name__ == "__main__":
    main()
