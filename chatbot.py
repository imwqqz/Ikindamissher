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
    HER_SPEAKER,
    USER_SPEAKER,
    build_corpus,
    build_model,
    bpe_from_state,
    detect_user,
    parse_wa,
    persona_lines,
)

BOT_LABEL = "her"


def clip_msg(msg, max_tokens, tokenizer):
    toks = tokenizer.encode(msg)
    if len(toks) <= max_tokens:
        return msg
    return tokenizer.decode(toks[:max_tokens])


def match_score(prompt_toks, question, tokenizer):
    q = tokenizer.encode(question)
    common = sum(1 for t in prompt_toks if t in q)
    if common == 0 or not prompt_toks or not q:
        return 0.0
    return common / (len(prompt_toks) ** 0.5 * len(q) ** 0.5)


# Retrieve similar users1 <=> her exchanges as in-context demos (T5, section 2.1 https://arxiv.org/abs/2005.14165)
def few_shot_pairs(text, shots, demo_len, tokenizer, prompt=""):
    pairs = chat_pairs(text)
    if not pairs:
        return "", 0
    selected = pick_demos(pairs, prompt, shots, tokenizer)
    context = "\n".join(
        f"{USER_SPEAKER}: {clip_msg(q, demo_len, tokenizer)}\n"
        f"{HER_SPEAKER}: {clip_msg(a, demo_len, tokenizer)}"
        for q, a in selected
    )
    return context, len(selected)


def chat_pairs(text):
    # Extract consecutive user->her exchanges from a normalized corpus
    turns = []
    for line in text.splitlines():
        match = re.match(rf"^({USER_SPEAKER}|{HER_SPEAKER}): (.*)$", line)
        if match:
            turns.append((match.group(1), match.group(2).strip()))
    pairs = []
    for i in range(1, len(turns)):
        if turns[i][0] == HER_SPEAKER and turns[i - 1][0] == USER_SPEAKER:
            pairs.append((turns[i - 1][1], turns[i][1]))
    return pairs


def pick_demos(pairs, prompt, shots, tokenizer):
    # Rank candidate demos by token overlap with the user's prompt (T5, section 2.1 https://arxiv.org/abs/2005.14165)
    prompt_tokens = tokenizer.encode(prompt) if prompt else []
    if not prompt_tokens or shots <= 0:
        return []
    scored = sorted(
        ((match_score(prompt_tokens, q, tokenizer), (q, a)) for q, a in pairs),
        key=lambda item: item[0],
        reverse=True,
    )
    return [pair for score, pair in scored if score > 0][:shots]


# Text-to-text prompt: persona + demos + history + her hint (T5, section 2.1 https://arxiv.org/abs/2005.14165)
def build_prompt(args, state, tokenizer, prompt, history, corpus_text):
    description = ""
    if Path(args.description).exists():
        description = "\n".join(persona_lines(
            Path(args.description).read_text(encoding="utf-8")))

    demos = ""
    n_shots = 0
    if corpus_text and args.shots > 0:
        demos, n_shots = few_shot_pairs(corpus_text, args.shots,
                                        args.demo_len, tokenizer, prompt)

    max_ctx = state["cfg"]["max_len"] - 40
    parts = [p for p in (description, demos) if p]
    prefix = "\n".join(parts)
    hist = "\n".join(history)
    full = "\n".join(p for p in (prefix, hist, f"{USER_SPEAKER}: {prompt}") if p)
    ids = tokenizer.encode(full + f"\n{HER_SPEAKER}: ")
    if len(ids) > max_ctx:
        ids = ids[-max_ctx:]
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
    # Normalize the export into canonical user1/her turns for demo retrieval
    if not Path(args.data).exists():
        return ""
    raw_text = Path(args.data).read_text(encoding="utf-8")
    user = args.user or detect_user(raw_text, HER_SPEAKER)
    aliases = {user: USER_SPEAKER} if user.lower() != USER_SPEAKER else None
    corpus_text = parse_wa(raw_text, speakers=(USER_SPEAKER, HER_SPEAKER),
                           aliases=aliases)
    if args.shots > 0:
        print(f"few-shot retrieval on {args.data} (user: {user})")
    return corpus_text


def generate_reply(args, state, model, tokenizer, prompt, history, corpus_text, device):
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
    reply = tokenizer.decode(gen).strip()
    reply = re.sub(r"^[:\s]+", "", reply)
    return re.split(rf"(?:\n|{USER_SPEAKER}:)", reply)[0].strip()


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    ap = argparse.ArgumentParser(
        description="chatbot: talk to her using the memory trained by her.py")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--description",
                    default=str(Path(__file__).parent / "data" / "input" / "description.txt"))
    ap.add_argument("--user", default=None,
                    help="your speaker name in the export (default: auto-detect)")
    ap.add_argument("--shots", type=int, default=2,
                    help="real user1/her exchanges matched to your prompt")
    ap.add_argument("--demo-len", type=int, default=20,
                    help="max tokens per demo message")
    ap.add_argument("--max-reply", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repeat-penalty", type=float, default=1.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[Warning] CUDA is not available. Exiting.")
        sys.exit(1)
    print(f"device: {device}")

    state, model, tokenizer = load_assets(args, device)
    corpus_text = load_corpus(args)

    history = []
    print(f"chatting with her - type 'exit' to quit")
    while True:
        prompt = input(f"<{USER_SPEAKER}>: ").strip()
        if not prompt or prompt.lower() in ("exit", "quit"):
            break
        reply = generate_reply(args, state, model, tokenizer, prompt,
                               history, corpus_text, device)
        if reply:
            print(f"{BOT_LABEL}: {reply}")
            history.append(f"{USER_SPEAKER}: {clip_msg(prompt, 15, tokenizer)}")
            history.append(f"{HER_SPEAKER}: {clip_msg(reply, 15, tokenizer)}")
            history = history[-16:]


if __name__ == "__main__":
    main()