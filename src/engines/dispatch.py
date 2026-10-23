"""The engine door: loads a brain, builds its prompts, generates replies.

One entry point for both brains (engines/qwen, engines/legacy). Everything
here is engine-agnostic plumbing - the client (client/repl.py) and the daemon
(server/service.py) only touch load_assets / generate_reply /
generate_proactive and never know which model answered.
"""
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

from config import DEFAULT_BPE
from data import (
    _split_turns,
    detect_user,
    first_turns,
    openers_near_hour,
    parse_wa,
    persona_lines,
    speaker_names,
)
from engines.legacy.model import build_model
from tokenizer import BytePairEncoder, bpe_from_state


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


def conversations(text):
    """Split a normalized corpus into conversations (blank-line separated)."""
    user_name, her_name = speaker_names()
    turn = re.compile(rf"^({re.escape(user_name)}|{re.escape(her_name)}): (.*)$")
    grouped = []
    current = []
    for line in text.splitlines():
        if not line.strip():
            if current:
                grouped.append(current)
                current = []
            continue
        match = turn.match(line)
        if match:
            current.append((match.group(1), match.group(2).strip()))
    if current:
        grouped.append(current)
    return grouped


def _demo_candidates(grouped, user_name, her_name):
    """(question, answer, conversation index, position) for each user->her pair."""
    candidates = []
    for index, conversation in enumerate(grouped):
        for position in range(len(conversation) - 1):
            speaker, question = conversation[position]
            next_speaker, answer = conversation[position + 1]
            if speaker == user_name and next_speaker == her_name:
                candidates.append((question, answer, index, position))
    return candidates


def _idf_weights(candidates, tokenizer):
    """Inverse document frequency per token across the demo questions.

    Rare tokens (a topic word) weigh more than filler, so a demo that shares
    the actual subject beats a generic greeting that matches everything.
    """
    document_counts = Counter()
    for candidate in candidates:
        document_counts.update(set(tokenizer.encode(candidate[0])))
    total = max(1, len(candidates))
    return {token: math.log((total + 1) / (count + 1)) + 1.0
            for token, count in document_counts.items()}


def build_demo_index(text, tokenizer):
    """Static retrieval state: conversations, user->her candidates, IDF weights.

    All three depend only on the corpus and the tokenizer, so they are built
    once per session instead of re-derived on every typed prompt.
    """
    user_name, her_name = speaker_names()
    grouped = conversations(text)
    candidates = _demo_candidates(grouped, user_name, her_name)
    return grouped, candidates, _idf_weights(candidates, tokenizer)


def _weighted_overlap(prompt_tokens, question_tokens, weights):
    """Cosine similarity over IDF-weighted token multisets."""
    if not prompt_tokens or not question_tokens:
        return 0.0
    prompt_counts = Counter(prompt_tokens)
    question_counts = Counter(question_tokens)
    shared = sum(weights.get(token, 1.0)
                 * min(prompt_counts[token], question_counts[token])
                 for token in prompt_counts)
    norm_p = math.sqrt(sum(weights.get(t, 1.0) * c * c
                           for t, c in prompt_counts.items()))
    norm_q = math.sqrt(sum(weights.get(t, 1.0) * c * c
                           for t, c in question_counts.items()))
    return shared / (norm_p * norm_q) if norm_p and norm_q else 0.0


def pick_demos(candidates, weights, prompt, shots, tokenizer):
    # Rank candidate demos by IDF-weighted token overlap with the prompt
    # (tf-idf retrieval, weighted by inverse document frequency)
    prompt_tokens = tokenizer.encode(prompt) if prompt else []
    if not prompt_tokens or shots <= 0:
        return []
    seen = set()
    scored = []
    for question, answer, index, position in candidates:
        key = question.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        tokens = tokenizer.encode(question)
        score = _weighted_overlap(prompt_tokens, tokens, weights)
        # Ties: a question of similar length is a better-shaped demo.
        scored.append((score, -abs(len(tokens) - len(prompt_tokens)),
                       index, (question, answer, index, position)))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    picked = []
    used = set()
    for score, _, index, candidate in scored:
        # Skip a lower-ranked duplicate rather than dropping the slot: the
        # next conversation's best match takes it.
        if score <= 0 or index in used:
            continue
        used.add(index)
        picked.append(candidate)
        if len(picked) == shots:
            break
    return picked


def _chain_demo(conversation, position, demo_len, tokenizer, max_lines=6):
    """A contiguous slice of one conversation, starting at the matched pair.

    Consecutive turns from the *same* conversation read as a single exchange
    instead of unrelated pairs stitched together, which is what made replies
    drift to another conversation's topic.
    """
    lines = []
    for speaker, message in conversation[position:position + max_lines]:
        text = clip_msg(message, demo_len, tokenizer)
        if not text:
            break
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def few_shot_pairs(index, shots, demo_len, tokenizer, prompt=""):
    # Retrieve similar user<=>her exchanges as in-context demos, each chained
    # within its own conversation (T5's in-context framing, section 2.1)
    grouped, candidates, weights = index
    if not candidates:
        return "", 0
    selected = pick_demos(candidates, weights, prompt, shots, tokenizer)
    blocks = [b for b in (_chain_demo(grouped[index], position, demo_len,
                                      tokenizer)
                          for _, _, index, position in selected) if b]
    # Blank line between demos: the same scene break training learned.
    return "\n\n".join(blocks), len(blocks)


def _read_description(args):
    # Path("") resolves to ".", which exists, so an unset --description used to
    # read_text() a directory and die with PermissionError.
    if args.description and Path(args.description).exists():
        return "\n".join(persona_lines(
            Path(args.description).read_text(encoding="utf-8")))
    return ""


def _select_demos(demo_index, args, prompt, tokenizer):
    if demo_index and demo_index[1] and args.shots > 0:
        return few_shot_pairs(demo_index, args.shots,
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


def build_prompt(args, state, tokenizer, prompt, history, demo_index):
    # Text-to-text prompt: persona + demos + history + her hint
    # (T5, section 2.1 https://arxiv.org/abs/2005.14165)
    user_name, her_name = speaker_names()
    description = _read_description(args)
    demos, n_shots = _select_demos(demo_index, args, prompt, tokenizer)

    # Headroom must cover the reply, or RoPE overflows past max_len.
    max_ctx = state["cfg"]["max_len"] - args.max_reply - 1
    if max_ctx < 1:
        raise SystemExit(
            f"--max-reply {args.max_reply} leaves no room in a context of "
            f"{state['cfg']['max_len']}; lower it")
    # Persona and demos are context, not the live conversation: a blank line
    # separates them from the history, which stays contiguous with the prompt.
    prefix = "\n\n".join(p for p in (description, demos) if p)
    tail = "\n".join([*history, f"{user_name}: {prompt}"])
    full = f"{prefix}\n\n{tail}" if prefix else tail
    # No trailing space after the label: a bare space encodes as a token that
    # never appears in training (a space always merges into the next word), so
    # the model would start from an out-of-distribution token.
    ids = tokenizer.encode(full + f"\n{her_name}:")
    return _trim_prompt(ids, max_ctx, tokenizer), n_shots


def load_assets(args, device):
    # Load checkpoint, model, and tokenizer; exits if no checkpoint exists
    if args.engine == "qwen":
        return _load_qwen(args, device)
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


def _load_qwen(args, device):
    """Qwen engine assets: HF model + her adapter + the legacy BPE."""
    import engines.qwen.qwen_lm as qwen_lm
    adapter = Path(args.adapter) if args.adapter else qwen_lm.ADAPTER_DIR
    if not adapter.exists():
        print(f"adapter not found at {adapter}")
        print("train the qwen engine first with: uv run python src/engines/qwen/qwen_lm.py")
        raise SystemExit(1)
    hf_tok, qmodel = qwen_lm.load_engine(adapter=adapter, device=device)
    print(f"qwen engine: adapter {adapter.name}")
    legacy_tok = BytePairEncoder.load(Path(DEFAULT_BPE))
    print("few-shot retrieval on the legacy tokenizer (bpe.json)")
    return {"cfg": {}, "qwen": (hf_tok, qmodel)}, qmodel, legacy_tok


def resolve_max_reply(args):
    # One bump per engine lives here so client/server/eval share it.
    if args.engine == "qwen" and args.max_reply == 40:
        args.max_reply = 80   # the qwen tokenizer packs fewer chars per token
    return args


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


def _clean_reply(text, user_name, her_name):
    """Drop learned speaker labels and stop at the first user turn.

    With the scored turn-end sentinel, her reply is a whole multi-message turn
    and legitimately contains "her: " continuation labels; the REPL prints its
    own prefix, so the inner labels are stripped for display.
    """
    labels = (f"{her_name}:", f"{user_name}:")
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith(f"{user_name}:"):
            break
        for label in labels:
            if line.startswith(label):
                line = line[len(label):].strip()
                break
        line = line.lstrip(":").strip()
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def _history_to_messages(history, user_name, her_name):
    """History lines ("user: x" / "her: y") -> chat template messages."""
    msgs = []
    for line in history:
        if line.startswith(f"{user_name}: "):
            msgs.append({"role": "user",
                         "content": line[len(f"{user_name}: "):]})
        elif line.startswith(f"{her_name}: "):
            msgs.append({"role": "assistant",
                         "content": line[len(f"{her_name}: "):]})
    return msgs


def _qwen_messages(args, prompt, history, demo_index, tokenizer):
    """Qwen engine prompt: persona + demos LIVE INSIDE the system text.

    A chat template has no scene-break blank line between roles, so demos must
    never enter the message list, or she answers to a demo instead of you. The
    system block is the one place plain text (like the legacy prompt head) is
    safe.
    """
    description = _read_description(args)
    demos, _ = _select_demos(demo_index, args, prompt, tokenizer)
    system = "\n\n".join(p for p in (description, demos) if p)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    user_name, her_name = speaker_names()
    messages.extend(_history_to_messages(history, user_name, her_name))
    messages.append({"role": "user", "content": prompt})
    return messages


def _qwen_reply(hf_tok, hf_model, messages, args):
    import engines.qwen.qwen_lm as qwen_lm
    return qwen_lm.reply(
        hf_tok, hf_model, messages,
        max_new_tokens=args.max_reply, temperature=args.temperature,
        top_p=args.top_p, repeat_penalty=args.repeat_penalty)


def generate_reply(args, state, model, tokenizer, prompt, history, demo_index,
                   device):
    if args.engine == "qwen":
        hf_tok, hf_model = state["qwen"]
        messages = _qwen_messages(args, prompt, history, demo_index, tokenizer)
        text = _qwen_reply(hf_tok, hf_model, messages, args)
        user_name, her_name = speaker_names()
        return _clean_reply(text, user_name, her_name)
    user_name, her_name = speaker_names()
    ids, _ = build_prompt(args, state, tokenizer, prompt, history, demo_index)
    gen = model.generate(
        torch.tensor([ids], device=device),
        max_new=args.max_reply,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repeat_penalty=args.repeat_penalty,
        stop_ids=tokenizer.stop_ids(),
    )
    return _clean_reply(tokenizer.decode(gen), user_name, her_name)


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
    if args.engine == "qwen":
        hf_tok, hf_model = state["qwen"]
        # opener demos live inside the system text, same rule as the REPL
        demos = _opener_demos(candidates, args, tokenizer)
        head = _read_description(args)
        lines = ([head] if head else []) + [f"{her_name}: {d}" for d in demos]
        messages = []
        if lines:
            messages.append({"role": "system",
                             "content": "\n\n".join(lines)})
        # the empty user turn is a synthetic "she speaks first" cue; a reply
        # that comes back empty falls back to a real opener like the legacy path
        messages.append({"role": "user", "content": ""})
        text = _qwen_reply(hf_tok, hf_model, messages, args)
        user_name, her_name = speaker_names()
        reply = _clean_reply(text, user_name, her_name)
        if reply:
            return reply
        return candidates[0]["text"] if candidates else ""
    demos = _opener_demos(candidates, args, tokenizer)
    head = _read_description(args)
    lines = ([head] if head else []) + [f"{her_name}: {d}" for d in demos]
    max_ctx = max(1, state["cfg"]["max_len"] - args.max_reply - 1)
    ids = _trim_prompt(tokenizer.encode("\n".join(lines) + f"\n{her_name}:"),
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
    user_name, her_name = speaker_names()
    reply = _clean_reply(tokenizer.decode(gen), user_name, her_name)
    if reply:
        return reply
    # Sampling can return nothing (immediate stop); a real opener is faithful.
    return candidates[0]["text"] if candidates else ""
