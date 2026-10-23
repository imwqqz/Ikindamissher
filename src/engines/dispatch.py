"""The engine door: loads the qwen brain, builds its prompts, generates replies.

Everything here is engine-agnostic plumbing (retrieval, persona, corpus) plus
the qwen call sites. The client (client/repl.py) and the daemon
(server/service.py) only touch load_assets / generate_reply / generate_proactive
and never know which model answered.
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

from config import DEFAULT_BPE
from data import (
    detect_user,
    first_turns,
    openers_near_hour,
    parse_wa,
    persona_lines,
    speaker_names,
)
from tokenizer import BytePairEncoder

from chatlog import LOG_PATH
import engines.qwen.chat as qwen_chat
import turnlog


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


def _context(args, prompt, demo_index, tokenizer):
    """(persona, demos, system): what she replies from, kept for the audit log."""
    description = _read_description(args)
    demos, _ = _select_demos(demo_index, args, prompt, tokenizer)
    system = "\n\n".join(p for p in (description, demos) if p)
    return description, demos, system


def _demo_answers(demo_index, prompt, tokenizer, shots=2):
    """Her own turns from the demos retrieved for this prompt."""
    if not demo_index or not demo_index[1]:
        return []
    picked = pick_demos(demo_index[1], demo_index[2], prompt, shots, tokenizer)
    return [answer for _, answer, _, _ in picked]


def _sample(args, hf_tok, hf_model, messages, user_name, her_name):
    """N cleaned candidate replies plus the first thought (audit material)."""
    candidates, thought = [], ""
    for _ in range(max(1, getattr(args, "candidates", 1))):
        visible, think = qwen_chat.generate(hf_tok, hf_model, messages, args)
        candidates.append(qwen_chat.clean_reply(visible, user_name, her_name))
        thought = thought or think
    return candidates, thought


def _choose(candidates, reference, tokenizer, weights):
    """Pick the sample that best matches how she actually talks.

    The reference is her own retrieved turns; the IDF-weighted overlap is the
    same scorer retrieval uses, so the pick is explainable in the log.
    """
    if reference:
        ref_tokens = tokenizer.encode(reference)
        scored = [(text, _weighted_overlap(tokenizer.encode(text), ref_tokens,
                                           weights))
                  for text in candidates]
    else:
        scored = [(text, 0.0) for text in candidates]
    if not any(text for text, _ in scored):
        return "", "all candidates empty", scored
    chosen, score = max(scored, key=lambda item: (bool(item[0]), item[1]))
    why = (f"overlap {score:.3f} vs her retrieved turns" if reference
           else "first non-empty sample (no reference turns)")
    return chosen, why, scored


def _log_turn(args, source, user_text, scored, chosen, why, context, thought):
    """The live feed (console) and the permanent record (SQLite) in one place."""
    if getattr(args, "turn_log", True):
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {source} in: {user_text or '(proactive)'}")
        for text, score in scored:
            mark = "*" if text == chosen else " "
            print(f"     {mark} {score:.3f}  {text!r}")
        print(f"     -> {chosen or '(silence)'} ({why})")
        if thought:
            print(f"     thought: {thought}")
    try:
        turnlog.record(source, user_text, scored, chosen, why, context,
                       thought)
    except Exception as exc:  # telemetry must never take the chat down
        print(f"[telemetry] {exc}")


def generate_reply(args, state, model, tokenizer, prompt, history, demo_index,
                   device, source="unknown"):
    hf_tok, hf_model = state["qwen"]
    description, demos, system = _context(args, prompt, demo_index, tokenizer)
    messages = qwen_chat.build_messages(system, prompt, history)
    user_name, her_name = speaker_names()
    candidates, thought = _sample(args, hf_tok, hf_model, messages,
                                  user_name, her_name)
    reference = " ".join(_demo_answers(demo_index, prompt, tokenizer))
    weights = demo_index[2] if demo_index else {}
    chosen, why, scored = _choose(candidates, reference, tokenizer, weights)
    _log_turn(args, source, prompt, scored, chosen, why,
              {"persona": description, "demos": demos,
               "history": list(history)}, thought)
    return chosen


def load_assets(args, device):
    # Load the HF model + her adapter; exits if the adapter is missing
    return _load_qwen(args, device)


def _load_qwen(args, device):
    """Qwen engine assets: HF model + her adapter + the retrieval BPE."""
    import engines.qwen.qwen_lm as qwen_lm
    adapter = Path(args.adapter) if args.adapter else qwen_lm.ADAPTER_DIR
    if not adapter.exists():
        print(f"adapter not found at {adapter}")
        print("train the qwen engine first with: uv run python src/engines/qwen/qwen_lm.py")
        raise SystemExit(1)
    hf_tok, qmodel = qwen_lm.load_engine(adapter=adapter, device=device)
    print(f"qwen engine: adapter {adapter.name}")
    tok = BytePairEncoder.load(Path(DEFAULT_BPE))
    print("few-shot retrieval on the original tokenizer (bpe.json)")
    return {"qwen": (hf_tok, qmodel)}, qmodel, tok


def resolve_max_reply(args):
    # qwen's tokenizer packs fewer chars per token than the retrieval BPE, so
    # its default budget is twice the one the chat config was written with.
    if args.max_reply == 40:
        args.max_reply = 80
    return args


def load_corpus(args):
    # Normalize the export + live chat log into canonical turns for retrieval.
    user_name, _ = speaker_names()
    corpus_text = ""
    export = Path(args.data)
    if export.exists():
        raw = export.read_text(encoding="utf-8")
        user = args.user or detect_user(raw)
        aliases = {user.lower(): user_name} if user.lower() != user_name else None
        corpus_text = parse_wa(raw, aliases=aliases)
        if args.shots > 0:
            print(f"few-shot retrieval on {export} (user: {user})")
    live = Path(LOG_PATH)
    if live.exists() and live.stat().st_size:
        live_text = live.read_text(encoding="utf-8").strip("\n")
        if live_text:
            head = corpus_text.rstrip("\n") + "\n\n" if corpus_text else ""
            corpus_text = head + live_text + "\n"
            if args.shots > 0:
                print(f"live memory: {live}")
    return corpus_text


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


def generate_proactive(args, state, model, tokenizer, openers, device,
                       source="daemon"):
    """One unprompted message, biased toward how she opens chats at this hour."""
    hf_tok, hf_model = state["qwen"]
    _, her_name = speaker_names()
    off_hour = args.proactive_hour if args.proactive_hour is not None \
        else time.localtime().tm_hour
    pool = openers_near_hour(openers, off_hour, args.proactive_window)
    # opener demos live inside the system text, same rule as the REPL
    demos = _opener_demos(pool, args, tokenizer)
    head = _read_description(args)
    lines = ([head] if head else []) + [f"{her_name}: {d}" for d in demos]
    # the empty user turn is a synthetic "she speaks first" cue; a reply that
    # comes back empty falls back to a real opener
    messages = qwen_chat.build_messages("\n\n".join(lines), "", [])
    user_name, her_name = speaker_names()
    candidates, thought = _sample(args, hf_tok, hf_model, messages,
                                  user_name, her_name)
    chosen, why, scored = _choose(candidates, " ".join(demos), tokenizer, {})
    reply = chosen or (pool[0]["text"] if pool else "")
    _log_turn(args, source, "", scored, reply, why,
              {"persona": head, "demos": "\n".join(demos), "history": []},
              thought)
    return reply
