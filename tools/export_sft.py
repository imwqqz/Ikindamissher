import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import (CHAT_CONFIG, DEFAULT_DATA, DEFAULT_DESC, DEFAULT_QA,
                    PROJECT_ROOT, load_config)
from data import (WA_LINE, WA_NOISE, configure_speakers, detect_user,
                  persona_lines, persona_qa_turns, speaker_names)

# match 'name: message' only by accident, via the colon inside the timestamp
SYSTEM = re.compile(
    r"^(\d{1,2}/\d{1,2}/\d{2,4},[^-\n]*- )?"
    r"(messages and calls are end-to-end encrypted|"
    r"your security code with .*? changed\.?)\s*$",
    re.I,
)


def _new_turn(line, speakers, aliases, stats):
    """(matched, turn) for one line. Filtered lines match but yield no turn."""
    match = WA_LINE.match(line)
    if not match:
        return False, None
    name, msg = match.group(1).strip(), match.group(2).strip()
    name = aliases.get(name.lower(), name.lower())
    if name not in speakers:
        stats["unknown_speaker"] += 1
        return True, None
    if WA_NOISE.match(msg):
        stats["media_or_deleted"] += 1
        return True, None
    if not msg:
        stats["empty"] += 1
        return True, None
    return True, [name, msg]


def _continuation(line, turns, stats):
    if not line.strip():
        return
    if turns:
        turns[-1][1] += "\n" + line.rstrip()
        stats["continuation"] += 1
    else:
        stats["system_notice"] += 1


def parse_turns(text, speakers, aliases):
    """Return (turns, stats) where turns is [(speaker, message)].

    A line with a timestamp starts a turn; without one it continues the turn
    above it. A line that can continue nothing is a system notice.
    """
    turns = []
    stats = Counter()
    for line in text.splitlines():
        if SYSTEM.match(line.strip()):
            stats["system_notice"] += 1
            continue
        matched, turn = _new_turn(line, speakers, aliases, stats)
        if matched:
            if turn is not None:
                turns.append(turn)
            continue
        _continuation(line, turns, stats)
    return turns, stats


def _fold_turn(current, convs, role, msg, orphans):
    if role == "user" or not current:
        if role == "assistant":
            return [], orphans + 1    # her reply with no question before it
        if current:
            convs.append(current)
            orphans += 1              # her side already closed; this is unpaired
        current = []
    if current and current[-1]["role"] == role:
        current[-1]["content"] += "\n" + msg
    else:
        current.append({"role": role, "content": msg})
    return current, orphans


def group_conversations(turns, user, her_name):
    """Fold a flat turn list into conversations of user->assistant exchanges.

    Same-speaker runs merge, since a chat template emits one block per role.
    """
    convs = []
    current = []
    orphans = 0
    for name, msg in turns:
        role = "user" if name == user else "assistant"
        current, orphans = _fold_turn(current, convs, role, msg, orphans)
    if current:
        if len(current) < 2:
            orphans += 1
        else:
            convs.append(current)
    return convs, orphans


def to_sharegpt(conversations):
    """LLaMA-Factory's sharegpt loader expects from/value, not role/content."""
    name = {"user": "human", "assistant": "gpt"}
    return [
        {"conversations": [{"from": name[m["role"]], "value": m["content"]} for m in c]}
        for c in conversations
    ]


def percentile(values, quantile):
    values = sorted(values)
    if not values:
        return 0
    return values[min(len(values) - 1, int(len(values) * quantile))]


def _load_config(args):
    # Canonical labels come from the gitignored chatbot config.
    if not CHAT_CONFIG.exists():
        return {}
    return load_config(CHAT_CONFIG)


def _resolve_speakers(args, cfg):
    user = args.user_speaker or cfg.get("user_speaker")
    her_name = args.her_speaker or cfg.get("her_speaker")
    if not user or not her_name:
        raise SystemExit(
            "no speaker names configured; pass --user-speaker and --her-speaker"
        )
    configure_speakers(user, her_name)
    return user.lower(), her_name.lower()


def _drop_long_conversations(convs, max_chars):
    """Drop any message longer than max_chars; emptied conversations fall away."""
    if not max_chars:
        return convs, Counter()
    too_long = Counter()
    kept = []
    for c in convs:
        keep = []
        for m in c:
            if len(m["content"]) > max_chars:
                too_long[m["role"]] += 1
            else:
                keep.append(m)
        if len(keep) >= 2:
            kept.append(keep)
    return kept, too_long


def _persona_block(args, her_name):
    if not (args.persona and args.description.exists()):
        return None
    persona = "\n".join(persona_lines(
        args.description.read_text(encoding="utf-8"), alias=her_name))
    return re.sub(rf"^{her_name}:\s*", "", persona, flags=re.M).strip()


def _qa_rows(qa_path, user, her_name):
    """Answer-shaped persona facts as user->her SFT rows.

    Reuses data.py's q:/a: parser; a trailing user turn with no her reply
    carries no completion and is dropped.
    """
    if not (qa_path and Path(qa_path).exists()):
        return []
    lines = persona_qa_turns(Path(qa_path).read_text(encoding="utf-8"),
                             alias=her_name)
    rows, current = [], []
    for line in lines:
        name, _, msg = line.partition(": ")
        role = "user" if name.lower() == user else "assistant"
        if role == "user":
            if len(current) >= 2:
                rows.append([{"role": r, "content": c}
                             for r, c in current])
            current = [("user", msg)]
        else:
            current.append(("assistant", msg))
    if len(current) >= 2:
        rows.append([{"role": r, "content": c} for r, c in current])
    return rows


def _build_rows(convs, fmt, persona):
    if fmt == "sharegpt":
        rows = to_sharegpt(convs)
        key, lead = "conversations", {"from": "system", "value": persona}
    else:
        rows = [{"messages": c} for c in convs]
        key, lead = "messages", {"role": "system", "content": persona}
    if persona:
        for row in rows:
            row[key] = [lead, *row[key]]
    return rows


def _write_rows(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _her_chars(convs):
    return sum(len(m["content"]) for c in convs
               for m in c if m["role"] == "assistant")


def _real_tokens(convs):
    """Token counts via the qwen tokenizer the engine actually trains with."""
    from transformers import AutoTokenizer
    src = PROJECT_ROOT / "data" / "her_qwen_adapter"
    tok = AutoTokenizer.from_pretrained(
        str(src) if src.exists() else "Qwen/Qwen3-1.7B")
    total = trainable = 0
    for c in convs:
        for m in c:
            n = len(tok.encode(m["content"]))
            total += n
            if m["role"] == "assistant":
                trainable += n
    return total, trainable


def _tally(convs):
    # Two her-shares: quoting either alone is misleading.
    train_chars = _her_chars(convs)
    kept_chars = sum(len(m["content"]) for c in convs for m in c)
    turns_per = [len(c) for c in convs]
    msg_chars = [len(m["content"]) for c in convs for m in c]
    return train_chars, kept_chars, turns_per, msg_chars


def _report(args, rows, out, fmt, stats, turns, orphans, too_long,
            train_chars, kept_chars, turns_per, msg_chars, parsed_chars,
            parsed_total, persona, her_name, qa_convs, tokens):
    print(f"wrote {len(rows)} conversations to {out}  ({fmt} format)")
    print(f"  persona Q&A rows    {qa_convs}")
    print(f"  turns parsed        {len(turns):,}")
    print(f"  continuation kept   {stats['continuation']} lines (not flattened away)")
    print(f"  system notice       {stats['system_notice']}")
    print(f"  media/deleted       {stats['media_or_deleted']:,}")
    print(f"  unknown speaker     {stats['unknown_speaker']}")
    print(f"  unpaired turns      {orphans:,} (no completion, cannot train)")
    if too_long:
        print(f"  over {args.max_chars} chars  {dict(too_long)} message(s) dropped")
    print(f"  turns/conversation  p50={percentile(turns_per, .5)} p90={percentile(turns_per, .9)} "
          f"max={max(turns_per, default=0)}")
    print(f"  message chars       p50={percentile(msg_chars, .5)} p90={percentile(msg_chars, .9)} "
          f"max={max(msg_chars, default=0)}")
    if msg_chars:
        print(f"  mean message        {statistics.mean(msg_chars):.0f} chars")
    if parsed_total:
        print(f"  her share, all text {parsed_chars[her_name] / parsed_total:.1%}")
    if kept_chars:
        print(f"  her share, trainable{train_chars / kept_chars:.1%} <- assistant loss "
              f"sees this")
    if tokens is not None:
        tot, trainable = tokens
        print(f"  tokens, qwen toknzr {tot:,} shown / {trainable:,} assistant "
              f"(real counts, includes template wrap when training)")
    else:
        print(f"  approx tokens       {kept_chars // 4:,} (guess; tokenizer "
              f"unavailable)")
    if persona:
        print(f"  system message      {len(persona):,} chars of persona on every row")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="export a WhatsApp chat to SFT JSONL")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--format",
        choices=["messages", "sharegpt"],
        default="messages",
        help="messages: role/content, read by TRL and Unsloth. "
        "sharegpt: from/value, read by LLaMA-Factory's sharegpt loader.",
    )
    ap.add_argument("--user-speaker", default=None)
    ap.add_argument("--her-speaker", default=None)
    ap.add_argument(
        "--persona",
        action="store_true",
        help="prepend description.txt as a system message on every example",
    )
    ap.add_argument("--description", type=Path, default=DEFAULT_DESC)
    ap.add_argument(
        "--qa", default=str(DEFAULT_QA),
        help="persona Q&A file (q:/a:) exported as extra user->her rows; "
             "pass an empty string to disable"
    )
    ap.add_argument(
        "--max-chars",
        type=int,
        default=2000,
        help="drop any single message longer than this (0 disables). A pasted "
        "essay is not conversation and teaches the model to write essays.",
    )
    args = ap.parse_args(argv)

    cfg = _load_config(args)
    user, her_name = _resolve_speakers(args, cfg)

    raw = args.data.read_text(encoding="utf-8")
    aliases = {}
    detected = detect_user(raw, her_name)
    if detected != user:
        aliases[detected] = user

    turns, stats = parse_turns(raw, {user, her_name}, aliases)
    parsed_chars = {u: 0 for u in (user, her_name)}
    for name, msg in turns:
        parsed_chars[name] += len(msg)
    parsed_total = sum(parsed_chars.values())

    convs, orphans = group_conversations(turns, user, her_name)
    convs, too_long = _drop_long_conversations(convs, args.max_chars)
    # Path("") resolves to ".", which exists: an unset --qa must stay unset.
    qa_convs = _qa_rows(Path(args.qa) if args.qa else None,
                        user, her_name)
    convs = convs + qa_convs
    persona = _persona_block(args, her_name)
    rows = _build_rows(convs, args.format, persona)
    _write_rows(rows, args.out)

    try:
        tokens = _real_tokens(convs)
    except Exception:
        tokens = None    # no adapter, no network: fall back to the chars//4 guess

    train_chars, kept_chars, turns_per, msg_chars = _tally(convs)
    _report(args, rows, args.out, args.format, stats, turns, orphans, too_long,
            train_chars, kept_chars, turns_per, msg_chars, parsed_chars,
            parsed_total, persona, her_name, len(qa_convs), tokens)


if __name__ == "__main__":
    main()
