import re
from collections import Counter
from pathlib import Path
from typing import List, Optional

# Speaker labels resolved at run time; a literal would land in git history.
_SPEAKERS = {"user": "", "her": ""}


def speaker_names():
    """Return the canonical (user, persona) labels for turns and prompts."""
    return _SPEAKERS["user"], _SPEAKERS["her"]


def configure_speakers(user: Optional[str] = None, her: Optional[str] = None):
    # Stored lowercased to match what parse_wa emits.
    if user:
        _SPEAKERS["user"] = user.strip().lower()
    if her:
        _SPEAKERS["her"] = her.strip().lower()
    return speaker_names()

# 2. Convert a WhatsApp export into `speaker: message` turns. (T5, section 2.1 https://arxiv.org/abs/2005.14165).
WA_LINE = re.compile(
    r"^\d{1,2}/\d{1,2}/\d{2,4},?.*?- ([^:]+): (.*)$"
)
WA_NAME_BOOTSTRAP = re.compile(r"^[^:\n]+: ")

# WhatsApp's time field for gap detection; the date is not needed, so only the
# clock (and an optional a.m./p.m. marker) is captured.
WA_TIME = re.compile(
    r"^\d{1,2}/\d{1,2}/\d{2,4},?\s*(\d{1,2}):(\d{2})(?::\d{2})?"
    r"\s*(?:([ap])\.?\s?m\.?)?",
    re.I,
)

WA_NOISE = re.compile(
    r"(<media omitted>|media omitted|you deleted this message|this message was deleted)",
    re.I,
)


def _wa_turn(line: str, aliases, speakers):
    match = WA_LINE.match(line)
    if not match:
        return None
    name, msg = match.group(1).strip(), match.group(2).strip()
    name_lower = aliases.get(name.lower(), name.lower())
    if WA_NOISE.match(msg) or not msg or name_lower not in speakers:
        return None
    return f"{name_lower}: {msg}"


def parse_wa(text: str, speakers=None, aliases=None):
    # Normalize the exporter's speaker names to canonical turns. (T5, section 2.1 https://arxiv.org/abs/2005.14165)
    if speakers is None:
        speakers = speaker_names()
    speakers = tuple(s.lower() for s in speakers)
    aliases = {k.lower(): v.lower() for k, v in (aliases or {}).items()}
    lines = [turn for turn in (_wa_turn(line, aliases, speakers)
                               for line in text.splitlines()) if turn]
    return "\n".join(lines) + "\n"
def detect_user(text: str, label: str = None):
    # Pick the most frequent non-her speaker so any WhatsApp contact works. (T5, section 2.1 https://arxiv.org/abs/2005.14165)
    user_name, her_name = speaker_names()
    if label is None:
        label = her_name
    counts = Counter()
    for line in text.splitlines():
        match = WA_LINE.match(line)
        if match:
            name = match.group(1).strip().lower()
            if name and name != label.lower():
                counts[name] += 1
    return counts.most_common(1)[0][0] if counts else user_name


def _line_minute(line: str):
    """Minutes past midnight for a WhatsApp line, or None without a clock."""
    match = WA_TIME.match(line)
    if not match:
        return None
    hour, minute = int(match.group(1)), int(match.group(2))
    marker = (match.group(3) or "").lower()
    if marker == "p" and hour < 12:
        hour += 12
    elif marker == "a" and hour == 12:
        hour = 0
    if hour > 23:
        return None
    return hour * 60 + minute


def first_turns(text: str, label: str = None, aliases=None, gap_minutes: int = 30):
    """Her turns that open a conversation, with the hour they were sent.

    A conversation boundary is a time-of-day gap of at least `gap_minutes`
    since the previous parsed message (wrapping midnight), so no date parsing
    is needed. Only her turns are returned: they are the "writes first"
    examples, e.g. messaging in the evening to ask to play.
    """
    _, her_name = speaker_names()
    label = (her_name if label is None else label).lower()
    aliases = {k.lower(): v.lower() for k, v in (aliases or {}).items()}
    openers = []
    previous_minute = None
    for line in text.splitlines():
        match = WA_LINE.match(line)
        if not match:
            continue
        name = match.group(1).strip().lower()
        name = aliases.get(name, name)
        msg = match.group(2).strip()
        if not msg or WA_NOISE.match(msg):
            continue
        minute = _line_minute(line)
        quiet = (previous_minute is None or minute is None
                 or (minute - previous_minute) % (24 * 60) >= gap_minutes)
        if name == label and quiet:
            openers.append({"hour": None if minute is None else minute // 60,
                            "text": msg})
        if minute is not None:
            previous_minute = minute
    return openers


def openers_near_hour(openers, hour, window: int = 3):
    """Openers sent within `window` hours of `hour` (circular); else all."""
    if hour is None:
        return list(openers)
    near = [o for o in openers
            if o["hour"] is not None
            and min((o["hour"] - hour) % 24, (hour - o["hour"]) % 24) <= window]
    return near or list(openers)
def _persona_line(line: str, rename, canonical: str):
    m = WA_NAME_BOOTSTRAP.match(line)
    if m:
        name = m.group(0)[:-2].strip().lower()
        if name in rename:
            return [f"{rename[name]}: {line[m.end():]}"]
        return [line]
    if len(line) <= 400:
        return [f"{canonical}: {line}"]
    return [f"{canonical}: {chunk.strip()}"
            for chunk in re.split(r"(?<=[.!?])\s+", line) if chunk.strip()]


def persona_lines(text: str, label: str = None, alias: str = None):
    # Persona description as `name: line` turns (T5, section 2.1 https://arxiv.org/abs/2005.14165)
    user_name, her_name = speaker_names()
    label = her_name if label is None else label
    # Lowercased to match parse_wa, else one person gets two leading tokens.
    canonical = label.lower()
    rename = {canonical: canonical, user_name: user_name}
    if alias and alias.lower() not in rename:
        rename[alias.lower()] = canonical
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if line:
            lines.extend(_persona_line(line, rename, canonical))
    return lines
def _qa_turn(stripped: str, user_name: str, her_name: str):
    m = WA_NAME_BOOTSTRAP.match(stripped)
    if m:
        who = m.group(0)[:-2].strip().lower()
        msg = stripped[m.end():]
    else:
        who, msg = "", stripped
    if not msg.strip():
        return []
    if who in ("q", "user", user_name.lower()):
        return [f"{user_name}: {msg}"]
    if who in ("a", "her", her_name.lower()):
        return [f"{her_name}: {msg}"]
    return [f"{her_name}: {stripped}"]


def persona_qa_turns(text: str, label: str = None, alias: str = None):
    """Persona Q&A as alternating `user: q` / `her: a` turns.

    The fact sheet is declarative, so training on it alone teaches the model to
    *state* its attributes, never to *answer a question about* them: given
    "cual es tu color favorito" the corpus contains no turn that continues with
    "amarillo". Each fact therefore also needs an answer-shaped example, or
    the model has no gradient path from a question to the fact.

    Authored as `q:` / `a:` so the file carries no speaker name.
    """
    user_name, her_name = speaker_names()
    her_name = her_name if label is None else label
    lines = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped and not stripped.startswith("#"):
            lines.extend(_qa_turn(stripped, user_name, her_name))
    return lines
def spread_turns(conv: str, block: List[str], repeats: int) -> str:
    """Interleave `block` through `conv` so it is not stranded at one offset.

    A block appended once lands in a single window and is seen by ~0.15% of
    training examples. Splicing it at `repeats` evenly spaced boundaries gives
    it a real share of the loss.
    """
    if not block or repeats <= 0:
        return conv
    lines = conv.splitlines()
    if not lines:
        return conv
    step = max(1, len(lines) // repeats)
    out = []
    for i, line in enumerate(lines):
        if i and i % step == 0:
            out.extend(block)
        out.append(line)
    out.extend(block)
    return "\n".join(out) + "\n"
def target_char_mask(corpus: str, her_name: str) -> List[bool]:
    """True for the characters of her turns, False everywhere else.

    Completion-only loss: the corpus interleaves both speakers, so an unmasked
    loss spends most terms on text generation never asks for.
    TRL SFTTrainer, train_on_prompt=false.
    """
    mask = [False] * len(corpus)
    prefix = f"{her_name}: "
    pos = 0
    lines = corpus.split("\n")
    previous_her = False
    for n, line in enumerate(lines):
        if line.startswith(prefix):
            # The first line of a turn carries the prompt label, so its prefix
            # is unscored. Later lines of the same turn repeat the prefix, and
            # the model must emit it between messages, so those are scored.
            start = pos if previous_her else pos + len(prefix)
            # +1 scores the newline that ends the line: between messages it is
            # the separator the model must generate, and after the last line it
            # is the token that used to be the only stop signal.
            end = pos + len(line) + (1 if n < len(lines) - 1 else 0)
            for i in range(start, min(end, len(mask))):
                mask[i] = True
        previous_her = line.startswith(prefix)
        pos += len(line) + 1  # +1 for the newline that split removed
    return mask


def her_turn_ends(corpus: str, her_name: str) -> List[int]:
    """Char offsets just past each maximal run of her lines (turn ends).

    A turn of hers is several consecutive `her: ` lines, so only the last line
    of a run ends a turn. Offsets point at the start of the next line (or
    len(corpus) at the end), which is where the tokenizer injects the turn-end
    sentinel.
    """
    prefix = f"{her_name}: "
    ends: List[int] = []
    offset = 0
    in_run = False
    for line in corpus.split("\n"):
        if line.startswith(prefix):
            in_run = True
        elif in_run:
            ends.append(offset)
            in_run = False
        offset += len(line) + 1
    if in_run:
        ends.append(len(corpus))
    return ends
def _split_turns(conv: str, val_fraction: float):
    # Contiguous tail split on a turn boundary; windows overlap by block-1, so a
    # random split leaks nearly every validation token into training.
    user_name, _ = speaker_names()
    lines = conv.splitlines()
    if len(lines) < 8 or val_fraction <= 0:
        return conv, ""
    n_val = max(2, int(len(lines) * val_fraction))
    cut = len(lines) - n_val
    # start validation on a user turn so user->her pairs stay intact
    while cut < len(lines) and not lines[cut].startswith(f"{user_name}:"):
        cut += 1
    if cut >= len(lines) - 2:
        return conv, ""
    return "\n".join(lines[:cut]) + "\n", "\n".join(lines[cut:]) + "\n"
def _resolve_speakers(label, user_name, her_name):
    if label is None:
        label = her_name
    if not user_name or not her_name:
        missing = "user" if not user_name else "persona"
        raise SystemExit(
            f"no {missing} name is configured, so turns would have no label.\n"
            'Pass both explicitly, e.g. --user-speaker "<name>" '
            '--her-speaker "<name>", or set user_speaker and her_speaker in '
            "the config file.\nThey are absent from the source on purpose: a "
            "literal would end up in git history.")
    return label, user_name, her_name


def _resolve_aliases(user, label, user_name, her_name):
    aliases = {}
    if user.lower() != user_name:
        aliases[user.lower()] = user_name
    if label.lower() != her_name:
        aliases[label.lower()] = her_name
    return aliases
def _append_persona(train: str, desc_text: str, label: str) -> str:
    desc = "\n".join(persona_lines(desc_text, alias=label))
    if not desc:
        return train
    return train.rstrip("\n") + "\n\n" + desc + "\n"


def _persona_qa_lines(qa_path, label):
    if qa_path is None or not Path(qa_path).exists():
        return []
    return persona_qa_turns(Path(qa_path).read_text(encoding="utf-8"),
                            alias=label)

# (T5, section 2.1 https://arxiv.org/abs/2005.14165).
def build_corpus(data_path: Path, desc_path: Path, label: str = None,
                 user: Optional[str] = None, val_fraction: float = 0.0,
                 qa_path: Optional[Path] = None, persona_repeat: int = 0):
    # Returns the corpus string, or (train, val) when val_fraction > 0.
    user_name, her_name = speaker_names()
    label, user_name, her_name = _resolve_speakers(label, user_name, her_name)
    raw = data_path.read_text(encoding="utf-8")
    if user is None:
        user = detect_user(raw, label)
    aliases = _resolve_aliases(user, label, user_name, her_name)
    corpus = parse_wa(raw, aliases=aliases or None)
    # lowercase, to match what parse_wa emits for every other turn
    corpus = corpus.replace(f"\n{label.lower()}: ", f"\n{her_name}: ")

    train, val = _split_turns(corpus, val_fraction)
    # Persona goes to train only: it is appended at the corpus end, so a tail
    # split would hold all of it out.
    if desc_path.exists():
        train = _append_persona(train, desc_path.read_text(encoding="utf-8"),
                                label)
    # Answer-shaped persona examples, spread through the corpus so each one is
    # seen by many windows instead of only the ones that land on it.
    qa = _persona_qa_lines(qa_path, label)
    if qa:
        train = spread_turns(train, qa, persona_repeat)
    return (train, val) if val else train
