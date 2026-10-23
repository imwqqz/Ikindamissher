import re
from collections import Counter
from typing import Optional

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

# 2. Convert a WhatsApp export into `speaker: message` turns.
WA_LINE = re.compile(
    r"^\d{1,2}/\d{1,2}/\d{2,4},?.*?- ([^:]+): (.*)$"
)
WA_NAME_BOOTSTRAP = re.compile(r"^[^:\n]+: ")

# A silence this long between messages starts a new conversation in the corpus.
CONVERSATION_GAP_MINUTES = 60

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


def parse_wa(text: str, speakers=None, aliases=None,
             gap_minutes: int = CONVERSATION_GAP_MINUTES):
    # Normalize the exporter's speaker names to canonical turns. A silence of
    # `gap_minutes` or more becomes a blank line, i.e. a conversation boundary.
    if speakers is None:
        speakers = speaker_names()
    speakers = tuple(s.lower() for s in speakers)
    aliases = {k.lower(): v.lower() for k, v in (aliases or {}).items()}
    lines = []
    previous_minute = None
    for line in text.splitlines():
        turn = _wa_turn(line, aliases, speakers)
        if not turn:
            continue
        minute = _line_minute(line)
        if (lines and minute is not None and previous_minute is not None
                and (minute - previous_minute) % (24 * 60) >= gap_minutes):
            lines.append("")
        lines.append(turn)
        if minute is not None:
            previous_minute = minute
    return "\n".join(lines) + "\n"
def detect_user(text: str, label: str = None):
    # Pick the most frequent non-her speaker so any WhatsApp contact works.
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
    # Persona description as `name: line` turns
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
