import sys
from pathlib import Path


def _find_project_root(start):
    for base in (start, *start.parents):
        if (base / "pyproject.toml").exists():
            return base
    return None


PROJECT_ROOT = (
    _find_project_root(Path.cwd())
    or _find_project_root(Path(__file__).resolve().parent)
    or Path(__file__).resolve().parents[1]
)

DEFAULT_DATA = PROJECT_ROOT / "data" / "input" / "wa_out.txt"
DEFAULT_DESC = PROJECT_ROOT / "data" / "input" / "description.txt"
DEFAULT_QA = PROJECT_ROOT / "data" / "input" / "persona_qa.txt"
DEFAULT_BPE = PROJECT_ROOT / "data" / "input" / "bpe.json"

CHAT_CONFIG = PROJECT_ROOT / "configs" / "chatbot.config.json"

LIVE_DIR = PROJECT_ROOT / "data" / "live"


def move_legacy(old, new):
    # file may be held by a running old daemon; retried next start
    import shutil
    old, new = Path(old), Path(new)
    if not old.exists() or new.exists():
        return
    new.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(old), str(new))
    except OSError as exc:
        print(f"[live] could not move {old} -> {new}: {exc}")

_META_KEYS = ("help", "config", "dump_config")


def request_cpu_fallback():
    try:
        answer = input("CUDA is not available. Continue on CPU? [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in ("y", "yes")


def load_config(path):
    import json
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"config not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"config {path} is not valid JSON: {exc}")
    if not isinstance(data, dict):
        raise SystemExit(f"config {path} must hold a JSON object, "
                         f"not {type(data).__name__}")
    return data


def _coerce(action, value):
    if isinstance(getattr(action, "const", None), bool):
        if not isinstance(value, bool):
            raise SystemExit(f"config: {action.dest} must be true or false, "
                             f"not {value!r}")
        return value
    if action.choices and value not in action.choices:
        raise SystemExit(f"config: {action.dest}={value!r} must be one of "
                         f"{', '.join(map(str, action.choices))}")
    if action.type is not None and value is not None:
        try:
            return action.type(value)
        except (TypeError, ValueError):
            raise SystemExit(f"config: {action.dest}={value!r} is not a valid "
                             f"{getattr(action.type, '__name__', action.type)}")
    return value


def apply_config(ap, config):
    if not config:
        return
    known = {a.dest: a for a in ap._actions if a.dest not in _META_KEYS}
    unknown = sorted(set(config) - set(known))
    if unknown:
        import difflib
        hints = []
        for name in unknown:
            close = difflib.get_close_matches(name, list(known), n=2, cutoff=0.6)
            hints.append(f"{name} -> {', '.join(close)}" if close else name)
        raise SystemExit(
            f"config: unknown option(s): {', '.join(unknown)}\n"
            f"  closest match: {'; '.join(hints)}\n"
            f"  valid options: {', '.join(sorted(known))}")
    ap.set_defaults(**{k: _coerce(known[k], v) for k, v in config.items()})


def _config_path_from_argv(argv):
    for i, token in enumerate(argv):
        if token == "--config" and i + 1 < len(argv):
            return Path(argv[i + 1])
        if token.startswith("--config="):
            return Path(token.split("=", 1)[1])
    return None


def _bootstrap_config(ap, argv, default_path):
    raw = list(argv) if argv is not None else sys.argv[1:]
    explicit = _config_path_from_argv(raw)
    path = explicit or (default_path if default_path and default_path.exists() else None)
    if path is None:
        return None
    config = load_config(path)
    if not explicit:
        print(f"config: using defaults from {path}", flush=True)
    return config


def resolve_config_args(ap, argv=None, default_path=None):
    builtin_defaults = {a.dest: a.default for a in ap._actions}
    apply_config(ap, _bootstrap_config(ap, argv, default_path))
    args = ap.parse_args(argv)
    _maybe_dump_config(ap, args, builtin_defaults)
    return args


def _default_for(defaults, action):
    if defaults and action.dest in defaults:
        return defaults[action.dest]
    return action.default


def _to_relative_path(value, root):
    if not isinstance(value, str):
        return value
    try:
        candidate = Path(value).resolve()
        if candidate.is_relative_to(root):
            return candidate.relative_to(root).as_posix()
    except (OSError, ValueError):
        pass
    return value


def dump_config(ap, path, defaults=None):
    import json
    root = PROJECT_ROOT
    body = {}
    for action in ap._actions:
        if action.dest in _META_KEYS:
            continue
        value = _to_relative_path(_default_for(defaults, action), root)
        body[action.dest] = value
    ordered = {k: body[k] for k in sorted(body)}
    Path(path).write_text(json.dumps(ordered, indent=2) + "\n", encoding="utf-8")
    return ordered


def _maybe_dump_config(ap, args, builtin_defaults):
    if args.dump_config:
        written = dump_config(ap, args.dump_config, builtin_defaults)
        print(f"wrote {len(written)} options to {args.dump_config}")
        raise SystemExit(0)
