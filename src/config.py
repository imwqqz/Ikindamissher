import argparse
import sys
from pathlib import Path

from data import speaker_names

def _find_project_root(start):
    # The pre-restructure entry scripts resolved data/ and configs/ next to
    # themselves, i.e. at the repo root. Installed console scripts run copies of
    # these modules from site-packages, so anchor on the working tree instead:
    # walk up from the cwd to the project's pyproject.toml, falling back to this
    # module's own checkout only when the cwd is outside the project.
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
DEFAULT_CKPT = PROJECT_ROOT / "data" / "her_model.pt"
DEFAULT_DESC = PROJECT_ROOT / "data" / "input" / "description.txt"
DEFAULT_QA = PROJECT_ROOT / "data" / "input" / "persona_qa.txt"
DEFAULT_BPE = PROJECT_ROOT / "data" / "input" / "bpe.json"

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "her.config.json"

SIZE_PRESETS = {
    "nano": dict(d_model=128, n_layers=4, n_heads=4, d_ff=512, max_len=128),
    "mini": dict(d_model=256, n_layers=6, n_heads=8, d_ff=1024, max_len=256),
    "small": dict(d_model=384, n_layers=8, n_heads=8, d_ff=1536, max_len=512),
}

# Dimensions a --size preset can supply, and a --d-model style flag can override.
ARCH_KEYS = ("d_model", "n_layers", "n_heads", "d_ff")


def _resolve_stored_block(args, state, rank):
    # A checkpoint fixes the architecture, so its context wins over a flag.
    stored = state["cfg"].get("max_len")
    if not stored:
        return
    if args.block is not None and args.block != stored:
        if rank == 0:
            print(f"resume: --block {args.block} ignored; checkpoint "
                  f"trained with context {stored}", flush=True)
    args.block = stored


def resolve_block(args, state, rank):
    # --block owns max_len; the size preset must not override it or RoPE is sized
    # for a context the data does not use.
    if state is not None:
        _resolve_stored_block(args, state, rank)
        return
    if args.block is None:
        preset = SIZE_PRESETS.get(args.size) or {}
        args.block = preset.get("max_len", 128)


def _apply_size_preset(args, cfg, size):
    if not size:
        return
    for k, v in size.items():
        # max_len is owned by --block; see resolve_block.
        if k == "max_len":
            continue
        src = getattr(args, "arch_src", {}).get(k)
        if src == "cli":
            continue                        # --d-model on the flag line
        if getattr(args, "size_cli", False):
            cfg[k] = v                      # --size on the flag line beats
        elif src is None:                   # a config file, which only
            cfg[k] = v                      # outranks the preset as a default


def build_cfg(args, tokenizer):
    size = SIZE_PRESETS.get(args.size)
    # Persist the turn labels: the checkpoint is the only record of how its turns
    # were labelled, and chatbot.py reads them from here.
    user_name, her_name = speaker_names()
    cfg = {
        "user_speaker": user_name,
        "her_speaker": her_name,
        "vocab_size": tokenizer.vocab_size,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "d_ff": args.d_ff,
        "max_len": args.block,
        "dropout": args.dropout,
        "target_only_loss": bool(getattr(args, "target_only_loss", False)),
        "objective": args.objective,
        "lora": args.lora or args.qlora,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "quant": "nf4" if args.qlora else "none",
        "gc": getattr(args, "gc", False),
        "tokenizer_arch": "bpe",
    }
    if size:
        _apply_size_preset(args, cfg, size)
    return cfg
# keys that describe the invocation rather than the run, so they are not
# written into a generated config file
_META_KEYS = ("help", "config", "dump_config")


def load_config(path):
    # A config only supplies defaults; argparse stays the schema, so a flag on the
    # command line always beats the file.
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
    # The JSON author gets argparse's validation, so "epochs": "20" behaves
    # like --epochs 20.
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
        # Refuse unknown keys: a misspelled option that is silently dropped leaves you
        # training with settings you did not choose.
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
    # Shared by her.py and chatbot.py: an explicit --config wins, else the
    # script's own config file is auto-discovered, else built-in defaults.
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
    """Config supplies defaults, explicit flags win. Exits on --dump-config."""
    # Snapshot before apply_config; see parse_args.
    builtin_defaults = {a.dest: a.default for a in ap._actions}
    apply_config(ap, _bootstrap_config(ap, argv, default_path))
    args = ap.parse_args(argv)
    if args.dump_config:
        written = dump_config(ap, args.dump_config, builtin_defaults)
        print(f"wrote {len(written)} options to {args.dump_config}")
        raise SystemExit(0)
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
    # Relative paths, so a committed example works on any machine instead of
    # pinning the absolute layout of whoever generated it.
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


def _arch_sources(args, argv):
    """flag > config > --size preset, and each key records which of the three
    it came from."""
    tokens = set(sys.argv[1:] if argv is None else argv)
    return {
        k: ("cli" if f"--{k.replace('_', '-')}" in tokens else "config")
        for k in ARCH_KEYS
        if f"--{k.replace('_', '-')}" in tokens or k in args._from_config
    }, "--size" in tokens


def _maybe_dump_config(ap, args, builtin_defaults):
    if args.dump_config:
        written = dump_config(ap, args.dump_config, builtin_defaults)
        print(f"wrote {len(written)} options to {args.dump_config}")
        raise SystemExit(0)


def parse_args(argv=None, config=None):
    ap = argparse.ArgumentParser(
        description="her: memorize the chat into a scalable decoder-only transformer",
    )
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--description", default=str(DEFAULT_DESC))
    ap.add_argument("--persona-prefix", action="store_true",
                    help="prepend the fact sheet to every training window, "
                         "masked from the loss, so each example is conditioned "
                         "on it instead of only the windows that reach the "
                         "corpus end")
    ap.add_argument("--persona-qa", default=str(DEFAULT_QA),
                    help="question/answer turns for the facts; declarative "
                         "facts alone teach the model to state them, not "
                         "answer a question about them")
    ap.add_argument("--persona-repeat", type=int, default=0,
                    help="how many times to splice the Q&A block through the "
                         "corpus; 0 disables it")
    ap.add_argument("--user", default=None,
                    help="your speaker name in the export (default: auto-detect)")
    ap.add_argument("--user-speaker", default=None,
                    help="canonical name for you in training turns "
                         "(default: same as the export name)")
    ap.add_argument("--her-speaker", default=None,
                    help="canonical name for the persona in training turns; "
                         "required, and kept out of the source on purpose, so a "
                         "private name never lands in git history")
    ap.add_argument("--tokenizer-file", default=str(DEFAULT_BPE))
    ap.add_argument("--fresh", action="store_true",
                    help="retrain from scratch, ignore checkpoint and BPE cache")
    ap.add_argument("--config", default=None, metavar="FILE",
                    help="JSON file of defaults; explicit flags still win "
                         f"(default: {DEFAULT_CONFIG.name} if it exists)")
    ap.add_argument("--dump-config", default=None, metavar="FILE",
                    help="write every option and its default to FILE, then exit")
    ap.add_argument("--size", default="nano",
                    choices=["none", "nano", "mini", "small"],
                    help="architecture preset; an explicit --d-model/"
                         "--n-layers/--n-heads/--d-ff, from the command line "
                         "or a config file, overrides the preset's value")

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--val-fraction", type=float, default=0.1,
                    help="tail fraction of turns held out for validation; 0 "
                         "disables it (no generalization signal, no early stop)")
    ap.add_argument("--val-every", type=int, default=250,
                    help="run validation every N optimizer steps")
    ap.add_argument("--patience", type=int, default=10,
                    help="stop after this many validations without improvement; "
                         "0 disables early stopping, which is what a memorization "
                         "run wants since its val loss rises as it memorizes")
    ap.add_argument("--keep-last", action="store_true",
                    help="save the final weights instead of the best-val ones; "
                         "for memorizing, the last weights are the ones you want")
    ap.add_argument("--spike-factor", type=float, default=4.0,
                    help="abort if the train loss stays above this multiple of "
                         "its running best for --spike-patience steps; 0 "
                         "disables the spike test (a non-finite loss always "
                         "aborts)")
    ap.add_argument("--spike-patience", type=int, default=3,
                    help="consecutive spiking steps tolerated before aborting")
    ap.add_argument("--block", type=int, default=None,
                    help="training sequence length; defaults to the size "
                         "preset's context (nano 128, mini 256, small 512)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dropout", type=float, default=0.1,
                    help="dropout probability (default 0.1; 0 for memorization)")
    ap.add_argument("--schedule", default="invsqrt",
                    choices=["invsqrt", "constant", "cosine"],
                    help="LR schedule after warmup: inverse-square-root decay "
                         "(Attention Is All You Need, section 5.3 "
                         "https://arxiv.org/abs/1706.03762), flat constant "
                         "(T5, section 3.4 https://arxiv.org/abs/2005.14165), "
                         "or cosine anneal to ~0 (SGDR, section 2.1 "
                         "https://arxiv.org/abs/1608.03983)")
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="label smoothing eps_ls (Attention Is All You Need, "
                    "section 5.4 https://arxiv.org/abs/1706.03762)")
    ap.add_argument("--target-only-loss", action="store_true",
                    help="score only her turns. The corpus interleaves both "
                         "speakers and the model is only ever asked to "
                         "continue after her label, so the other half of the "
                         "stream is otherwise learned and then suppressed at "
                         "sampling time. Same idea as the completion-only loss "
                         "TRL's SFTTrainer applies to assistant turns "
                         "(train_on_prompt=false).")
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="micro-batches per optimizer step (T5, section 3.4 "
                         "https://arxiv.org/abs/2005.14165)")
    ap.add_argument("--precision", default="fp32", choices=["fp32", "bf16", "fp16"])

    ap.add_argument("--objective", default="lm", choices=["lm", "span"])
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--qlora", action="store_true",
                    help="NF4-quantize frozen base + train LoRA adapters "
                         "(QLoRA, section 3.4 https://arxiv.org/abs/2305.18290; "
                         "LoRA, section 4.1 https://arxiv.org/abs/2106.09685)")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.0)

    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=512)
    ap.add_argument("--vocab-size", type=int, default=4096)

    ap.add_argument("--gc", action="store_true", help="gradient checkpointing")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model")
    ap.add_argument("--ddp", action="store_true", help="data-parallel (torchrun)")
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--log-best", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    if config is None:
        config = _bootstrap_config(ap, argv, DEFAULT_CONFIG)
    # Snapshot the built-in defaults before apply_config overwrites them:
    # --dump-config reports the defaults, and reading action.default afterwards
    # would emit the local config, speaker names included.
    builtin_defaults = {a.dest: a.default for a in ap._actions}
    apply_config(ap, config)
    args = ap.parse_args(argv)
    # Remember which keys came from a config file, to tell 'asked for' from default.
    args._from_config = set(config or ())
    args.arch_src, args.size_cli = _arch_sources(args, argv)
    _maybe_dump_config(ap, args, builtin_defaults)
    return args
