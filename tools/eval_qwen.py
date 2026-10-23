"""Phase 5 gate: legacy vs qwen side-by-side on fixed prompts.

Gates (qwen column): no thinking-marker leak in any reply, latency under the
ceiling after warmup, no empty replies. --gate exits non-zero on failure.
Nothing here hardcodes speaker names; labels come from the runtime config.
"""

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch

import engines.dispatch as engine
import client.repl as cli
from config import resolve_config_args
from data import speaker_names

# fixed prompts, sorted roughly from small talk to needy; \n = multi-message
PROMPT_FILE = Path(__file__).resolve().parent / "configs" / "eval_prompts.txt"
PROMPTS = [
    "hola como andas",
    "que hiciste hoy",
    "te extraño bro",
    "queres jugar minecraft?",
    "no pude dormir nada",
    "me voy al gym dale",
    "que comiste hoy",
    "tenua fiebre",
    "nos vemos mañana temprano",
    "eh viste el partido?",
    "estoy re cansado",
    "dale nos juntamos",
    "que onda el nuevo juego",
    "hoy no puedo salgo",
    "te llamé y no respondiste",
    "sos lo mas",
    "buenas noches",
    "me siento solo",
    "te acuerdas del plan del finde",
    "que vais a hacer por navidad",
    "vamos por una fanta o por una coca",
    "perdon me quedé dormido",
]

LATENCY_CEILING_S = 8.0
WARMUP_REPLIES = 1


def run_engine(args, engine_name, prompts):
    """[(prompt, reply, seconds)] for one engine; skips if it cannot load."""
    device = cli._setup_device(args.cpu)
    try:
        state, model, tokenizer = engine.load_assets(args, device)
    except SystemExit as exc:
        print(f"skipping engine {engine_name}: {exc}")
        return None
    cfg = state["cfg"]
    cli._configure_from_checkpoint(args, cfg)
    corpus_text = engine.load_corpus(args, cfg)
    demo_index = engine.build_demo_index(corpus_text, tokenizer) \
        if corpus_text else None
    rows = []
    for i, prompt in enumerate(prompts):
        torch.manual_seed(args.seed + i)
        t0 = time.perf_counter()
        reply = engine.generate_reply(args, state, model, tokenizer, prompt,
                                    [], demo_index, device)
        rows.append((prompt, reply, time.perf_counter() - t0))
    del model
    torch.cuda.empty_cache()
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate", action="store_true",
                    help="exit non-zero unless the qwen gates pass")
    opts = ap.parse_args(argv)

    prompts = PROMPTS
    if PROMPT_FILE.exists():
        prompts = [l for l in PROMPT_FILE.read_text(
            encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")]

    user_name, her_name = speaker_names()
    results = {}
    for engine in ("legacy", "qwen"):
        args = resolve_config_args(cli.build_parser(),
                                   ["--engine", engine], cli.CHAT_CONFIG)
        args = engine.resolve_max_reply(args)
        print(f"\n=== engine: {engine} ({len(prompts)} prompts) ===")
        results[engine] = run_engine(args, engine, prompts)

    qwen_rows = results["qwen"] or []
    legacy_rows = results["legacy"] or []
    print(f"\n{'prompt':<40} | {'legacy':<38} | {'qwen'}")
    print("-" * 122)
    for i, p in enumerate(prompts):
        leg = legacy_rows[i][1] if i < len(legacy_rows) else "-"
        qwn = qwen_rows[i][1] if i < len(qwen_rows) else "-"
        one = lambda s: s.replace("\n", " / ")[:38]  # noqa: E731
        print(f"{p[:40]:<40} | {one(leg):<38} | {one(qwn)}")

    failures = 0
    if qwen_rows:
        open_m, close_m = _think_markers()
        for i, (prompt, reply, secs) in enumerate(qwen_rows):
            if not reply:
                failures += 1
                print(f"[gate] empty reply #{i}: {prompt!r}")
            if open_m in reply or close_m in reply:
                failures += 1
                print(f"[gate] thinking-token leak #{i}: {prompt!r}")
            if i >= WARMUP_REPLIES and secs > LATENCY_CEILING_S:
                failures += 1
                print(f"[gate] latency {secs:.1f}s > {LATENCY_CEILING_S}s "
                      f"#{i}: {prompt!r}")
        print(f"\nqwen: {len(qwen_rows)} replies | gates "
              f"{'FAILED' if failures else 'passed'} "
              f"(ceiling {LATENCY_CEILING_S}s, warmup {WARMUP_REPLIES} skipped)")
    elif opts.gate:
        failures += 1

    if opts.gate:
        sys.exit(1 if failures else 0)


def _think_markers():
    import engines.qwen.qwen_lm as qwen_lm
    return qwen_lm.think_markers()


if __name__ == "__main__":
    main()

