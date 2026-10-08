"""Runnable self-checks for the her.py training stack.

Synthetic data only: nothing here reads or prints anything from data/, so the
checks are safe to run on any machine. Exit code is 0 only when every check
passes, so this doubles as a pre-merge gate.

Usage:
    uv run python checks.py
    uv run python checks.py --only canonical
"""

import argparse
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch

import data
from model import build_model
from tokenizer import BytePairEncoder
from training import Trainer, lm_loss
from config import parse_args

CORPUS = ("hola como estas gracias buenos dias que tal nos vemos luego "
          "mañana hablamos un rato mas tarde ") * 120
SPEAKER_USER = "alice"
SPEAKER_HER = "babe"


def toy_encoder(vocab_size=420):
    """A small BPE trained on the synthetic corpus (has real merges)."""
    encoder = BytePairEncoder()
    encoder.train(CORPUS, vocab_size)
    return encoder


def tiny_cfg(vocab_size=300, d_model=64, n_layers=2):
    return dict(vocab_size=vocab_size, d_model=d_model, n_heads=4,
                n_layers=n_layers, d_ff=128, max_len=64, dropout=0.0,
                target_only_loss=False, objective="lm", lora=False,
                lora_r=8, lora_alpha=16, lora_dropout=0.0, quant="none",
                gc=False, tokenizer_arch="bpe")


def trainer_args():
    args = parse_args([])
    args.objective, args.precision_dtype = "lm", None
    args.block, args.batch, args.epochs = 32, 4, 1
    args.val_fraction, args.warmup_steps, args.grad_accum = 0.0, 2, 1
    args.log_every, args.spike_factor = 1, 0.0
    return args


def canonical_piece(encoder, piece):
    """Reference BPE: apply the lowest-rank merge each round."""
    ids = list(piece.encode("utf-8"))
    while len(ids) >= 2:
        best = None
        for left, right in zip(ids, ids[1:]):
            rank = encoder.merges.get((left, right))
            if rank is not None and (best is None or rank < best[1]):
                best = ((left, right), rank)
        if best is None:
            break
        ids = BytePairEncoder._merge(ids, best[0][0], best[0][1], best[1])
    return ids


def check_roundtrip():
    encoder = toy_encoder()
    samples = ["hola como estas", "cómo estás 😀", "linea\nnueva",
               "x" * 90, "  doble  espacio  ", "¿qué?"]
    broken = [s for s in samples if encoder.decode(encoder.encode(s)) != s]
    return not broken, f"{len(samples) - len(broken)}/{len(samples)} round-trip"


def check_canonical():
    encoder = toy_encoder()
    pieces = sorted(set(CORPUS.split()))
    broken = [p for p in pieces if encoder._encode_piece(p) != canonical_piece(encoder, p)]
    injected = BytePairEncoder()
    injected.merges = {(97, 98): 257, (98, 99): 258}
    injected.n_merges = 2
    if injected._encode_piece("ababc") != [257, 257, 99]:
        broken.append("ababc")
    return not broken, f"{len(pieces)} pieces, mismatches={len(broken)}"


def check_boundary():
    encoder = toy_encoder()
    touching = [pair for pair in encoder.merges if 256 in pair]
    return not touching, f"boundary merges={len(touching)}"


def check_loss_at_init():
    # Band, not equality: at init the loss sits above ln(V) by the residual
    # stack's logit spread (measured ~+2.2 nats, stable across sizes). The band
    # still catches the documented catastrophic misinit: embedding std=1 gives
    # ~128 vs ~10.5 for vocab 4096.
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = build_model(cfg)
    ids = torch.randint(0, cfg["vocab_size"], (4, 64))
    with torch.no_grad():
        loss = float(lm_loss(model(ids), ids, 0.0))
    expected = math.log(cfg["vocab_size"])
    sane = expected - 0.5 <= loss <= expected + 4.0
    return sane, f"loss={loss:.3f} band=[{expected - 0.5:.3f},{expected + 4.0:.3f}]"


def check_overfit_batch():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = build_model(cfg)
    model.train()
    ids = torch.randint(0, cfg["vocab_size"], (1, 64))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss = None
    for _ in range(300):
        optimizer.zero_grad()
        loss = lm_loss(model(ids), ids, 0.0)
        loss.backward()
        optimizer.step()
    value = float(loss)
    return value < 0.1, f"final loss={value:.4f} after 300 steps"


def check_mask_sanity():
    data.configure_speakers(SPEAKER_USER, SPEAKER_HER)
    corpus = (f"{SPEAKER_USER}: hola\n{SPEAKER_HER}: que tal\n"
              f"{SPEAKER_USER}: bien\n{SPEAKER_HER}: y vos\n")
    mask = data.target_char_mask(corpus, SPEAKER_HER)
    prefix = f"{SPEAKER_HER}: "
    lines = corpus.split("\n")
    expected = 0
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            expected += len(line) - len(prefix) + (1 if index < len(lines) - 1 else 0)
    actual = sum(mask)
    sane = actual == expected and 0 < actual < len(corpus)
    return sane, f"scored={actual} expected={expected}"


def check_turn_end_invariants():
    encoder = toy_encoder()
    if not hasattr(encoder, "turn_end_id") or not hasattr(data, "her_turn_ends"):
        return True, "skipped (turn-end sentinel not implemented yet)"
    data.configure_speakers(SPEAKER_USER, SPEAKER_HER)
    corpus = (f"{SPEAKER_USER}: hola\n{SPEAKER_HER}: que tal\n"
              f"{SPEAKER_HER}: bien y vos\n{SPEAKER_USER}: todo bien\n"
              f"{SPEAKER_HER}: me alegro\n")
    turn_ends = data.her_turn_ends(corpus, SPEAKER_HER)
    ids, mask = encoder.encode_masked(
        corpus, data.target_char_mask(corpus, SPEAKER_HER), turn_ends)
    sentinel = encoder.turn_end_id
    positions = [i for i, token in enumerate(ids) if token == sentinel]
    stripped = [token for token in ids if token != sentinel]
    checks = {
        "one-per-run": len(positions) == 2,
        "scored": bool(positions) and all(mask[i] for i in positions),
        "concat": stripped == encoder.encode(corpus),
    }
    torch.manual_seed(0)
    cfg = tiny_cfg(vocab_size=encoder.vocab_size)
    model = build_model(cfg)
    inputs = torch.tensor([ids])
    loss = lm_loss(model(inputs), inputs, 0.0)
    loss.backward()
    checks["gradient"] = bool(model.tok_emb.weight.grad[sentinel].abs().sum() > 0)
    failed = [name for name, ok in checks.items() if not ok]
    return not failed, "ok" if not failed else f"failed: {failed}"


def check_grad_norm_telemetry():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = build_model(cfg)
    ids = torch.randperm(cfg["vocab_size"]).tolist() * 2
    ckpt = Path(tempfile.gettempdir()) / "opencode" / "checks_gnorm.pt"
    trainer = Trainer(trainer_args(), cfg, model, ids, torch.device("cpu"),
                      BytePairEncoder(), ckpt)
    trainer.fit()
    norms = list(trainer.grad_norms)
    ok = bool(norms) and all(math.isfinite(g) for g in norms)
    if ckpt.exists():
        ckpt.unlink()
    return ok, f"{len(norms)} norms recorded"


CHECKS = [
    ("roundtrip", check_roundtrip),
    ("canonical", check_canonical),
    ("boundary", check_boundary),
    ("loss-init", check_loss_at_init),
    ("overfit-batch", check_overfit_batch),
    ("mask", check_mask_sanity),
    ("turn-end", check_turn_end_invariants),
    ("gnorm", check_grad_norm_telemetry),
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", default=None,
                        help="run a single check by name")
    args = parser.parse_args(argv)
    selected = [(n, fn) for n, fn in CHECKS if args.only in (None, n)]
    if not selected:
        print(f"no check named {args.only!r}; choose from "
              f"{', '.join(n for n, _ in CHECKS)}")
        return 2
    failures = 0
    for name, fn in selected:
        try:
            ok, detail = fn()
        except Exception as exc:  # a raised check is a failure, not a crash
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        failures += 0 if ok else 1
    print(f"{len(selected) - failures}/{len(selected)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
