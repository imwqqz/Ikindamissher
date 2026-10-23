"""Runnable self-checks for the her.py training stack.

Synthetic data only: nothing here reads or prints anything from data/, so the
checks are safe to run on any machine. Exit code is 0 only when every check
passes, so this doubles as a pre-merge gate.

Usage:
    uv run python tools/checks.py
    uv run python tools/checks.py --only canonical
"""

import argparse
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import data
from engines.her.model import build_model
from tokenizer import BytePairEncoder
from engines.her.training import Trainer, lm_loss
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
    # band, not equality: residual logit spread sits ~+2.2 nats above ln(V); catches the std=1 misinit (~128 vs ~10.5, REPORT.md N8)
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
              f"{SPEAKER_HER}: bien\n{SPEAKER_USER}: ok\n")
    mask = data.target_char_mask(corpus, SPEAKER_HER)
    prefix = f"{SPEAKER_HER}: "
    lines = corpus.split("\n")
    expected = 0
    previous_her = False
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            expected += len(line) - (0 if previous_her else len(prefix))
            expected += 1 if index < len(lines) - 1 else 0
        previous_her = line.startswith(prefix)
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


def check_turn_generation():
    # end-to-end sentinel proof: overfit a two-message turn, generate; the reply keeps the intra-turn newline without bleeding into the user's turn
    encoder = toy_encoder()
    if not hasattr(encoder, "turn_end_id"):
        return True, "skipped (turn-end sentinel not implemented yet)"
    data.configure_speakers(SPEAKER_USER, SPEAKER_HER)
    corpus = (f"{SPEAKER_USER}: hola\n{SPEAKER_HER}: hola que tal\n"
              f"{SPEAKER_HER}: como estas\n{SPEAKER_USER}: bien y vos\n"
              f"{SPEAKER_HER}: todo bien\n")
    import her
    ids, labels = her._masked_labels(encoder, corpus, SPEAKER_HER)
    torch.manual_seed(0)
    cfg = tiny_cfg(vocab_size=encoder.vocab_size, d_model=64, n_layers=2)
    cfg["max_len"] = 128
    model = build_model(cfg)
    inputs = torch.tensor([ids])
    targets = torch.tensor([labels])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(600):
        optimizer.zero_grad()
        loss = lm_loss(model(inputs), targets, 0.0)
        loss.backward()
        optimizer.step()
    model.eval()
    prompt = encoder.encode(f"{SPEAKER_USER}: hola\n{SPEAKER_HER}:")
    gen = model.generate(torch.tensor([prompt]), max_new=40, temperature=1e-8,
                         top_k=1, stop_ids=encoder.stop_ids())
    reply = encoder.decode(gen)
    checks = {
        "multi-message": "\n" in reply,
        "no-user-bleed": f"{SPEAKER_USER}:" not in reply,
        "stopped-early": len(gen) < 40,
        "prompt-no-bare-space": prompt[-1] not in set(encoder.encode(" ")),
    }
    failed = [name for name, ok in checks.items() if not ok]
    return not failed, "ok" if not failed else f"failed: {failed}"


def check_conversation_end():
    encoder = toy_encoder()
    if not hasattr(encoder, "conversation_end_id"):
        return True, "skipped (conversation sentinel not implemented yet)"
    data.configure_speakers(SPEAKER_USER, SPEAKER_HER)
    corpus = (f"{SPEAKER_USER}: hola\n{SPEAKER_HER}: que tal\n\n"
              f"{SPEAKER_USER}: buenas\n{SPEAKER_HER}: hola\n")
    scenes = data.conversation_ends(corpus)
    turn_ends = data.her_turn_ends(corpus, SPEAKER_HER)
    ids, mask = encoder.encode_masked(corpus,
                                      data.target_char_mask(corpus, SPEAKER_HER),
                                      turn_ends, scenes)
    sentinel = encoder.conversation_end_id
    positions = [i for i, token in enumerate(ids) if token == sentinel]
    stripped = [t for t in ids if t not in (sentinel, encoder.turn_end_id)]
    checks = {
        "one-per-boundary": len(scenes) == 1 and len(positions) == 1,
        "scored": bool(positions) and all(mask[i] for i in positions),
        "concat": stripped == encoder.encode(corpus),
        "distinct-ids": sentinel != encoder.turn_end_id,
    }
    failed = [name for name, ok in checks.items() if not ok]
    return not failed, "ok" if not failed else f"failed: {failed}"


def check_reply_shape():
    import engines.dispatch as chatbot
    user, her = SPEAKER_USER, SPEAKER_HER
    raw = (f": hola que tal\n{her}: como estas\n{her}: nos vemos\n"
           f"{user}: y vos\n")
    cleaned = chatbot._clean_reply(raw, user, her)
    checks = {
        "non-empty": bool(cleaned),
        "no-user-bleed": f"{user}:" not in cleaned,
        "no-inner-label": f"{her}:" not in cleaned,
        "multi-line-kept": "\n" in cleaned,
        "leading-colon": not cleaned.startswith(":"),
    }
    failed = [name for name, ok in checks.items() if not ok]
    return not failed, "ok" if not failed else f"failed: {failed}"


def check_grad_norm_telemetry():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = build_model(cfg)
    ids = torch.randperm(cfg["vocab_size"]).tolist() * 2
    with tempfile.TemporaryDirectory() as tmp:
        # checkpoint in a fresh temp dir so the gate runs on any machine
        ckpt = Path(tmp) / "checks_gnorm.pt"
        trainer = Trainer(trainer_args(), cfg, model, ids, torch.device("cpu"),
                          BytePairEncoder(), ckpt)
        trainer.fit()
        norms = list(trainer.grad_norms)
    ok = bool(norms) and all(math.isfinite(g) for g in norms)
    return ok, f"{len(norms)} norms recorded"


def check_val_events():
    # log.csv twin: the first validation always saves (best_val starts at
    # inf), an unchanged val logs a stale row, and exhausted patience takes
    # the early-stop branch
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = build_model(cfg)
    ids = torch.randperm(cfg["vocab_size"]).tolist() * 2
    val_ids = torch.randperm(cfg["vocab_size"]).tolist() * 3
    with tempfile.TemporaryDirectory() as tmp:
        log_dir = Path(tmp) / "runs" / "her_model"
        trainer = Trainer(trainer_args(), cfg, model, ids, torch.device("cpu"),
                          BytePairEncoder(), Path(tmp) / "her_model.pt",
                          val_ids=val_ids, log_dir=log_dir)
        # a real run always validates after training steps, so loss is finite
        inputs, targets = next(iter(trainer.sampler))
        trainer._record_loss(trainer._train_step(inputs, targets))
        trainer._optimizer_step()
        trainer._validate_on_rank0(10, 1)
        trainer._validate_on_rank0(20, 1)
        trainer.patience = 1
        third = trainer._validate_on_rank0(30, 1)
        events = [line.rsplit(",", 1)[1]
                  for line in (log_dir / "log.csv")
                  .read_text(encoding="utf-8").splitlines()[1:]]
    checks = {
        "early-stop": bool(third) and trainer.stopped_early
                      and trainer.best_step == 10,
        "rows": events == ["save", "val_saved", "val", "val"],
    }
    failed = [k for k, ok in checks.items() if not ok]
    return not failed, "ok" if not failed else f"failed: {failed}"


def check_cli_flags():
    # --d-model=512 must count as asked-for in both spellings, else a --size preset silently overwrites it
    from config import cli_flags, parse_args
    args = parse_args(["--size", "small", "--d-model=512"], config={})
    seen = "--d-model" in cli_flags(["--d-model=512"])
    checks = {
        "equals-form": seen,
        "space-form": "--d-model" in cli_flags(["--d-model", "512"]),
        "arch-source": args.arch_src.get("d_model") == "cli",
    }
    failed = [name for name, ok in checks.items() if not ok]
    return not failed, "ok" if not failed else f"failed: {failed}"


def check_demo_retrieval():
    # one conversation must still yield two demos, the second from another, on separate lines
    import engines.dispatch as chatbot
    encoder = toy_encoder()
    data.configure_speakers(SPEAKER_USER, SPEAKER_HER)
    corpus = (f"{SPEAKER_USER}: gato lindo\n{SPEAKER_HER}: miau\n"
              f"{SPEAKER_USER}: gato feo\n{SPEAKER_HER}: miau dos\n\n"
              f"{SPEAKER_USER}: gato otro\n{SPEAKER_HER}: miau tres\n")
    index = chatbot.build_demo_index(corpus, encoder)
    text, blocks = chatbot.few_shot_pairs(index, 2, 20, encoder, "gato")
    checks = {
        "backfilled": blocks == 2,
        "seam": "\n\n" in text,
    }
    failed = [name for name, ok in checks.items() if not ok]
    return not failed, f"{blocks} demos" if not failed else f"failed: {failed}"


def check_qwen_template():
    """Qwen3 template, thinking disabled: assistant rows carry exactly one
    EMPTY thinking block, texts survive encode->decode, and a generation
    prompt pre-closes the block (documented force-non-thinking mechanism)."""
    try:
        import engines.qwen.qwen_lm as qwen_lm
    except ImportError as exc:
        return False, f"qwen stack missing: {exc}"
    tok = qwen_lm.get_tokenizer()
    open_m, close_m = qwen_lm.think_markers(tok)
    hallo = ({"role": "system", "content": "Sos una amiga cercana."},
             {"role": "user", "content": "hola\ncomo andas"},
             {"role": "assistant", "content": "aca ando\ny vos"})
    text = qwen_lm.apply_template(hallo, tok)
    checks = {
        "one open/close per assistant row":
            text.count(open_m) == 1 and text.count(close_m) == 1,
        "empty thinking block":
            text.count(open_m + "\n\n" + close_m) == 1,
        "assistant text survives":
            "aca ando" in tok.decode(tok(text, add_special_tokens=False)
                                     .input_ids, skip_special_tokens=False),
    }
    gen = qwen_lm.apply_template(hallo[:2], tok, add_generation_prompt=True)
    tail = gen.split("<|im_start|>")[-1]
    checks["generation prompt pre-closes the block"] = \
        0 <= tail.find(open_m) < tail.find(close_m)
    failed = [k for k, ok in checks.items() if not ok]
    return not failed, "template ok" if not failed else f"failed: {failed}"


def check_qwen_mask():
    """encode_row scores only assistant turns as real token ids; headers and
    user/system text get -100. Decode-as-ids must equal the assistant turn
    (this test exists because a mask/ids mix-up trained silently to loss 0)."""
    try:
        import engines.qwen.qwen_lm as qwen_lm
    except ImportError as exc:
        return False, f"qwen stack missing: {exc}"
    tok = qwen_lm.get_tokenizer()
    row = ({"role": "user", "content": "cuando vienes"},
           {"role": "assistant", "content": "pasado mañana\ncon calma"},
           {"role": "user", "content": "digo yo"})
    ids, labels = qwen_lm.encode_row(row, tok)
    scored = [l for l in labels if l != -100]
    from_ids = tok.decode([i for i, l in zip(ids, labels) if l != -100],
                          skip_special_tokens=False)
    checks = {
        "scored ids are real tokens":
            len(scored) > 4 and all(isinstance(l, int) and 0 <= l < len(tok)
                                    for l in scored),
        "decode(ids) carries the reply": "pasado mañana" in from_ids,
        "decode(labels) carries the reply":
            "pasado mañana" in tok.decode(scored, skip_special_tokens=False),
    }
    failed = [k for k, ok in checks.items() if not ok]
    return not failed, "mask ok" if not failed else f"failed: {failed}"


def check_qwen_strip_think():
    """strip_think: closed blocks removed with the visible text kept; an
    unclosed block (truncation) shows nothing after the open marker."""
    try:
        import engines.qwen.qwen_lm as qwen_lm
    except ImportError as exc:
        return False, f"qwen stack missing: {exc}"
    open_m, close_m = qwen_lm.think_markers()
    cases = ((f"presa{open_m}razon{close_m}hola", "presahola"),
             (f"{open_m}razon{close_m}", ""),
             (f"{open_m}sin cerrar", ""))
    for raw, want in cases:
        if qwen_lm.strip_think(raw) != want:
            return False, f"strip_think({raw!r}) != {want!r}"
    return True, "strip ok"
def check_qa_rows():
    """_qa_rows folds q:/a: lines into user->her rows and drops answer-less
    tails (added with the --qa '' disable crash fix, export_sft)."""
    import export_sft
    data.configure_speakers(SPEAKER_USER, SPEAKER_HER)
    with tempfile.TemporaryDirectory() as tmp:
        qa = Path(tmp) / "qa.txt"
        qa.write_text("# comment\n"
                      "a: stray first answer\n"
                      "q: color favorito\na: amarillo\n"
                      "q: dos en fila\na: una\na: dos\n"
                      "q: final sin respuesta\n", encoding="utf-8")
        rows = export_sft._qa_rows(qa, SPEAKER_USER, SPEAKER_HER)
    expected = [
        [{"role": "user", "content": "color favorito"},
         {"role": "assistant", "content": "amarillo"}],
        [{"role": "user", "content": "dos en fila"},
         {"role": "assistant", "content": "una"},
         {"role": "assistant", "content": "dos"}],
    ]
    checks = {
        "stray answer / unanswered tail dropped": len(rows) == 2,
        "q/a pairs kept": rows == expected,
    }
    failed = [k for k, ok in checks.items() if not ok]
    return not failed, "rows ok" if not failed else f"failed: {failed}"


def check_session_roundtrip():
    import server.service as service
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "s.json"
        service.save_session({"history": [f"{SPEAKER_USER}: hola",
                                          f"{SPEAKER_HER}: hola"]}, path)
        back = service.load_session(path)
        ok = back.get("history") == [f"{SPEAKER_USER}: hola", f"{SPEAKER_HER}: hola"]
        return ok, "session round-trip ok"


def check_session_corrupt():
    import server.service as service
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "s.json"
        path.write_text("{not valid json", encoding="utf-8")
        back = service.load_session(path)
        sidecar = path.with_suffix(".corrupt.json")
        ok = back == {"history": []} and sidecar.exists()
        return ok, "corrupt session quarantined ok"


def check_queue_roundtrip():
    import server.service as service
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "q.sqlite3"
        service._init_queue(path)
        i1 = service._queue_push("hola", 1000.0, path)
        i2 = service._queue_push("chau", 2000.0, path)
        rows = service._queue_since(0, path)
        new = service._queue_since(i1, path)
        checks = {
            "both rows in order": [r["id"] for r in rows] == [i1, i2],
            "since=only newer row": [r["text"] for r in new] == ["chau"],
        }
        failed = [k for k, ok in checks.items() if not ok]
        return not failed, "queue ok" if not failed else f"failed: {failed}"


CHECKS = [
    ("roundtrip", check_roundtrip),
    ("canonical", check_canonical),
    ("boundary", check_boundary),
    ("loss-init", check_loss_at_init),
    ("overfit-batch", check_overfit_batch),
    ("mask", check_mask_sanity),
    ("turn-end", check_turn_end_invariants),
    ("conversation-end", check_conversation_end),
    ("turn-generation", check_turn_generation),
    ("reply-shape", check_reply_shape),
    ("demo-retrieval", check_demo_retrieval),
    ("cli-flags", check_cli_flags),
    ("gnorm", check_grad_norm_telemetry),
    ("val-events", check_val_events),
    ("qwen-template", check_qwen_template),
    ("qwen-mask", check_qwen_mask),
    ("qwen-strip", check_qwen_strip_think),
    ("qa-rows", check_qa_rows),
    ("session-roundtrip", check_session_roundtrip),
    ("session-corrupt", check_session_corrupt),
    ("queue-roundtrip", check_queue_roundtrip),
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

