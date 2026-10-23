import argparse
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data
from tokenizer import BytePairEncoder

CORPUS = ("hola como estas gracias buenos dias que tal nos vemos luego "
          "mañana hablamos un rato mas tarde ") * 120
SPEAKER_USER = "alice"
SPEAKER_HER = "babe"


def toy_encoder(vocab_size=420):
    """A small BPE trained on the synthetic corpus (has real merges)."""
    encoder = BytePairEncoder()
    encoder.train(CORPUS, vocab_size)
    return encoder


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


def check_reply_shape():
    import engines.qwen.chat as qwen_chat
    user, her = SPEAKER_USER, SPEAKER_HER
    raw = (f": hola que tal\n{her}: como estas\n{her}: nos vemos\n"
           f"{user}: y vos\n")
    cleaned = qwen_chat.clean_reply(raw, user, her)
    checks = {
        "non-empty": bool(cleaned),
        "no-user-bleed": f"{user}:" not in cleaned,
        "no-inner-label": f"{her}:" not in cleaned,
        "multi-line-kept": "\n" in cleaned,
        "leading-colon": not cleaned.startswith(":"),
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


def check_chatlog():
    """Appends stay corpus-parseable: one line per turn, multi-line flattened,
    and a quiet gap opens a new conversation with a blank line."""
    import chatlog
    data.configure_speakers(SPEAKER_USER, SPEAKER_HER)
    gap = data.CONVERSATION_GAP_MINUTES * 60
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "live.txt"
        chatlog.append_turn("hola", "hey", path=path)
        chatlog.append_turn(None, "solo", path=path)
        chatlog.append_turn("otra\nmulti", "vale\nbien", path=path,
                            now=time.time() + 2 * gap)
        text = path.read_text(encoding="utf-8")
    expected = (f"{SPEAKER_USER}: hola\n{SPEAKER_HER}: hey\n"
                f"{SPEAKER_HER}: solo\n"
                f"\n{SPEAKER_USER}: otra multi\n{SPEAKER_HER}: vale bien\n")
    return text == expected, ("log ok" if text == expected
                              else f"unexpected log: {text!r}")


def check_turnlog():
    """One audited turn round-trips: candidates, chosen reply, why, thought."""
    import turnlog
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "telemetry.sqlite3"
        turnlog.record("repl", "hola",
                       [("hey", 0.5), ("otra cosa", 0.1)], "hey",
                       "overlap 0.500 vs her retrieved turns",
                       {"persona": "p", "demos": "d", "history": ["u: hola"]},
                       "porque si", path=path, now=1000.0)
        rows = turnlog.rows(path=path, limit=5)
        other = turnlog.rows(path=path, limit=5, source="daemon")
    checks = {
        "row stored": len(rows) == 1,
        "candidates kept": [c["text"] for c in rows[0]["candidates"]]
                           == ["hey", "otra cosa"],
        "scores kept": [c["score"] for c in rows[0]["candidates"]]
                       == [0.5, 0.1],
        "chosen/thought kept": rows[0]["chosen"] == "hey"
                               and rows[0]["thought"] == "porque si",
        "source filter": other == [],
    }
    failed = [name for name, ok in checks.items() if not ok]
    return not failed, "audit ok" if not failed else f"failed: {failed}"


CHECKS = [
    ("roundtrip", check_roundtrip),
    ("canonical", check_canonical),
    ("boundary", check_boundary),
    ("reply-shape", check_reply_shape),
    ("demo-retrieval", check_demo_retrieval),
    ("qwen-template", check_qwen_template),
    ("qwen-mask", check_qwen_mask),
    ("qwen-strip", check_qwen_strip_think),
    ("qa-rows", check_qa_rows),
    ("session-roundtrip", check_session_roundtrip),
    ("session-corrupt", check_session_corrupt),
    ("queue-roundtrip", check_queue_roundtrip),
    ("chatlog", check_chatlog),
    ("turnlog", check_turnlog),
]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="run the repo self-checks")
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
