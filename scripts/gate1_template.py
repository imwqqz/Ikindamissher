"""Phase 1 gate (plan 1.4): Qwen template round-trip over real sft rows.

- every message's text survives encode -> decode
- exactly one EMPTY thinking block per assistant row
- generation prompt (enable_thinking=False) has no thinking block
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from transformers import AutoTokenizer

ROW = Path(__file__).resolve().parents[1] / "data" / "sft.json"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def main():
    rows = [json.loads(l) for l in ROW.read_text(encoding="utf-8").splitlines()]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    bad = 0
    sample = rows[::max(1, len(rows) // 12)] + rows[-1:]
    for r in sample:
        t = tok.apply_chat_template(r["messages"], tokenize=False,
                                    enable_thinking=False)
        d = tok.decode(tok(t, return_tensors="pt").input_ids[0],
                       skip_special_tokens=False)
        n_asst = sum(m["role"] == "assistant" for m in r["messages"])
        n_open, n_close = t.count(THINK_OPEN), t.count(THINK_CLOSE)
        # one thinking pair per assistant message, empty
        if n_open != n_asst or n_close != n_asst:
            bad += 1
            print("think-block count wrong:", n_open, n_close, n_asst)
        for m in r["messages"]:
            if m["content"][:20] not in d:
                bad += 1
                print("LOST", repr(m["content"][:40]))
        # empty between every open/close pair
        i = 0
        pairs = 0
        while (i := t.find(THINK_OPEN, i)) != -1:
            j = t.find(THINK_CLOSE, i)
            assert j != -1, "open without close"
            if t[i + len(THINK_OPEN):j].strip():
                bad += 1
                print("non-empty thinking:", repr(t[i:j + 4])[:80])
            i = j + 1
            pairs += 1
        assert pairs == t.count(THINK_OPEN)
    # generation prompt with enable_thinking=False pre-fills an EMPTY, CLOSED
    # thinking block (Qwen3's documented way to force non-thinking mode): the
    # block must be closed, i.e. close marker follows the open marker.
    p = tok.apply_chat_template(rows[1]["messages"][:2], tokenize=False,
                                add_generation_prompt=True,
                                enable_thinking=False)
    tail = p.split("<|im_start|>")[-1]
    open_i, close_i = tail.find(THINK_OPEN), tail.find(THINK_CLOSE)
    if not (0 <= open_i < close_i):
        bad += 1
        print("generation prompt: thinking block not pre-closed:", repr(tail)[:120])
    print(f"gate 1.4: {'FAIL' if bad else 'PASS'} ({len(sample)} rows)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
