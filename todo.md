# todo

## Fix bugs and known limits

- [ ] `encode_row` in `src/engines/qwen/qwen_lm.py`: a literal `<|im_end|>` inside
  chat content splits the loss span (content is chat text, not tokens). Upgrade
  only if a generated reply ever actually contains that string.
- [ ] `tools/export_sft.py` token report is `chars // 4`, a guess. Use the real
  tokenizer before trusting the number for budget decisions.

## Coming up

- [ ] S2 retrain: `uv run python src\engines\qwen\qwen_lm.py --epochs 6` →
  `uv run python tools\checks.py` (exit 0) + a REPL eyeball.
- [ ] Live-chat memory: append every turn to a corpus file, feed it to demo
  retrieval, and retrain the adapter on demand (`tools/train_from_chats.py`).
