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
- [ ] Console UX: optional `/new`, `/save`, streaming output (its own branch).
- [ ] Avatar: three.js + VRM page served by the daemon, audio from /speak.
- [ ] Optional: cloned voice (coqui XTTS sidecar) if her real timbre matters
  more than a single-environment setup; Piper is realistic but not her.
- [ ] WhatsApp bridge (own branch): unofficial automation risks a ban; the
  official path is the WhatsApp Business API. The daemon's HTTP API is the
  integration point, so no daemon surgery should be needed.
