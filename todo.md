# todo

## Fix bugs and known limits

- [ ] `encode_row` in `src/engines/qwen/qwen_lm.py`: a literal `<|im_end|>` inside
  chat content splits the loss span (content is chat text, not tokens). Upgrade
  only if a generated reply ever actually contains that string.
- [ ] Resume-with-shortfall in `src/engines/her/checkpoint.py`: a partial epoch
  rounds up, so a resume can retrain up to one epoch of extra steps. Split
  `fit()` by steps when the wasted training time matters.
- [ ] `tools/export_sft.py` token report is `chars // 4`, a guess. Use the real
  tokenizer before trusting the number for budget decisions.

## Cleanup

- [ ] Comments cite `docs/REPORT.md` (N1-N17), which was deleted with docs/.
  Rewire or drop the citations in `src/engines/qwen/qwen_lm.py` (most),
  `src/engines/her/model.py`, and `src/engines/her/training.py`.
- [ ] `tools/checks.py` usage header still says `uv run python checks.py`; the
  file moved to `tools/`.
- [ ] Commit the pending work: new `tools/checks.py` checks (val-events,
  session-roundtrip, session-corrupt, queue-roundtrip), the `%.6g` CSV format
  in `src/engines/her/training.py`, and the `virtual` source in `uv.lock`.
- [ ] Decide the fate of the two spikes: `scripts/spike_phase0.py` (NF4 QLoRA
  proving ground, done) and `scripts/gate1_template.py` (template gate, its
  checks now live in `tools/checks.py`). Delete or graduate them[^1] — neither
  is tracked yet.

[^1]: If kept, move them under `scripts/plan/` and note in the header which plan
phase they gate.
