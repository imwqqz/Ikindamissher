# todo

## Fix bugs and known limits

- [ ] `encode_row` in `src/engines/qwen/qwen_lm.py`: a literal `<|im_end|>` inside
  chat content splits the loss span (content is chat text, not tokens). Upgrade
  only if a generated reply ever actually contains that string.
- [ ] Resume-with-shortfall in `src/engines/her/checkpoint.py`: a partial epoch
  rounds up, so a resume can retrain up to one epoch of extra steps. Split
  `fit()` by steps when the wasted training time matters.

## Cleanup

## Coming up

- [ ] S2 retrain: `uv run python src\engines\qwen\qwen_lm.py --epochs 6` →
  gate with `uv run python tools\eval_qwen.py --gate` (exit 0) + a REPL eyeball.
- [ ] After ~2 stable weeks of 24/7 qwen: retire the fire escape — delete
  `src/engines/her/` + `her.py` in one commit (git history keeps it forever).
- [ ] `--engine legacy` was renamed to `--engine her`; remember it when hand-
  typing REPL flags.
