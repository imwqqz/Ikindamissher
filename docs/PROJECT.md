# her — building a persona model from a private chat export

*From scratch: tokenizer, transformer, trainer, sampler, retrieval. On a laptop GPU.*
*This is the project report and the story in one document.*

> **Privacy note:** this document contains only aggregate metrics and synthetic
> examples. No message content and no speaker names from the real corpus appear
> here, and none are committed anywhere in the repository.

---

## 0. Origin — why this project exists

This project was motivated by the video
**"Turning my ex girlfriend into an AI because I miss her..."**
(https://www.youtube.com/watch?v=9Hqv09cffzA). The premise there is the same one
we inherited: take a private chat export, clean it, and train a model to
**simulate how one specific person writes** — not a generic chatbot.

Three things in that video shaped what we built:

1. **Clean the data before touching a model.** The video starts by saving the
   messages and writing a small program to tidy them. Our `src/data.py` is the
   same idea taken further: speaker aliasing to two canonical labels, noise
   removal, and a blank line after every ≥60-minute silence so the model can
   tell one conversation from the next.
2. **Verify the initial loss against theory.** The video checks that training
   starts at `-ln(1/V)`: with a 131-token vocabulary, about `4.87`. That is the
   Karpathy "verify loss @ init" check, and it is check **C4** in our harness —
   where we measure the whole band (`ln(4098) ≈ 8.32`, plus the residual stack's
   logit spread, giving roughly 5.2–9.7) and separately confirm that a bad
   initialization explodes to ~128.
3. **Conversational state matters more than raw generation.** The video adds
   chat history and indentation of the AI's messages so replies sit in context.
   On our side that instinct became the retrieval system (similar real
   exchanges are shown as examples), the scored **turn-end** sentinel (the model
   learns to stop a turn), and a prompt-hygiene fix for a single trailing space
   that silently derailed generation.

One deliberate divergence: that project trained with a framework, while we chose
to build the tokenizer, transformer, trainer, sampler and retrieval **from
scratch**. The payoff was visibility — every bug in this document was found
because there was no abstraction hiding it, and `checks.py` exists precisely
because "it trains but works a little worse" is the default failure mode of this
kind of project.

---

## 1. Executive summary

A from-scratch GPT-style decoder-only transformer (no pretrained weights) was
trained to speak in the style of one specific person from a ~77,000-token chat
export. The codebase went through four phases — audit, feasibility, replan,
execution — and shipped 27 commits behind a 13-check self-test gate.

| Artifact | Recipe | Train loss | Role |
|---|---|---|---|
| `her_model_memory.pt` | full corpus, dropout 0, `--keep-last` | **0.0100** (6,095 steps) | the chattable model |
| `her_model_style.pt` | 90/10 val split, dropout 0.1, best-val | 2.9849 @ step 500 (best val 4.749) | controlled experiment; non-viable at this corpus size |

Measured headline results:

1. A validation-driven "learn the style" recipe is **non-viable** at 77k tokens
   and 15.7M parameters: val loss bottomed at step 500 and rose monotonically
   afterwards. Generalization is supplied at inference (retrieval), not by early
   stopping.
2. Lexical demo retrieval produced topic drift; **conversation-chained demos**
   plus a scored **turn-end** and **conversation-end** sentinel pair fixed the
   "patchwork conversation" failure mode (vocabulary 4,097 → 4,098; breaking).
3. An inference-only **prompt-construction bug** — a trailing space after the
   persona label encoded a token that never occurs in training — caused derailing
   generations. The self-check harness found it, not code review.

---

## 2. How it works

### 2.1 Data pipeline

```
chat export
  │  parse_wa: regex turns "dd/mm/yy, hh:mm - Name: message"
  │    · speaker aliasing → two canonical labels (user / persona)
  │    · media/deleted/unknown speakers dropped
  │    · ≥60-minute silence → blank line (conversation boundary)
  ▼
contiguous tail split for validation (time-ordered, turn boundary)
  │  persona facts → interleaved turns
  │  persona Q&A → spread n× through the corpus
  ▼
target_char_mask: True on the persona's characters (labels + line endings)
  ▼
byte-level BPE
  │  pretokenize → merges (rank order) → ids
  │  inject scored sentinels: turn_end per persona turn,
  │    conversation_end per blank-line boundary
  ▼
sliding-window batches (prefix-conditioned) → transformer → AdamW
  ▼
checkpoint {model, cfg, tokenizer, optimizer, scaler, rng, ema, best-val}
  ▼
chat: persona + chained demos + history + "user: <prompt>" + persona label
      → KV-cache sampling → stop on the turn-end sentinel
```

### 2.2 Model

- Decoder-only transformer: pre-norm blocks with RMSNorm, causal multi-head
  attention with RoPE (interleaved q/k pairs), GELU FFN, sqrt(d_model)-scaled
  embeddings, weight-tied output head, KV cache, `scaled_dot_product_attention`.
- Proven optional paths: NF4 quantization, LoRA adapters, gradient checkpointing,
  DDP, torch.compile with an eager fallback on Windows.
- Installed during the project: rotary caches cast to the autocast dtype, an fp16
  `GradScaler` with persisted state, and an explicit batch-1 generation guard.

### 2.3 Tokenizer

```
0..255      bytes
256         word-boundary sentinel (never merged)
257..       learned merges, rank order (encode = lowest-rank merge first)
…           span sentinels (100, reserved for span-corruption training)
X           turn_end          scored; generation stop token
X+1         conversation_end  scored; marks scene breaks
            vocab = base + sentinels + 2 = 4,098 for this corpus
```

Encoding applies merges **by rank**, not by frequency (a bug reproduced with a
crafted case `ababc → [257, 257, 99]`), and training can never merge through the
boundary id.

### 2.4 Training loop

| Component | Behavior |
|---|---|
| Loss | next-token, optional label smoothing, completion-only mask on persona turns (continuation prefixes included) |
| Optimizer | AdamW, betas (0.9, 0.95), eps 1e-9, weight decay 1e-2 on linear maps only (norms/embeddings/biases excluded) |
| Schedule | linear warmup → invsqrt / cosine / constant; persisted with scaler and full RNG state (deterministic resume) |
| Stability | pre-clip grad-norm + p90 + clip-rate telemetry; configurable `--grad-clip` (0 disables); windowed loss-spike guard that aborts without saving |
| Validation | contiguous-tail windows, patience-gated early stop, best-val retention (`--no-keep-last` override) |
| Safety | refuses non-finite saves; loud warning when optimizer state cannot be restored |

### 2.5 Generation

KV-cache prefill, then per-token sampling with repeat penalty, top-k and top-p,
stopping on the turn-end sentinel. Prompts end at the persona label **colon**; a
trailing space would be an out-of-distribution token (training spaces always
merge into the following word). Reply cleaning strips learned intra-turn labels
and leading colons while preserving multi-message line breaks.

### 2.6 Self-check harness (`checks.py`, 13/13)

| Check | Guards against |
|---|---|
| round-trip | tokenizer/driver regressions on accents, emoji, newlines |
| canonical encode | encode-order drift |
| boundary merges | BPE merging through id 256 |
| loss @ init band | catastrophic initialization (10.51 vs 128.0 measured) |
| overfit one batch | masks, alignment, optimizer, LR — the master detector |
| mask sanity | scored fraction equals persona character fraction |
| turn-end invariants | sentinel placement, scoring, concat equivalence |
| conversation-end invariants | one sentinel per boundary, scored, concat equivalence |
| turn generation | generated turn stops at the sentinel, no user bleed |
| reply shape | labels/colons stripped, multi-line preserved |
| demo retrieval | backfill + conversation seam in the few-shot block |
| cli flags | --flag=value counts as an explicit request |
| grad-norm telemetry | norms recorded for the clip decision |

Every change was gated on this harness (exit code 0 to merge). Vectorizing
`mask_spans` reduced a 2,000×512-row scan from 1,967.8 ms to 66.6 ms (~30×).

---

## 3. Measured results

### 3.1 Memorization run (windowed train loss)

```
9.49 ─► 4.79 ─► 3.80 ─► 2.96 ─► 2.30 ─► 1.53 ─► 0.80 ─► 0.33 ─► 0.15 ─► 0.09 ─► 0.01
  100    200     300     400     500     600     700     800     900   1000  done
```

5 epochs, 6,025 optimizer steps, ~45 minutes, 45–70k tok/s. The endpoint is a
model that reconstructs the transcript nearly exactly. For a persona model that
is the point: it is where the slang, rhythm and message length live.

### 3.2 Validation run (the negative result)

```
val:   5.11 → 4.75 (best, step 500) → 5.06 → 5.52 → 5.92 → 6.21 → 6.50 → 6.70  (stale 6/8)
train: 3.26 →                  1.75  →   …                       →  0.14
```

Validation bottoms early and then rises while train loss collapses: memorization
outpaces any style signal at this corpus size. Early stop fired around step
2,156 of 5,390; the best-val checkpoint (step 500) was retained. The runbook
records this as "the recipe to skip until the corpus grows."

### 3.3 Gradient-norm telemetry

| Clip threshold | p90 norm | Clip rate | LR |
|---|---|---|---|
| 1.0 | 14.06 → 3.08 | 50–78% | 2.0e-4 |
| 1.0 | 12.70 → 1.86 | 19–57% | 1.0e-4 |
| 3.5 | early p90 14.06 | 50% at step 100 | 2.0e-4 |

Halving the learning rate did not cure the clip rate: threshold 1.0 was simply
tight for this model. The threshold is now a flag (default 1.0; 0 disables).

### 3.4 Chat quality by stage (qualitative)

| Stage | Failure mode observed |
|---|---|
| memorization only | unrelated conversations stitched into one turn ("collage") |
| + turn-end sentinel and labeled scoring | clean multi-line turns, correct stopping |
| + few-shot demos | in-voice but topic drift |
| + IDF-weighted, deduplicated, conversation-chained demos | replies stay inside one exchange; novel prompts extend the retrieved exchange |

The retrieval layer is not garnish at this scale — it is the only part of the
system doing "learn about the topic I just raised."

---

## 4. What it is, and what it isn't

It **is** a from-scratch BPE + decoder LM + KV-cache sampler + retrieval stack,
trained to the limit of what 77k tokens support: a solid voice, turn mechanics,
stop discipline, multi-message timing.

It is **not** a small assistant you can ask anything. There is no intent
alignment, no state across sessions, no abstraction past lexical overlap. Those
are not tuning knobs; they are different training signals:

- a prompt→response objective instead of a flat stream (the exporter already
  produces that pair format), and
- session facts plus an optional sidecar adapter, so a chat can teach new things
  without touching the base weights.

The capability ladder: replay → voice → in-voice-on-novel → intent-match →
session evolution. This project lands the first three and documents the path to
the last two.

---

## 5. Findings ledger

| Phase | Shipped | Dropped (premise not reproduced) | Deferred |
|---|---|---|---|
| Audit | 20 findings across tokenizer/training/model/data/chat | NF4 odd-parity packing (padding guarantees even) | sentinel budget (only above block ~1024) |
| Feasibility | 12 committed | BPE pair-key collision (base grows dynamically) | true batch>1 generation (unused) |
| Replan | 4 batches | — | conversation-scoped windows (needs evidence) |

Breaking changes (each requires a retrain):

1. encode by merge rank, no boundary merges — token ids change
2. scored turn-end sentinel — vocab +1, corpus format
3. conversation-end sentinel with 60-minute gaps — vocab +2 (4,098), corpus format

All retrains use one script that backs up every artifact, gates on the checks,
and supports `-Only memory` and `-GradClip`.

---

## 6. Five lessons

1. **Build the dial, then read it.** Logging pre-clip gradient norm, its p90 and
   the clip rate turned "is something wrong?" into a one-glance diagnosis.
2. **Trust an overfit test over a loss curve.** Overfitting a single batch to
   near-zero loss is the cheapest bug detector in deep learning — and it was the
   same instinct as the video's initial-loss check, generalized.
3. **Spend compute on retrieval, not on shrinking the model.** It beats early
   stopping at small scale.
4. **Make breaking changes boring.** Vocabulary grew twice; each time the change
   was labeled, scripted, backed up, and retrained the same way. That is the
   difference between a race condition and a routine.
5. **Test with a synthetic human.** Every check replays toy "alice/babe"
   sentences, so the suite is shareable and no private text ever leaks into a
   log or a commit.

---

## 7. Reproduction

```powershell
uv sync
.\scripts\run_models.ps1                      # checks gate -> memory + style
uv run python checks.py                       # expect 13/13
uv run chatbot.py --ckpt data\her_model_memory.pt
```

Model artifacts live under `data/` (gitignored). No chat content and no speaker
names are committed anywhere in the repository.

**Motivation credit:** the project's framing and its "verify loss @ init / clean
the data / keep chat history" instincts trace to
*"Turning my ex girlfriend into an AI because I miss her..."*
(https://www.youtube.com/watch?v=9Hqv09cffzA).
