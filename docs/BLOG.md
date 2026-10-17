# Teaching a 15M-parameter model to talk like one specific person

*From a private chat export. From scratch. On a laptop GPU.*

> Privacy note: this post uses aggregate numbers and synthetic examples only —
> no message content and no speaker names from the real corpus.

## The constraint that made it interesting

No pretrained weights, no hosted APIs, no off-the-shelf dataset. Just a chat
export of one person, an empty `nn.Module`, and a GPU in a laptop. The usual
advice would be "fine-tune something bigger." This project does the opposite:
build the whole stack — tokenizer, model, trainer, sampler, retrieval — and find
out what a small model can actually learn about a single voice.

## The boring part that decides everything

A chat export is timestamps, contact names, `(media omitted)` noise, and
speakers labelled however the phone felt. The pipeline turns that into a corpus:

1. Parse each line into (time, speaker, message).
2. Alias speaker names to two stable labels, so the model never sees two
   spellings of the same person.
3. Drop noise and unknown speakers.
4. Insert a blank line after any ≥60-minute silence. Without this, the model
   trains on one flat stream of stitched-together days — which is exactly why
   small chat models jump between unrelated topics.

Then a byte-level BPE tokenizer is trained from scratch on that corpus: ~3,700
merges, plus reserved ids for span corruption, a turn-end, and a conversation-end
sentinel.

```
0..255      bytes
256         word-boundary (never merges)
257..       learned merges, rank order
…           span sentinels
X           turn_end          ← "I finished this message"
X+1         conversation_end  ← "that conversation ended"
```

Here is the trick that makes it cohere: **plain text has no stop switch**. Two
lines of the same person look identical at the byte level to one line followed by
a reply. So the model cannot learn "hand the mic back." The fix is to teach a
token to mean it — `turn_end` is scored as the last token of every turn, and
generation stops there.

And the best bug of the project lived right next door: a prompt ending in
`"name: "` produces a **bare-space token** that training never emits (spaces
always merge into the following word). The model receives an out-of-distribution
token and either floods spaces or drifts. Ending prompts at the colon fixed it.
One character, invisible in any diff — found by an automated check, not by review.

## The architecture, unglamorous

No `nn.Transformer`, no wrappers.

- Decoder-only, pre-norm, RoPE on interleaved q/k, causal attention, KV cache,
  weight-tied head, GELU FFN; optional NF4/LoRA, DDP, gradient checkpointing.
- AdamW with an empirical recipe (betas 0.9/0.95, decay on linear maps only),
  linear warmup, three schedules.
- Checkpointing that includes optimizer, scaler, **RNG state** and metrics, so a
  resume continues the sampling stream instead of replaying the seed.
- `checks.py`: 11 synthetic-data self-checks gating every merge. It caught the
  prompt-space bug, a collation NaN edge, and a label-width edge before any of
  them could ship as "trains fine but subtly wrong."

## The numbers, including the failure

### Memorization (this works)

```
9.49 ─► 4.79 ─► 3.80 ─► 2.96 ─► 2.30 ─► 1.53 ─► 0.80 ─► 0.33 ─► 0.15 ─► 0.09 ─► 0.01
  100    200     300     400     500     600     700     800     900   1000  done
```

~45 minutes on a 4060, 45–70k tok/s. The endpoint reconstructs the transcript
nearly perfectly. For a persona model, "downloading the transcript" is a feature:
it is where the slang, rhythm and message length come from.

### The style experiment (this fails, usefully)

Hold out 10% of the conversation, add dropout, stop on validation, keep the best
checkpoint — the recipe that normally teaches generalization:

```
val:   5.11 → 4.75 (best, step 500) → 5.06 → 5.52 → 5.92 → 6.21 → 6.50 → 6.70
train: 3.26 →              1.75  →   …                    →  0.14
```

Validation bottoms 500 steps in and then climbs forever. At 77k tokens, a 15M
model memorizes too fast for any style signal to consolidate. The runbook records
it as "the recipe to skip until the corpus grows."

### The chat A/B that matters

| Stage | What went wrong |
|---|---|
| memorization output in a REPL | unrelated conversations stitched into one turn |
| + turn-end sentinel and labeled scoring | clean turns, correct stopping |
| + few-shot demos | in-voice, but topics still hopped |
| + IDF-weighted, dedup'd, conversation-chained demos | replies stay inside one exchange |

The retrieval layer is not garnish at this scale — it is the only part of the
system doing "learn about the topic I just raised."

## What it is, and what it isn't

It **is** a from-scratch BPE + decoder LM + KV-cache sampler + retrieval stack,
trained to the limit of what 77k tokens support: a solid voice, turn mechanics,
stop discipline, multi-message timing.

It is **not** a small assistant you can ask anything. There is no intent
alignment, no state across sessions, no abstraction past lexical overlap. Those
are not tuning knobs; they are different training signals:

- a prompt→response objective instead of a flat stream, and
- session facts plus an optional sidecar adapter, so a chat can teach new things
  without touching the base weights.

The capability ladder: replay → voice → in-voice-on-novel → intent-match →
session evolution. This project lands the first three and documents the path to
the last two.

## Five lessons

1. **Build the dial, then read it.** Logging pre-clip gradient norm, its p90 and
   the clip rate turned "is something wrong?" into a one-glance diagnosis.
2. **Trust an overfit test over a loss curve.** Overfitting a single batch to
   near-zero loss is the cheapest bug detector in deep learning.
3. **Spend compute on retrieval, not on shrinking the model.** It beats early
   stopping at small scale.
4. **Make breaking changes boring.** Vocabulary grew twice; each time the change
   was labeled, scripted, backed up, and re-trained the same way. That is the
   difference between a race condition and a routine.
5. **Test with a synthetic human.** Every check replays toy "alice/babe"
   sentences, so the suite is shareable and no private text ever leaks into a
   log or a commit.
