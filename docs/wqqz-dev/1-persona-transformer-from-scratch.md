| title | I trained a from-scratch AI on a year of chat logs — she learned a voice, not a mind |
| description | No pretrained weights, no APIs, one laptop GPU. A byte-level tokenizer, a 15.7M-parameter transformer, and a private chat export — what broke, what worked, and why small models fail at "understanding." |
| date | 2026-10-08 |
| tags | machine learning, pytorch, from scratch |

> [!CAUTION] Privacy first
> Everything below is aggregate numbers and synthetic examples. No message
> content, no names from the real corpus, and no private data are committed
> anywhere — the model artifacts live in a gitignored `data/` directory. If you
> train on someone's real words, this is the minimum bar.

A year ago I watched a video of someone training a model to write like their ex
girlfriend.[^video] Same premise here: take a private chat export, and teach a
model to *simulate how one specific person writes*. Not a generic assistant.
Not a companion bot. One person's voice, syntax, timing, and humor.

The difference is what I refused to use: no pretrained weights, no hosted API,
no fine-tuning of something huge. ~~Just fine-tune a GPT and ship it~~ was the
obvious route; I wanted to know what the parts actually do. A byte-level BPE
tokenizer and a decoder-only transformer, both written from scratch, trained on
a laptop GPU. That decision cost weeks and bought something specific — every bug
in this post was visible, because no abstraction was hiding it.

Here is the whole story: the boring part that decides everything, three bugs
that trained happily and lied, one experiment that failed usefully, and the two
tokens that finally made the chat make sense.

---

## 1. What the reference video got right

The video's build is worth stealing from before anything else:

- **Clean the data before touching a model.** A small script to tidy the export
  into clean turns. Mine became a proper pipeline (below).
- **Verify the loss at initialization.** It checks that training starts at
  $-\ln(1/V)$: with its 131-token vocabulary, about $4.87$. That one number
  catches a whole class of silent misconfiguration.
- **Conversational state matters more than raw generation.** It adds chat
  history and indentation of the AI's messages so replies sit in context.

Those three instincts became, on my side, a check in the test harness
(loss-at-init), a corpus pipeline with conversation boundaries, and a retrieval
layer. The divergence — building the stack myself — is what made the rest of
this post possible.

---

## 2. You can't skip the boring part

A chat export is timestamps, contact names, `(media omitted)` noise, and
speakers labelled however the phone felt that day. The pipeline does four
things, and every one of them matters more than any model tweak:

1. Parse each line into `(time, speaker, message)`.
2. Alias speaker names to **two canonical labels**, so the model never sees two
   spellings of the same person.
3. Drop noise and unknown speakers.
4. Insert a **blank line after any ≥60-minute silence**.

That last one is the difference between a model that reads a conversation and
one that reads *a pile of unrelated days stitched together*. Without it, replies
happily jump topics mid-turn — the classic "why is my small chatbot changing
subject" failure.

Then a **byte-level BPE tokenizer** is trained from scratch on that corpus. The
vocabulary is not just bytes and merges; it reserves space for structure:

```python title="src/tokenizer.py"
# vocabulary layout after training (this corpus: ~3,700 merges)
#   0..255        bytes
#   256           word-boundary sentinel  (never merged through)
#   257..         learned merges, in rank order
#   …             span sentinels (reserved for span-corruption training)
#   X             turn_end          scored; the generation stop token
#   X+1           conversation_end  scored; marks a scene break
#   vocab = base + sentinels + 2 = 4,098
```

Two design choices in there are the load-bearing ones:

- **Encoding applies merges by rank, not by frequency.** The original code
  picked the pair with the lowest *count* in the current piece — which is not
  BPE. I caught it with a crafted case: `"ababc"` encoded to `[257, 97, 258]`
  instead of the canonical `[257, 257, 99]`. A one-function fix, but it changes
  every token id, so it ships with a retrain.
- **Training can never merge through the boundary id.** Otherwise you spend
  vocabulary slots on merges the encoder can never produce.

---

## 3. The trick that makes turns possible

Plain text has no stop switch. Two messages from the same person look exactly
like one message followed by a reply: a newline. So the model cannot learn
"hand the mic back," and generation runs on until a hard character limit cuts it
mid-sentence.

The fix is to **teach a token to mean it**. `turn_end` is scored as the last
token of every turn, the same way sentence-ending punctuation already is. At
generation time, hitting it stops sampling:

```python title="src/training.py"
# in the corpus encoder: turn_end is injected (and scored) at each turn end
ids, flags = tok.encode_masked(text, target_char_mask(text, her_name),
                               her_turn_ends(text, her_name),
                               conversation_ends(text))
```

Later, the same machinery got a second sentinel — `conversation_end` — injected
at every blank-line boundary. It is the model's signal that a scene is over,
not a pause.

> [!WARNING] Breaking by design
> Both sentinels changed the vocabulary (`4,097 → 4,098`) and the corpus
> format. Token ids changing means **old checkpoints and the new code disagree
> about what an id means** — so every such change is labeled `BREAKING`,
> scripted, backed up, and followed by a retrain. Boring on purpose.

---

## 4. Three bugs that trained fine and lied

This is the part of small-model work nobody warns you about: a misconfigured
network rarely throws. It trains, the loss falls, and it works *a little worse*.
Three examples from this project:

### The trailing space

Chat prompts end with the persona label — `"name: "`. But a space at the end of
a piece **encodes as its own token**, and in training a space always merges into
the following word (`" hola"`). The model receives a token that never occurs in
its training stream. A small model's response: flood spaces, or drift into
gibberish.

The fix is one character: end prompts at the **colon**.

```python title="chatbot.py"
# before: encode(full + f"\n{her_name}: ")   <- bare-space token, out of distribution
ids = tokenizer.encode(full + f"\n{her_name}:")
```

### The collation NaN

Batching padded sequences to the widest row. A one-token row with no masked
spans computed a label width of `max_inp - 1 = 0` — all targets vanished, and
the span loss returned **NaN**. Deriving the width from the labels themselves
(and widening the input to `len(labels) + 1`) removed the entire class.

### The encode order

Covered above: `"ababc" → [257, 97, 258]`. Frequency is not rank.

> [!IMPORTANT] The harness is the real deliverable
> Every fix above is guarded by `checks.py` — thirteen self-checks on **synthetic**
> data (never the private corpus), gating every merge on an exit code:
> round-trip, canonical encode, no boundary merges, loss-at-init band, overfit
> one batch, mask sanity, both sentinel invariants, turn generation, reply
> shape, demo retrieval, CLI flag forms, gradient-norm telemetry.
> ```
> uv run python checks.py   # 13/13 required to merge
> ```
> The trailing-space bug was found by check C9, not by reading a diff. That is
> the whole argument for building the harness before building features.

---

## 5. The training run

Two configurations share one architecture. The first is the one that works:
the **whole corpus**, dropout off, no validation, keep the final weights. It
memorizes — deliberately. For a persona model, "download the transcript into
weights" is where the slang, rhythm and message length come from.

![Training loss over 5 epochs](/images/persona-loss.png)

The theoretical starting loss is a good sanity check before any of that:

$$\mathcal{L}_0 \approx \ln V = \ln 4098 \approx 8.32$$

We measure a band rather than an exact value — the residual stack spreads the
logits a little — but a *broken* initialization is unmistakable: it explodes to
around $128$ at this vocabulary size, versus $\approx 10.5$ when correct.

Gradient health is its own log line. The rule of thumb (from the tuning
playbook[^playbook]) is to set the clip threshold just above the typical
gradient norm, so clipping is a safety net rather than the normal case:

$$g' = \lambda \cdot \frac{g}{\lVert g \rVert}\quad\text{when}\quad \lVert g \rVert > \lambda$$

| Clip threshold | p90 norm | Clip rate | Learning rate |
| --- | :--- | ---: | ---: |
| 1.0 | 14.06 → 3.08 | 50–78% | 2.0e-4 |
| 1.0 | 12.70 → 1.86 | 19–57% | 1.0e-4 |
| 3.5 | early p90 14.06 | 50% at step 100 | 2.0e-4 |

![Gradient norm and clip rate](/images/persona-gradnorm.png)

Halving the learning rate did not fix the clip rate — the threshold was simply
tight for this model. It is now a flag (`--grad-clip`, `0` disables), decided by
telemetry rather than guesswork.

> [!TIP] Make the dial readable before you need it
> Log the **pre-clip** norm, its running p90, and the clip percentage on every
> logged step. "Is something wrong with training?" becomes a one-glance answer,
> and you stop tuning blind.

---

## 6. The experiment that failed usefully

The second configuration is the textbook recipe: hold out 10% of the
conversation, add dropout, stop when validation stops improving, keep the best
checkpoint. This is how you teach *generalization* — usually.

![Validation rises while training falls](/images/persona-val.png)

Validation bottoms out 500 steps in and then climbs forever while training loss
collapses. The reading is blunt: at ~77,000 tokens, a 15.7M-parameter model
memorizes too fast for any style signal to consolidate. The best checkpoint is
under-trained, and its chat is visibly worse than the memorizing model's.

I kept the run and its graph. **A negative result is a result.** It is now a
line in the runbook — "skip this recipe until the corpus grows" — and it is the
reason the shipped system leans on retrieval instead.

---

## 7. What actually fixed the chat

The memorizing model alone produced the "collage": unrelated conversations
stitched into one turn, because training windows slide across conversation
boundaries. Two changes, in order of effect:

1. **Conversation-chained few-shot demos.** At inference, find a real exchange
   similar to your prompt and show it as context. Naively, that means two
   orphan pairs from different days — and the model continues *that* mismatch.
   Chaining the matched pair with the turns that followed it, all from **one
   conversation**, makes the context read as a single exchange.
2. **Scored sentinels** (turn-end and conversation-end), so turns and scenes
   have explicit boundaries.

Demos are ranked by IDF-weighted token overlap — rare words like a topic word
count more than greetings that match everything — with a length tie-break and
deduplication:

| Stage | What went wrong |
| --- | --- |
| memorization only | unrelated conversations stitched into one turn |
| + turn-end sentinel | clean multi-line turns, correct stopping |
| + lexical demos | in-voice, but topics hopped |
| + IDF-weighted, chained demos | replies stay inside one exchange |

The honest summary: at this scale, **retrieval is not garnish — it is the only
part of the system doing "learn about the topic I just raised."**

---

## 8. What it is, and what it isn't

![Memorization curve and the failed validation run](/images/persona-hero.png)

It **is** a from-scratch BPE + decoder-only LM + KV-cache sampler + retrieval
stack, trained to the limit of what a year of chat can support: a convincing
voice, turn mechanics, stop discipline, multi-message timing.

It is **not** a small assistant you can ask anything. There is no intent
alignment, no state across sessions, no abstraction past lexical overlap. Those
are not tuning knobs — they are different training signals:

- a prompt→response objective instead of a flat stream (the exporter already
  emits that pair format),
- session facts plus an optional sidecar adapter, so a chat can teach new things
  without touching the base weights.

The ladder, stated plainly:

- [x] replay her messages (memorization)
- [x] speak in her voice on new prompts
- [x] stay in-voice on novel input via retrieval
- [ ] match *intent*, not just topic
- [ ] evolve per session

> [!CAUTION] One ethical line
> A model that imitates a real person is easy to build and easy to misuse.
> Mine never sees the network, never trains on anyone else's data, and the
> corpus is gitignored. If you build one: keep it local, keep it honest, and
> do not pretend the output is the person.

---

## 9. Five lessons

1. **Build the dial, then read it.** Pre-clip gradient norm, p90, clip rate.
2. **Trust an overfit test over a loss curve.** Overfit one batch to near-zero
   loss; it is the cheapest bug detector in deep learning.
3. **Spend compute on retrieval, not on shrinking the model.** It beats early
   stopping at small scale.
4. **Make breaking changes boring.** Label, script, back up, retrain — every
   time. That is the difference between a race condition and a routine.
5. **Test with a synthetic human.** Every check replays toy `alice/babe`
   sentences, so the suite is shareable and private text never leaks.

---

## 10. How to reproduce

```bash frame="terminal" group="run" tab="train"
uv sync
uv run python checks.py        # 13/13 before anything else
# Windows/PowerShell: the runbook is a .ps1 script
./scripts/run_models.ps1       # checks gate -> memory model (+ style experiment)
```

```bash frame="terminal" group="run" tab="chat"
# the memorizing model is the one to talk to
uv run chatbot.py --ckpt data/her_model_memory.pt

# tighter sampling reduces word-level glitches
uv run chatbot.py --ckpt data/her_model_memory.pt --temperature 0.6 --top-k 10
```

The whole stack — tokenizer, model, trainer, sampler, retrieval, checks — is a
few thousand lines of readable Python. That was the point: not to beat a big
model, but to *understand* one, all the way down.

[^video]: "Turning my ex girlfriend into an AI because I miss her…" — the project's motivating video. I cite it for the framing and its verify-loss-at-init habit; the code here is independent. https://www.youtube.com/watch?v=9Hqv09cffzA

[^recipe]: Andrej Karpathy, "A Recipe for Training Neural Networks" — verify the loss at init, overfit one batch, and the "training fails silently" principle behind the check harness. https://karpathy.github.io/2019/04/25/recipe/

[^playbook]: Google Research, "Deep Learning Tuning Playbook" — gradient clipping just above the p90 norm, unclipped gradient-norm logging, and the retrain/evaluation discipline. https://github.com/google-research/tuning_playbook

[^roformer]: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" — the rotary embeddings used in attention. https://arxiv.org/abs/2104.09864
