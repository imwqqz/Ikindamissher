* Attention Is All You Need                          https://arxiv.org/abs/1706.03762
    scaled dot-product / multi-head attention (3.2), residual blocks (3.1),
    FFN (3.3), scaling embeddings by sqrt(d_model) and weight tying (3.4),
    Adam + warmup/inverse-sqrt LR (5.3), label smoothing + dropout (5.4)
* Exploring the Limits of Transfer Learning with a
  Unified Text-to-Text Transformer (T5)              https://arxiv.org/abs/2005.14165
    text-to-text framing, bias-free pre-norm normalization (2.1),
    pre-norm placement + GELU activation (3.1), span corruption (3.2),
    training strategy / inverse-sqrt decay / dropout (3.4)
* LoRA: Low-Rank Adaptation of Large Language Models https://arxiv.org/abs/2106.09685
    low-rank update W = W0 + (alpha/r) * BA (4.1), applied to q/k/v/o (4.2),
    practical benefits: fewer trainable params (4.3)
* QLoRA: Efficient Finetuning of Quantized LLMs     https://arxiv.org/abs/2305.18290
    4-bit NormalFloat (NF4) block-wise quantization (3.1),
    double quantization of the constants (3.2),
    frozen base + trainable adapters (3.4 / Algorithm 1)
* RoFormer: Enhanced Transformer with Rotary Position Embedding
                                                     https://arxiv.org/abs/2104.09864
    rotary position embedding: pairs of dims rotate by base^(-2i/d),
    relative-position invariance (the q/k rotary in model.py)
* The Curious Case of Neural Text Degeneration        https://arxiv.org/abs/1904.09751
    nucleus (top-p) sampling (3.1) and the repetition/presence penalty (5)
* SGDR: Stochastic Gradient Descent with Warm Restarts https://arxiv.org/abs/1608.03983
    cosine annealing (2.1), the --schedule cosine option
* Neural Machine Translation of Rare Words with
  Subword Units (BPE)                                https://arxiv.org/abs/1508.07909
    byte-pair encoding (3), the tokenizer's merge loop
* TRL SFTTrainer (train_on_prompt=false)             https://huggingface.co/docs/trl/sft_trainer
    completion-only loss: score the assistant's turns, not the prompt
    (target_char_mask in data.py)
* A Recipe for Training Neural Networks (Karpathy)   https://karpathy.github.io/2019/04/25/recipe/
    verify loss @ init, overfit one batch, mask sanity, fix the seed, and the
    "training fails silently" principle behind checks.py
* Deep Learning Tuning Playbook (Google Research)    https://github.com/google-research/tuning_playbook
    gradient-clipping threshold just above the p90 norm, unclipped grad-norm
    logging, and the retrain/evaluation discipline behind Trainer's telemetry


```text
$ py chatbot.py
device: cuda
loaded checkpoint: data\her_model.pt (loss 0.0532, trained 12,995 steps)
objective: lm | lora: False | quant: none
few-shot retrieval on data/input/wa_out.txt
chatting with her - type 'exit' to quit
<you>: holus
her: holis
<you>: tqm
her: y eso te hace mal
<you>: exit
```
## Code example

Run everything from the repo root:

```ps1
# working directory: D:\dev\Git\Ikindamissher

uv sync                      # one-time setup (see Installation below)

py her.py                    # train: resumes data/her_model.pt if present
py her.py --fresh            # first run: build tokenizer, ignore stale ckpt
py chatbot.py                # chat REPL

uv run python checks.py      # self-checks on synthetic data; exit 0 required

# train the two side-by-side models (protected paths + pre-run backup):
.\scripts\run_models.ps1
```

## Chat recipe

Two models are trained side by side (`scripts/run_models.ps1`):

- `data/her_model_memory.pt` — full corpus, no validation, `--keep-last`.
  Memorizes the transcript; this is the model to chat with.
- `data/her_model_style.pt` — 10% held-out validation, `dropout 0.1`,
  `--no-keep-last` (the best-val checkpoint is retained).

Measured A/B (2026-10-08, ~77k tokens, 15.7M params): the validation-based
"style" run bottomed at step 500 and rose monotonically afterwards, so its
best checkpoint is under-trained and replies are incoherent. At this corpus
size, generalization comes from few-shot retrieval, not early stopping: chat
with the memory model, keep retrieval on, and tune `--temperature`/`--top-p`.

`--keep-last` is for memorization runs; a run with a validation split should
use `--no-keep-last` (or drop the key from the config), else the final weights
replace the best-val checkpoint. A conversation boundary (a long time gap in
the export) is a scored `conversation_end` token, so windows learn where one
conversation stops and the next starts.

## Project structure

```
Ikindamissher/
├── src/                       # library modules (installed as top-level modules)
│   ├── tokenizer.py           # byte-level BPE + 100 reserved sentinel ids
│   ├── data.py                # WhatsApp export -> corpus; speaker labels,
│   │                          #   persona, target_char_mask, build_corpus
│   ├── model.py               # GPT, RoPE, pre-norm blocks, KV cache, NF4/LoRA
│   ├── training.py            # lm + span losses, windows, validation, Trainer
│   ├── checkpoint.py          # load/save tokenizer, resume, reconcile_epochs
│   └── config.py              # argparse + JSON config + size presets
├── her.py                     # training entry point (--ema, --cpu, --grad-clip)
├── chatbot.py                 # chat REPL + few-shot retrieval
│                              #   (--proactive/--proactive-only, --raw-weights, --cpu)
├── export_sft.py              # WhatsApp export -> SFT jsonl
├── checks.py                  # synthetic self-checks, exit code gate
├── configs/                   # *_config.example.json + your gitignored locals
├── data/                      # gitignored: export, description, BPE, weights
├── .python-version            # 3.14.5
├── pyproject.toml
└── uv.lock
```

Data flow: `parse_wa` (data.py) turns the export into a corpus,
`target_char_mask` marks the persona's characters, windows are scored and fed
to `GPT` (model.py), which trains them into `data/her_model.pt`; `chatbot.py`
reads that checkpoint back and generates replies.

<!-- TODO: choose a license -->
Contributions, acknowledgments, and a formal license are TBD.