* Attention Is All You Need                          https://arxiv.org/abs/1706.03762
    scaled dot-product / multi-head attention (3.2), residual+LayerNorm blocks (3.1),
    FFN (3.3), scaling embeddings by sqrt(d_model) and weight tying (3.4),
    positional encoding (3.5), Adam + warmup/inverse-sqrt LR (5.3),
    label smoothing + dropout (5.4)
* Exploring the Limits of Transfer Learning with a
  Unified Text-to-Text Transformer (T5)              https://arxiv.org/abs/2005.14165
    text-to-text framing, simplified LayerNorm without bias (2.1),
    pre-norm placement + GELU activation (3.1), span corruption (3.2),
    training strategy / inverse-sqrt decay / dropout (3.4)
* LoRA: Low-Rank Adaptation of Large Language Models https://arxiv.org/abs/2106.09685
    low-rank update W = W0 + (alpha/r) * BA (4.1), applied to q/k/v/o (4.2),
    practical benefits: fewer trainable params (4.3)
* QLoRA: Efficient Finetuning of Quantized LLMs     https://arxiv.org/abs/2305.18290
    4-bit NormalFloat (NF4) block-wise quantization (3.1),
    double quantization of the constants (3.2), paged optimizers (3.3),
    frozen base + trainable adapters (3.4 / Algorithm 1)

## Project structure

```
Ikindamissher/
├── her.py                    # trainer + model; the entry point for training
│   ├── BytePairEncoder       # byte-level BPE, 100 reserved sentinel ids
│   ├── target_char_mask      # which characters of a turn are scored
│   ├── encode_masked         # text -> ids + scored flags; the loss sees only
│   │                         #   her turns, never yours
│   ├── GPT                   # RoPE, pre-norm blocks, weight tying
│   │   └── CausalSelfAttention   # multi-head attention with the KV cache
│   ├── GPT.generate          # prefill + sampled decode, stops at a turn end
│   └── ValWindows            # validation windows with global label offsets
├── chatbot.py                # talks to the trained model; few-shot retrieval
│                             #   primes it with real matching exchanges
├── export_sft.py             # export -> SFT jsonl for external trainers
├── her.config.example.json   # every training knob, with defaults
├── chatbot.config.example.json  # every decoding knob, with defaults
├── data/                     # gitignored: your export, tokenizer, weights
│   ├── input/wa_out.txt      # the WhatsApp export
│   ├── input/description.txt # the persona sheet
│   ├── input/bpe.json        # tokenizer, written on a fresh run
│   └── her_model.pt          # the trained weights
└── pyproject.toml
```

Data flow: the export goes through `parse_wa` into a corpus, `target_char_mask`
marks her characters, `encode_masked` turns that into scored windows, `GPT`
trains on them into `data/her_model.pt`, and `chatbot.py` reads it back.

## Getting started

Copy the example configs, then train and talk to her. Both configs are
gitignored because they hold the speaker names, which are deliberately absent
from the source so they can't reach git history.

```bash
copy her.config.example.json her.config.json
copy chatbot.config.example.json chatbot.config.json

# train; writes data/her_model.pt
py her.py

# chat
py chatbot.py
```

`--help` on either script lists every knob. A fresh run retrains the tokenizer
and ignores any existing checkpoint; drop `--fresh` to resume.
