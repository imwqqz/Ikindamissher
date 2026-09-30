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
├── src/
│   ├── tokenizer.py            # byte-level BPE, 100 reserved sentinel ids
│   ├── data.py                 # WhatsApp export -> corpus; speaker labels
│   │                           #   (configure_speakers, parse_wa, persona,
│   │                           #   target_char_mask, build_corpus)
│   ├── model.py                # GPT, RoPE, pre-norm blocks, weight tying
│   │                           #   (CausalSelfAttention with KV cache,
│   │                           #   GPT.generate, sample_probs, build_model)
│   ├── training.py             # losses, span corruption, ValWindows, Trainer
│   ├── checkpoint.py           # load/save tokenizer, resume, reconcile_epochs
│   └── config.py               # argparse + JSON config + size presets
├── her.py                      # training entry point (carved from her.py)
├── chatbot.py                  # talks to the trained model; few-shot retrieval
├── export_sft.py               # export -> SFT jsonl for external trainers
├── configs/
│   ├── her.config.example.json      # every training knob, with defaults
│   ├── chatbot.config.example.json  # every decoding knob, with defaults
│   ├── her.config.json              # gitignored (speaker names)
│   └── chatbot.config.json          # gitignored (speaker names)
├── data/                      # gitignored: your export, tokenizer, weights
│   ├── input/wa_out.txt       # the WhatsApp export
│   ├── input/description.txt  # the persona sheet
│   ├── input/bpe.json         # tokenizer, written on a fresh run
│   └── her_model.pt           # the trained weights
└── pyproject.toml
```

Data flow: the export goes through `parse_wa` (data.py) into a corpus,
`target_char_mask` marks her characters, `encode_masked` turns that into scored
windows, `GPT` (model.py) trains on them into `data/her_model.pt`, and
`chat.py` reads it back.

## Getting started

Copy the example configs, sync (installs `ikindamissher` into `.venv` and the
`her-train` / `her-chat` / `her-export-sft` commands), then train and talk to
her. Both real configs are gitignored because they hold the speaker names,
which are deliberately absent from the source so they can't reach git history.

```bash
copy configs\her.config.example.json configs\her.config.json
copy configs\chatbot.config.example.json configs\chatbot.config.json

uv sync

# train; writes data/her_model.pt
uv run her-train

# chat (or: uv run python -m chatbot)
uv run her-chat

wqqz@wqqz D:\..\Ikindamissher> uv run her-chat
config: using defaults from D:\dev\Git\Ikindamissher\configs\chatbot.config.json
device: cuda
loaded checkpoint: data\her_model.pt (loss 1.0207, trained 2,318 steps)
objective: lm | lora: False | quant: none
few-shot retrieval on data/input/wa_out.txt (user: wqqz)
chatting with her as 'her' - type 'exit' to quit
<wqqz>: holus
her: holus
<wqqz>: tqm
her: y eso te hace mal
<wqqz>: exit
```

The entry code itself lives at the root, so the old command lines keep working
unchanged: `py her.py`, `py chatbot.py`, `py export_sft.py`, and (from the
right cwd) `from her import X`.

`--help` on any entry point lists every knob (`uv run her-train --help`,
`--help` on the `chat` and `export_sft` commands as well). A fresh run
retrains the tokenizer and ignores any existing checkpoint; drop `--fresh` to
resume.
