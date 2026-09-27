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
|-- her.py                    trainer + model. Trains the decoder-only
|                             transformer on the chat export and writes the
|                             checkpoint. This is the entry point for training.
|   BytePairEncoder           byte-level BPE, 100 reserved sentinel ids
|   target_char_mask          which characters of a turn are scored
|   encode_masked             text -> (ids, scored flags); the loss sees only
|                             her turns, never yours
|   GPT / CausalSelfAttention RoPE, pre-norm blocks, KV cache
|   GPT.generate              prefill + sampled decode, stops on a turn end
|   ValWindows                validation windows with correct global label
|                             offsets
|-- chatbot.py                talks to the trained model. Few-shot retrieval
|                             finds real exchanges matching your prompt and
|                             primes the model with them.
|-- export_sft.py             converts the export to SFT jsonl for external
|                             trainers (Unsloth/TRL).
|-- her.config.example.json   every training knob, with defaults.
|-- chatbot.config.example.json  decoding knobs, with defaults.
|-- data/                     your export, tokenizer, checkpoint. Gitignored.
|   input/wa_out.txt          the WhatsApp export
|   input/description.txt     the persona sheet
|   input/bpe.json            tokenizer, written on a fresh run
|   her_model.pt              the trained weights
|-- test/                     scratch tests. Gitignored: they read the private
|                             export and the live checkpoint.
|   verify.py                 py -3 test/verify.py
`-- pyproject.toml
```

Data flow:

```
wa_out.txt ──> her.py (parse_wa) ──> corpus ──> target_char_mask ──>
    encode_masked ──> windows ──> GPT ──> data/her_model.pt ──> chatbot.py
```

Both `her.py` and `chatbot.py` read their speaker names from your local
`her.config.json` / `chatbot.config.json` (gitignored). The names are never
written into the source, so they cannot end up in git history. Copy the
example configs to start:

```
copy her.config.example.json her.config.json
copy chatbot.config.example.json chatbot.config.json
```

Train, then talk to it:

```
py her.py
py chatbot.py
```
