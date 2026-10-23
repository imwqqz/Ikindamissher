# Ikindamissher

A QLoRA fine-tune of Qwen3-1.7B on a WhatsApp export: she chats in a REPL or
through a 24/7 daemon, retrieving similar conversations from the corpus as
few-shot context, learning from every live chat, and auditing every reply.

## Demo

```text
$ uv run python client/repl.py
device: cuda
adapter loaded: data\her_qwen_adapter
few-shot retrieval on data/input/wa_out.txt
chatting with her - type 'exit' to quit
her: holis
<you>: tqm
her: y eso te hace mal
<you>: exit
```

## Commands

```ps1
# working directory: D:\dev\Git\Ikindamissher

uv sync                       # one-time setup (see Installation below)

uv run python tools/train_tokenizer.py     # build data/input/bpe.json (once)
uv run python tools/export_sft.py --out data/sft.json --persona
uv run python src/engines/qwen/qwen_lm.py  # fine-tune Qwen3 + the her adapter
uv run python tools/train_from_chats.py    # fold live chats in (manual, later)

uv run python client/repl.py   # chat REPL (loads the model in-process)
uv run python tools/checks.py  # self-checks on synthetic data; exit 0 required

# or chat through the 24/7 daemon (the client loads no model):
uv run python server\service.py
uv run python client\chat.py
uv run python tools\read_log.py --tail 10 --full
```

## Project structure

```
ikindamissher/
├── client/
│   ├── chat.py             # REPL over the daemon's HTTP API (no model loaded)
│   └── repl.py             # chat REPL that loads the model
├── server/
│   └── service.py          # 24/7 HTTP daemon: persistent session, proactive
│                           #   queue in SQLite, quiet-hours scheduler
├── src/                    # library modules (imported from the source tree)
│   ├── tokenizer.py        # byte-level BPE for demo retrieval
│   ├── data.py             # WhatsApp export -> speaker turns, persona, Q&A
│   ├── config.py           # JSON config files + argparse plumbing
│   ├── chatlog.py          # live turns -> data/live/live_chats.txt
│   ├── turnlog.py          # turn audit -> data/live/telemetry.sqlite3
│   └── engines/
│       ├── dispatch.py     # engine door: assets, prompts, replies
│       └── qwen/
│           ├── qwen_lm.py  # Qwen3 + her QLoRA: train, load, generate
│           └── chat.py     # chat rows, message building, reply cleanup
├── tools/                  # checks.py, train_tokenizer.py, export_sft.py,
│                           #   train_from_chats.py, read_log.py
├── configs/                # chatbot.config.example.json + gitignored locals
├── data/                   # gitignored: export, persona, BPE, adapter, sft
│   └── live/               #   live_chats.txt, chat_session.json,
│                           #   proactive.sqlite3, telemetry.sqlite3
├── .python-version
├── pyproject.toml
└── uv.lock
```

Data flow: `parse_wa` normalizes the export into `speaker: message` turns;
`build_demo_index` + `few_shot_pairs` retrieve similar exchanges into the
system text; `engines/qwen/chat.py` builds the message list and
`qwen_lm.reply` samples a reply. Training runs the other way:
`export_sft.py` writes `data/sft.json`, and `qwen_lm.train` fits the adapter
with completion-only labels.

## References

* Byte-pair encoding                                 https://arxiv.org/abs/1508.07909
    the retrieval tokenizer's merge loop (tokenizer.py)
* Language Models are Unsupervised Multitask Learners
  (GPT-2)                                            https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
    the byte-level BPE pretokenizer regex (tokenizer.py's BPE_PAT)
* LoRA: Low-Rank Adaptation of Large Language Models https://arxiv.org/abs/2106.09685
    low-rank update W = W0 + (alpha/r) * BA (4.1), applied by peft
    (qwen_lm.py's LoraConfig)
* QLoRA: Efficient Finetuning of Quantized LLMs     https://arxiv.org/abs/2305.18290
    4-bit NormalFloat (NF4) block-wise quantization (3.1), double quantization
    of the constants (3.2), frozen base + trainable adapters (3.4 / Algorithm 1)
* TRL SFTTrainer (train_on_prompt=false)             https://huggingface.co/docs/trl/sft_trainer
    completion-only loss: score the assistant's turns, not the prompt
    (encode_row's label mask in qwen_lm.py)

## License

This project is under the [LICENCE](https://github.com/imwqqz/Ikindamissher/blob/master/LICENSE).
