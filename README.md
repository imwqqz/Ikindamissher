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

## Code example

Run everything from the repo root:

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

## Chat recipe

The brain is Qwen3-1.7B with the `her` QLoRA adapter trained on the export
(`src/engines/qwen/qwen_lm.py`; adapter in `data/her_qwen_adapter/`). Chat with
retrieval on: `client/repl.py` matches similar user/persona exchanges from the
export and puts them in the system text, chained within one conversation
instead of unrelated pairs stitched together.

Speaker labels come from `configs/chatbot.config.json` (`user_speaker`,
`her_speaker`); they are kept out of the source on purpose. The retrieval
tokenizer (`data/input/bpe.json`) is built by `tools/train_tokenizer.py`; it
only affects demo matching, while generation uses Qwen's own tokenizer.

`sft.json` is produced by `tools/export_sft.py` (one JSONL row per
conversation, completion-only labels applied in `qwen_lm.encode_row`). Long
messages are dropped by `--max-chars`: a pasted essay is not conversation and
teaches the model to write essays.

**Live memory.** Every console/daemon exchange is appended to
`data/live_chats.txt` in the same corpus format (a silent hour starts a new
conversation). `load_corpus` reads that file too, so retrieval remembers past
chats with no training. To teach the weights, run
`uv run python tools/train_from_chats.py --epochs 1`: it backs up the adapter,
writes `data/sft_combined.json` (export rows + live rows, deduplicated), runs a
short QLoRA train, and prints the restart reminder.

**Every reply is audited.** Both clients sample `--candidates` replies
(default 3), keep the one that best overlaps her retrieved turns, and write a
row to `data/telemetry.sqlite3`: the candidates with their scores, the chosen
reply, why it won, the exact context she replied from (persona, memory demos,
history) and the thought. Read it with `uv run python tools/read_log.py --tail
10 --full` (or `--json`). Nothing in the audit path slows the reply.

`--think` enables Qwen3 reasoning: the thought is logged, never spoken. It adds
generation time, and the adapter was trained with thinking off, so use it for
inspection rather than daily chat.

## Project structure

```
Ikindamissher/
├── src/                       # library modules (imported from the source tree)
│   ├── tokenizer.py           # byte-level BPE for demo retrieval
│   ├── data.py                # WhatsApp export -> user/persona turns, persona
│   ├── chatlog.py             # live turns -> data/live_chats.txt (corpus format)
│   ├── turnlog.py             # turn audit log -> data/telemetry.sqlite3
│   ├── config.py              # JSON config files + argparse plumbing
│   └── engines/
│       ├── dispatch.py        # engine door: assets, prompts, replies
│       └── qwen/
│           ├── qwen_lm.py     # Qwen3 + her QLoRA: train, load, generate
│           └── chat.py        # chat rows, message building, reply cleanup
├── client/repl.py             # chat REPL that loads the model
├── client/chat.py             # thin REPL over the daemon's HTTP API
├── server/service.py          # 24/7 HTTP daemon (poll transport, proactive)
├── tools/                     # checks.py, train_tokenizer.py, export_sft.py,
│                              #   train_from_chats.py, read_log.py
├── configs/                   # chatbot.config.example.json + gitignored locals
├── data/                      # gitignored: export, persona, BPE, adapter, sft
├── .python-version            # 3.14.5
├── pyproject.toml
└── uv.lock
```

Data flow: `parse_wa` normalizes the export into `speaker: message` turns;
`build_demo_index` + `few_shot_pairs` retrieve similar exchanges into the
system text; `engines/qwen/chat.py` builds the message list and
`qwen_lm.reply` samples a reply. Training runs the other way:
`export_sft.py` writes `data/sft.json`, and `qwen_lm.train` fits the adapter
with completion-only labels.

Contributions, acknowledgments, this project is under [LICENCE](https://github.com/imwqqz/Ikindamissher/blob/master/LICENSE)
