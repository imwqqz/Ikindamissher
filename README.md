A QLoRA fine-tune of Qwen3-1.7B on a WhatsApp export: she chats in a REPL,
through a 24/7 daemon, or out loud with a local voice, retrieving similar
conversations from the corpus as few-shot context, learning from every live
chat, and auditing every reply.

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
uv run python server\service.py --fresh   # archive data/live, forget the chat
uv run python client\chat.py
uv run python tools\read_log.py --tail 10 --full

# or talk with your voice (edge TTS, Piper fallback + Whisper STT):
uv run python -m piper.download_voices es_AR-daniela-high --download-dir data/voice/piper
uv run python voice\server.py      # local voice service on :8779
uv run python client\voice.py
```

## Voice

`voice/server.py` is a local service on `:8779`. `POST /speak` renders text to
wav, one sentence at a time, with an edge neural voice (default
`es-UY-ValentinaNeural`; edge sends the text to Microsoft, so it is not
private) and falls back to the Piper ONNX voice when the network is down.
`POST /listen` records until silence or `POST /listen/stop` and transcribes
with Whisper; `GET /health` reports `state`, `rms` and `since` while it
records. `client/voice.py` wires it to the daemon: Enter to talk, Enter again
to stop, and she answers out loud.

A/B the Spanish voices, then pass your pick to the service:

```ps1
uv run python tools/voice_ab.py          # data/voice/ab/*.wav
uv run python voice\server.py --tts-voice es-AR-ElenaNeural --tts-rate -4% --tts-pitch -15Hz
```

Piper stays as the offline fallback; download a Spanish voice once (the
service picks the first `.onnx` in `data/voice/piper/`):

```ps1
uv run python -m piper.download_voices es_AR-daniela-high --download-dir data/voice/piper
```

## Project structure

```
ikindamissher/
├── client/
│   ├── chat.py             # REPL over the daemon's HTTP API (no model loaded)
│   ├── repl.py             # chat REPL that loads the model
│   └── voice.py            # voice chat: talk -> daemon -> speaks
├── voice/                  # local voice service
│   ├── engine.py           #   edge TTS + Piper fallback (text -> wav)
│   ├── server.py           #   :8779 /health, /speak, /listen
│   └── spike.py            #   pre-flight: load + one sentence
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
│   ├── live/               #   live_chats.txt, chat_session.json,
│   │                       #   proactive.sqlite3, telemetry.sqlite3
│   └── voice/piper/        #   Piper .onnx voices
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

## License

This project is under the [LICENCE](https://github.com/imwqqz/Ikindamissher/blob/master/LICENSE).
