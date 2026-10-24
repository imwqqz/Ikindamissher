# Todo

## P0

> Done 2026-10-23: `/listen/stop` + `state`/`rms` on `/health`, the client status
> line, and edge TTS with a Piper fallback are in. Follow-up: a silence gate and
> a degenerate-transcript filter stop Whisper inventing text on an empty mic
> (`www.mooji.org`), a typed/pasted line at the prompt is now a text turn
> instead of an ignored recording, and `last_tts` on `/health` plus a client
> warning expose a silent Piper fallback. Covered by `voice-text-turn`,
> `voice-hallucination` and `fresh-archive` checks. Left: one live client run
> (daemon + mic, incl. Enter-to-stop) and picking the voice from
> `data/voice/ab/`.

- [ ] Voice is push-to-talk as a toggle, not hold: Enter starts recording, Enter
  again stops. Same change adds the live status line, so a dead mic is not
  indistinguishable from a slow Whisper: recording + RMS, transcribing, waiting
  for her, speaking. `voice/server.py`: `POST /listen/stop` (`threading.Event`
  checked by the record loop) plus `state` / `rms` / `since` on `/health`;
  `client/voice.py`: run `listen()` on a thread and poll `/health` for the line.
  Keep `--silence` / `--max-seconds` as the safety net. Hold-to-release stays
  out: a console cannot report key-up without `GetAsyncKeyState` polling.
- [ ] Replace Piper as the default voice: it is clear but flat, and no Piper
  voice sounds like a person on a call. Shipped: `edge-tts` behind the
  `text -> wav` contract in `voice/engine.py`, one synthesis per sentence,
  `--tts {edge,piper}` with automatic Piper fallback, `last_tts` on `/health`
  and a client warning when the fallback fires, A/B harness in
  `tools/voice_ab.py`. Left: pick a voice from `data/voice/ab/` and set it as
  the default. XTTS-v2 (local, GPU, own venv, 20-30 s rioplatense reference)
  only if the catalog voice still sounds like an assistant. Spanish only - the
  persona is rioplatense, so Chinese-first models (GPT-SoVITS, CosyVoice) are
  out.

## Unfinished Plan

### Must Fix

- [ ] `tools/export_sft.py`: `_qa_rows` reports dropped unpaired questions / stray answers — confirm counts on a malformed Q&A file.
- [ ] `src/engines/qwen/qwen_lm.py` `encode_row`: a literal `<|im_end|>` inside chat content splits the loss span. Upgrade only if a generated reply actually contains that string.

### Should Fix

- [ ] `voice/server.py`: `/listen` uses a fixed RMS threshold. Take the speaking
  RMS from the live run, pick a default that fits the real mic (0.02 vs ~0.017
  room noise is close), and document `--threshold` / `--silence` in README.
- [ ] `src/engines/dispatch.py`: `_choose` scores with retrieval IDF weights, but proactive replies pass `{}` — decide whether opener scoring needs weights.
- [ ] `server/service.py`: `handle_message` holds `session_lock` while generating, so a slow generation blocks session reads; consider narrowing the lock.
- [ ] `tools/checks.py`: session/queue/turnlog checks use temp paths; add one check that the live `data/live/` defaults resolve correctly.
- [ ] `tools/export_sft.py` token report is `chars // 4`, a guess. Use the real tokenizer before trusting budget numbers.
- [ ] `src/config.py`: `move_legacy` runs at import; call it once from each entry point instead, so an import during a live daemon cannot move files under it.

### Reply Quality

- [ ] Measured on 40 turns (`data/live/telemetry.sqlite3` plus the archives):
  30 had every candidate at 0.000, 9 replies ran over two lines, and 0 echoed a
  demo line verbatim. So the scorer is mostly inert and the tie-break does the
  picking, while `--candidates 3` mostly triples latency. Next: compare
  `--candidates 1` and `--shots 4` on the daemon, and decide the
  proactive-weights question above. Acceptance: no reply over two lines in 20
  turns.

### Voice

- [ ] Whisper mishears casual rioplatense (`holies te quiera mucho` for `hola te
  quiero mucho`); compare `--whisper openai/whisper-medium` on the same clips
  before changing the default.
- [ ] `client/voice.py` playback uses `winsound` (Windows only); document or add a cross-platform fallback.

### Integration Test

- [ ] Create `test_all.py`: training pipeline (tokenizer -> export -> dry-run), imports, and live daemon/voice endpoints; support `--with-services`.
- [ ] Add the live integration test to `tools/checks.py` as an opt-in group (no model load in default checks).

### Avatar / Hologram

- [ ] `avatar-hologram` branch: three.js + VRM page served by the daemon; mouth moves with `/speak` audio; idle blink.
- [ ] Reuse `voice/server.py` `/speak`; no daemon surgery - the browser fetches the wav and plays it.
- [ ] Pick a VRM model (free VRoid sample or one made in VRoid Studio).

### WhatsApp Bridge

- [ ] Own branch. Unofficial automation risks a ban; the official path is the WhatsApp Business API.
- [ ] Integration point is the daemon HTTP API (`POST /message`, `GET /poll`), so no daemon surgery should be needed.
- [ ] Request logging (`server/service.py` `_access`) is the tool for debugging what the bridge sends.
