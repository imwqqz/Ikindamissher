# TODO

## AI CHATBOT

- [ ] [Feature] Persistent chat sessions: save and resume REPL context (--session) — WQQ-80
- [ ] [Feature] Memory implementation chatbot — WQQ-47
- [ ] [Test] Add a regression test for greeting replies — WQQ-79
- [ ] [Feature] Restore the data extraction tooling — WQQ-76
- [ ] [Bug] Reconcile speaker labels across config, corpus, and REPL — WQQ-74
- [ ] [Bug] Diagnose why chatbot replies read out-of-distribution — WQQ-73
- [ ] [Bug] Stop truncating target tail tokens in batch collation — WQQ-72
- [ ] [Bug] Handle odd element counts in NF4 parity packing — WQQ-71
- [ ] [Bug] Guard the BPE pair key above vocab size 4096 — WQQ-70

## Ideas

Unfiled directions discussed but not yet scoped or created in Linear.

- [ ] [Idea] True in-chat memorization: session facts + optional LoRA adapter so she learns from the conversation
  - Facts (e.g. "your name is One") → session store + prompt injection: reliable, deterministic, loadable.
  - Behavior changes → optional ephemeral LoRA adapter (sidecar artifact, never mutate data/her_model.pt).
  - Open questions: explicit vs implicit fact capture; key→value vs free-text; session-scoped vs global profile; weights at all?
- [ ] [Idea] Minecraft integration: she joins the game as a companion
  - Design A (recommended first): NPC/companion that chats and reacts via a mod/bridge calling the model.
  - Design B (later, separate issue): autonomous play — needs a new perception→policy→action layer the repo does not have.
  - Needs decisions: Fabric/Forge + MC version, single-player vs server, in-world entity vs chat-only, model delivery (local HTTP API in front of chatbot.py vs embedded).
  - Privacy: chat must stay local; no personal facts in committed files (same rules as data/ and configs/).
