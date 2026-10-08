import random
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

from checkpoint import load_checkpoint, load_tokenizer, resolve_model
from config import parse_args, request_cpu_fallback, resolve_block
from data import (
    build_corpus, configure_speakers, conversation_ends, her_turn_ends,
    persona_lines, speaker_names, target_char_mask,
)
from model import build_model, trainable_count, total_count
from training import Trainer

def load_corpus(args):
    data = Path(args.data)
    if not data.exists():
        print(f"training data not found at {data}")
        raise SystemExit(1)
    return build_corpus(data, Path(args.description), user=args.user,
                        val_fraction=args.val_fraction,
                        qa_path=Path(args.persona_qa)
                        if getattr(args, "persona_qa", None) else None,
                        persona_repeat=int(getattr(args, "persona_repeat", 0)
                                           or 0))
def setup_ddp():
    """Initialize the process group from torchrun env vars
    (data-parallel scaling of the training loop)."""
    import os
    import torch.distributed as dist
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    dist.init_process_group(backend="nccl")
    return rank, world, local
def init_process(args):
    # Device/process-group setup: DDP via torchrun, otherwise CUDA-only. (Data-parallel scaler.)
    rank, world, local = 0, 1, 0
    if args.ddp:
        rank, world, local = setup_ddp()
        device = torch.device("cuda", local)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            if not (args.cpu or request_cpu_fallback()):
                print("CUDA is not available. Exiting.")
                sys.exit(1)
            print("[Warning] running on CPU: training will be slow.")
    if rank == 0:
        print(f"device: {device} (world {world})")
    return device, rank, world, local
def maybe_compile(args, model, rank):
    # torch.compile with eager fallback on Windows (no Triton kernels).
    if not args.compile:
        return model
    if sys.platform == "win32":
        # Triton is not packaged for Windows; degrade to eager
        # (https://pytorch.org/docs/stable/torch.compiler.html)
        torch._dynamo.config.suppress_errors = True
        if rank == 0:
            print("[Warning] Windows: torch.compile will fall back to eager "
                  "(no Triton kernels)")
    model = torch.compile(model)
    if rank == 0:
        print("compiled model with torch.compile")
    return model
def apply_precision(args, rank):
    # Map --precision to a mixed-precision dtype (None = fp32 eager).
    if args.precision != "fp32":
        dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
        if rank == 0:
            print(f"precision: {args.precision}")
        return dtype
    return None
def report_setup(args, cfg, rank, n_train, n_total):
    # Rank-0-only banner describing the model before training starts.
    if rank != 0:
        return
    print(f"model: d_model={cfg['d_model']} layers={cfg['n_layers']} "
          f"heads={cfg['n_heads']} d_ff={cfg['d_ff']} block={cfg['max_len']}")
    if args.gc:
        print("gradient checkpointing enabled (recompute per block)")
    if args.lora or args.qlora:
        print(f"LoRA adapters: r={cfg['lora_r']} alpha={cfg['lora_alpha']} "
              f"dropout={cfg['lora_dropout']}; base "
              f"{'NF4-quantized' if cfg['quant'] == 'nf4' else 'frozen fp32'}")
    print(f"trainable params: {n_train:,} / {n_total:,}")
def _announce_start(args, trainer, rank):
    if rank != 0:
        return
    print(f"training for {args.epochs} epochs (~{trainer.total_steps:,} steps)")
    if trainer.val_windows is not None:
        print(f"validation: {trainer.val_windows.num_windows} held-out windows, "
              f"every {trainer.val_every} steps, patience {trainer.patience}")
    else:
        print("validation: DISABLED (--val-fraction 0) - training loss is "
              "the only signal and early stopping is off")
def _finalize_checkpoint(args, trainer, final_loss, steps_done, rank):
    if rank != 0:
        return
    if trainer.val_windows is None:
        trainer.save(final_loss, steps_done)
        print(f"saved checkpoint: {args.ckpt} (loss {final_loss:.4f}, "
              f"{steps_done:,} steps)")
    elif trainer.best_step:
        if trainer.diverged:
            # Nothing to write: the weights are not trustworthy and the
            # retained checkpoint is the last good one.
            print(f"kept previous checkpoint: {args.ckpt} (run diverged at "
                  f"step {steps_done:,}; weights not saved)")
        elif args.keep_last:
            # Best-val retention is backwards for a memorization model; --keep-last
            # writes the final weights instead.
            if trainer.save(final_loss, steps_done):
                print(f"saved final weights: {args.ckpt} (train loss "
                      f"{final_loss:.4f}, {steps_done:,} steps)")
        else:
            # _maybe_validate already wrote the retained checkpoint.
            print(f"retained checkpoint: {args.ckpt} (best val "
                  f"{trainer.best_val:.4f} at step {trainer.best_step:,}; "
                  f"stopped at {steps_done:,})")
def _teardown_process(args):
    if args.ddp:
        import torch.distributed as dist
        dist.destroy_process_group()
def run_training(args, cfg, model, ids, device, tok, state, rank, world, local,
                 val_ids=None, label_ids=None, val_label_ids=None,
                 prefix_ids=None):
    # Optimizer loop + final checkpoint/summary, then teardown the process group.
    trainer = Trainer(args, cfg, model, ids, device, tok, args.ckpt,
                      start_step=state["step"] if state else 0,
                      rank=rank, world=world, val_ids=val_ids, state=state,
                      label_ids=label_ids, val_label_ids=val_label_ids,
                      prefix_ids=prefix_ids)
    _announce_start(args, trainer, rank)
    t0 = time.perf_counter()
    final_loss, steps_done = trainer.fit()
    elapsed = time.perf_counter() - t0

    _finalize_checkpoint(args, trainer, final_loss, steps_done, rank)
    if rank == 0:
        # final_loss stays the train loss from fit().
        print_summary(args, cfg, tok, len(ids), trainable_count(model),
                      total_count(model), final_loss, steps_done,
                      state["step"] if state else 0, device, elapsed,
                      trainer=trainer)
    _teardown_process(args)
def _init_stdout():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
def _epochs_explicit(args, argv):
    # was --epochs passed on the CLI? resume must not silently override it
    return ("--epochs" in (sys.argv[1:] if argv is None else argv)
            # A config-supplied epochs means the same as the flag.
            or "epochs" in getattr(args, "_from_config", ()))
def _split_corpus(corpus):
    if isinstance(corpus, tuple):
        return corpus[0], corpus[1]
    return corpus, ""
def _masked_labels(tok, text, her_name):
    ends = her_turn_ends(text, her_name)
    scenes = conversation_ends(text)
    ids, flags = tok.encode_masked(text, target_char_mask(text, her_name),
                                   ends, scenes)
    label_ids = [i if f else -100 for i, f in zip(ids, flags)]
    return ids, label_ids


def _encode_turns(tok, text, her_name):
    """ids with scored turn-end and conversation-end sentinels (all scored)."""
    if not text:
        return []
    ids, _ = tok.encode_masked(text, [True] * len(text),
                               her_turn_ends(text, her_name),
                               conversation_ends(text))
    return ids


def _encode_lm(tok, text, her_name, mask_on):
    if mask_on:
        return _masked_labels(tok, text, her_name)
    return _encode_turns(tok, text, her_name), None


def _encode_corpus(args, corpus, val_corpus, her_name, tok):
    """Tokenize train/val corpora; masks make the loss completion-only.

    lm injects the scored turn-end sentinel so the model can end a multi-message
    turn; the span objective builds its own sentinel labels and is left alone.
    """
    mask_on = bool(getattr(args, "target_only_loss", False)) \
        and args.objective == "lm"
    if args.objective == "lm":
        ids, label_ids = _encode_lm(tok, corpus, her_name, mask_on)
        val_ids, val_label_ids = (
            (None, None) if not val_corpus
            else _encode_lm(tok, val_corpus, her_name, mask_on))
    else:
        ids, label_ids = tok.encode(corpus), None
        val_ids, val_label_ids = (
            (tok.encode(val_corpus), None) if val_corpus else (None, None))
    scored = (sum(1 for x in label_ids if x != -100) if mask_on else len(ids))
    return ids, label_ids, val_ids, val_label_ids, scored, mask_on
def _persona_prefix_ids(args, her_name, tok):
    # Persona prefix: the same fact sheet that chatbot.py puts at the head of an
    # inference prompt, so training and inference see one identical layout.
    if not getattr(args, "persona_prefix", False):
        return None
    desc = Path(args.description)
    if not desc.exists():
        return None
    head = "\n".join(persona_lines(desc.read_text(encoding="utf-8"),
                                   alias=her_name)) + "\n"
    prefix_ids = tok.encode(head)
    budget = max(1, args.block // 2)
    if len(prefix_ids) > budget:
        prefix_ids = prefix_ids[-budget:]
    return prefix_ids
def _report_corpus(args, rank, tok, ids, scored, prefix_ids, val_ids, mask_on):
    if rank != 0:
        return
    print(f"corpus: {len(ids):,} tokens, vocab {tok.vocab_size:,}")
    if mask_on:
        print(f"loss: completion-only, {scored:,} of {len(ids):,} tokens "
              f"scored ({100 * scored / max(len(ids), 1):.1f}%) - the rest "
              f"are {speaker_names()[0]}'s turns and persona labels")
    if prefix_ids:
        print(f"persona prefix: {len(prefix_ids)} tokens on every window, "
              f"masked from the loss; {max(1, args.block - len(prefix_ids))} "
              f"tokens of dialogue per window")
    if val_ids:
        print(f"held out: {len(val_ids):,} tokens for validation "
              f"({100 * len(val_ids) / (len(ids) + len(val_ids)):.1f}%)")
def main(argv=None):
    _init_stdout()

    args = parse_args(argv)
    args.epochs_explicit = _epochs_explicit(args, argv)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    # before load_corpus, which labels the persona turns
    configure_speakers(args.user_speaker, args.her_speaker)

    device, rank, world, local = init_process(args)
    state = load_checkpoint(args, rank, world)
    # Must run before resolve_model: build_cfg is skipped when resuming, so
    # args.block would stay None and BatchSampler would fail.
    resolve_block(args, state, rank)
    corpus = load_corpus(args)
    # With a validation split, fit the tokenizer on train text only so no
    # held-out tokens can leak into the vocabulary.
    corpus, val_corpus = _split_corpus(corpus)
    tok = load_tokenizer(args, corpus, state, rank)
    cfg, model = resolve_model(args, state, tok, rank, argv)

    n_train = trainable_count(model)
    n_total = total_count(model)
    report_setup(args, cfg, rank, n_train, n_total)

    model = maybe_compile(args, model, rank)
    args.precision_dtype = apply_precision(args, rank)

    model = model.to(device)
    if args.ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local])

    her_name = speaker_names()[1]
    ids, label_ids, val_ids, val_label_ids, scored, mask_on = \
        _encode_corpus(args, corpus, val_corpus, her_name, tok)
    prefix_ids = _persona_prefix_ids(args, her_name, tok)
    _report_corpus(args, rank, tok, ids, scored, prefix_ids, val_ids, mask_on)

    run_training(args, cfg, model, ids, device, tok, state, rank, world, local,
                 val_ids=val_ids, label_ids=label_ids,
                 val_label_ids=val_label_ids, prefix_ids=prefix_ids)
def print_summary(args, cfg, tok, n_tokens, n_train, n_total, final_loss,
                  steps_done, start_step, device, elapsed, trainer=None):
    print("----------------------------------------")
    print("training summary")
    print("----------------------------------------")
    if trainer is not None and trainer.val_windows is not None:
        print(f"val loss (best):  {trainer.best_val:.4f} at step "
              f"{trainer.best_step:,}")
        print(f"val loss (last):  {trainer.last_val:.4f}"
              if trainer.last_val == trainer.last_val else
              "val loss (last):  n/a")
        print(f"early stop:      {'yes' if trainer.stopped_early else 'no'}"
              f" (patience {trainer.patience}, every {trainer.val_every} steps)")
        if trainer.stopped_early:
            # train falling while val rises is overfitting, which here is the goal.
            print(f"note:           train {final_loss:.4f} vs val "
                  f"{trainer.best_val:.4f}; a rising val loss here means "
                  f"memorizing, not a problem to fix")
    else:
        print("val loss:        DISABLED")
    print(f"train loss:      {final_loss:.4f}")
    print(f"steps:           {steps_done:,} total ({start_step:,} carried over)")
    print(f"objective:       {cfg['objective']}")
    print(f"epochs:          {args.epochs}")
    print(f"block (max_len): {cfg['max_len']}")
    print(f"batch size:      {args.batch}")
    print(f"grad accum:      {args.grad_accum}")
    print(f"vocab:           {tok.vocab_size:,}")
    print(f"corpus tokens:   {n_tokens:,}")
    print(f"d_model:         {cfg['d_model']}")
    print(f"n_layers:        {cfg['n_layers']}")
    print(f"n_heads:         {cfg['n_heads']}")
    print(f"d_ff:            {cfg['d_ff']}")
    print(f"params:          {n_train:,} trainable / {n_total:,} total")
    print(f"lora:            r={cfg['lora_r']}, alpha={cfg['lora_alpha']}" if cfg['lora'] else "lora:            off")
    print(f"quant:           {cfg['quant']}")
    print(f"lr:              {args.lr}")
    print(f"grad clip:       {getattr(args, 'grad_clip', 1.0)}")
    print(f"warmup:          {args.warmup_steps}")
    print(f"schedule:        {args.schedule}")
    print(f"dropout:         {cfg['dropout']}")
    print(f"label smoothing: {args.label_smoothing}")
    print(f"device:          {device}")
    print(f"time:            {elapsed / 60:.1f} min")
    print(f"checkpoint:      {args.ckpt}")
if __name__ == "__main__":
    main()
