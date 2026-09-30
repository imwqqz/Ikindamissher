import sys
from pathlib import Path

import torch

from config import build_cfg
from model import build_model
from tokenizer import BytePairEncoder, bpe_from_state

def load_checkpoint(args, rank, world):
    # Resume from the checkpoint on rank 0, broadcast to the world (torchrun data parallel).
    ckpt = Path(args.ckpt)
    loaded = None
    if ckpt.exists() and not args.fresh and rank == 0:
        try:
            loaded = torch.load(ckpt, map_location="cpu", weights_only=False)
            print(f"resuming from {ckpt} (loss {loaded['loss']:.4f}, step {loaded['step']:,})")
        except Exception as exc:
            print(f"could not load checkpoint: {exc}; starting fresh")
    if args.ddp:
        import torch.distributed as dist
        loaded = [loaded]
        dist.broadcast_object_list(loaded, src=0)
        loaded = loaded[0]
    return loaded
def load_tokenizer(args, corpus, state, rank):
    # Tokenizer: from checkpoint state, from cache, or train fresh on rank 0.
    bpe_path = Path(args.tokenizer_file)
    if state is not None:
        tok = bpe_from_state(state["tokenizer"])
        if rank == 0:
            print(f"tokenizer from checkpoint: {tok.vocab_size - tok.n_sentinels:,} tokens")
        return tok
    if bpe_path.exists() and not args.fresh:
        tok = BytePairEncoder.load(bpe_path)
        if rank == 0:
            print(f"tokenizer cache: {bpe_path}")
        return tok
    return _train_tokenizer(args, corpus, rank, bpe_path)
def _train_tokenizer(args, corpus, rank, bpe_path):
    # BytePairEncoder.train can run on rank 0 only; other ranks load after the barrier.
    tok = None
    if rank == 0:
        print("training byte-level BPE tokenizer...")
        tok = BytePairEncoder(n_sentinels=100)
        tok.train(corpus, args.vocab_size)
        bpe_path.parent.mkdir(parents=True, exist_ok=True)
        tok.save(bpe_path)
        print(f"tokenizer saved: {bpe_path}")
    if args.ddp:
        import torch.distributed as dist
        dist.barrier()
    if tok is None:
        tok = BytePairEncoder.load(bpe_path)
    return tok
def _print_resume_override(args, stored_epochs, rank):
    if rank != 0 or not stored_epochs or args.epochs == stored_epochs:
        return
    print(f"resume: --epochs {args.epochs} overrides checkpoint's {stored_epochs}",
          flush=True)


def _report_remaining_epochs(args, rank, stored_target, stored_spe,
                            stored_step):
    if rank != 0:
        return
    done = stored_step / stored_spe
    print(f"resume: {done:.2f} of {stored_target} epoch(s) done "
          f"({stored_step:,}/{stored_target * stored_spe:,} steps), "
          f"running {args.epochs} more (pass --epochs to add more)",
          flush=True)


def _report_legacy_epochs(args, rank, stored_done):
    if rank != 0:
        return
    print(f"resume: checkpoint predates per-epoch tracking; its epoch "
          f"count ({stored_done}) was the request, not progress, so "
          f"training that many more. Pass --epochs to be precise.",
          flush=True)


def reconcile_epochs(args, state, rank):
    # Explicit --epochs wins, else the remaining shortfall of the original target.
    stored_done = state.get("epochs") or 0
    stored_target = state.get("epochs_target")
    stored_step = state.get("step") or 0
    stored_spe = state.get("steps_per_epoch")
    if args.epochs_explicit:
        _print_resume_override(args, stored_target or stored_done, rank)
        return
    if stored_target and stored_spe:
        # Shortfall in steps, not epochs: the best checkpoint sits in a partial epoch.
        remaining = max(0, stored_target * stored_spe - stored_step)
        args.epochs = max(1, remaining // stored_spe)
        _report_remaining_epochs(args, rank, stored_target, stored_spe,
                                 stored_step)
        return
    if stored_done:
        # Legacy checkpoint: epochs recorded the request, not completed work.
        args.epochs = stored_done
        _report_legacy_epochs(args, rank, stored_done)
def _flag_given(argv, name):
    return name in (sys.argv[1:] if argv is None else argv)


def _setting_given(args, argv, flag, key):
    """True when a setting was asked for, on the command line or in a config.

    A config value is as explicit a request as the same flag.
    """
    return (flag in (sys.argv[1:] if argv is None else argv)
            or key in getattr(args, "_from_config", ()))
def resolve_model(args, state, tok, rank, argv=None):
    # Reuse the checkpoint's architecture when possible; otherwise build from flags.
    if state is None:
        cfg = build_cfg(args, tok)
        return cfg, build_model(cfg)
    # Copy: cfg aliases state["cfg"], so writing to it would mutate the loaded
    # checkpoint in place.
    cfg = dict(state["cfg"])
    args.objective = cfg["objective"]
    # dropout is not architecture, so an explicit --dropout survives a resume.
    if _setting_given(args, argv, "--dropout", "dropout"):
        cfg["dropout"] = args.dropout
        if rank == 0:
            print(f"dropout:         {args.dropout} (explicit, overriding "
                  f"the checkpoint's {state['cfg']['dropout']})")
    # The mask changes which tokens are scored, so flipping it across a resume
    # makes the loss curve incomparable. The checkpoint's setting stands.
    if _setting_given(args, argv, "--target-only-loss", "target_only_loss"):
        cfg["target_only_loss"] = bool(args.target_only_loss)
    args.target_only_loss = bool(cfg.get("target_only_loss", False))
    # epochs is stored at checkpoint top level, not inside cfg.
    reconcile_epochs(args, state, rank)
    model = build_model(cfg)
    try:
        # allow lora/quant flags to resume from identical architecture
        model.load_state_dict(state["model"])
    except RuntimeError:
        pass
    else:
        return cfg, model
    if rank == 0:
        print("architecture mismatch with checkpoint; retraining from scratch")
    cfg = build_cfg(args, tok)
    return cfg, build_model(cfg)
