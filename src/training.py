import math
import random
import time
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F

from data import speaker_names
from tokenizer import bpe_state

# Global gradient-norm clip (Playbook: pick the threshold just above the
# typical norm; telemetry below shows whether this value is ever binding).
GRAD_CLIP = 1.0

# Causal LM loss with label smoothing (Attention Is All You Need, section 5.4 https://arxiv.org/abs/1706.03762)
def lm_loss(logits, targets, label_smoothing):
    flat = logits[:, :-1].reshape(-1, logits.size(-1))
    tgt = targets[:, 1:].reshape(-1)
    # -100 = excluded from the loss. Dropped before reducing, else the gradient
    # is rescaled by the scored fraction.
    keep = tgt.ne(-100)
    if not bool(keep.any()):
        # An all-one-speaker window has no valid target; cross_entropy returns NaN.
        return logits.sum() * 0.0
    if not bool(keep.all()):
        flat = flat[keep]
        tgt = tgt[keep]
    if label_smoothing <= 0:
        return F.cross_entropy(flat, tgt)
    vocab = flat.size(-1)
    logp = F.log_softmax(flat, dim=-1)
    correct = -logp.gather(1, tgt.unsqueeze(1)).squeeze(1)
    spread = -logp.sum(dim=-1)
    share = label_smoothing / vocab * spread
    return ((1 - label_smoothing) * correct + share).mean()

# T5-style span corruption on a single sequence (T5, section 3.2 https://arxiv.org/abs/2005.14165)
def mask_spans(mask, span_gap):
    # Coalesce masked runs into (start, end) spans, merging across small gaps.
    # Run boundaries come from one diff instead of an element-wise Python scan;
    # the short merge pass then touches only the runs, not the tokens.
    flags = mask.to(torch.int8)
    changes = torch.diff(flags, prepend=flags.new_zeros(1),
                         append=flags.new_zeros(1))
    starts = (changes == 1).nonzero().reshape(-1).tolist()
    ends = (changes == -1).nonzero().reshape(-1).tolist()
    spans = []
    for start, end in zip(starts, ends):
        if spans and start - spans[-1][1] <= span_gap:
            spans[-1] = (spans[-1][0], end)
        else:
            spans.append((start, end))
    return spans


def span_corrupt_pair(row, p=0.15, span_gap=3, encoder=None):
    mask = torch.rand(row.numel(), dtype=torch.float32) < p
    spans = mask_spans(mask, span_gap)
    if not spans:
        return row.clone(), torch.full((max(row.numel() - 1, 1),), -100, dtype=torch.long)
    inputs = []
    targets = []
    sent = 0
    prev = 0
    for s, e in spans:
        inputs.append(row[prev:s])
        inputs.append(torch.tensor([encoder.sentinel_id(sent)], dtype=row.dtype))
        targets.append(torch.tensor([encoder.sentinel_id(sent)], dtype=row.dtype))
        targets.append(row[s:e])
        prev = e
        sent += 1
    inputs.append(row[prev:])
    targets.append(torch.tensor([encoder.sentinel_id(sent)], dtype=row.dtype))
    inp = torch.cat(inputs)
    tgt = torch.cat(targets)
    full = torch.cat([inp, tgt])
    # predict only the target tokens: positions [len(inp)-1, len(full)-1)
    labels = torch.cat([torch.full((inp.numel() - 1,), -100, dtype=torch.long), tgt])
    return full, labels
def collate_span(rows, encoder):
    """Pad corrupted sequences + labels so logits[:, :-1] aligns with labels.

    Each row (ii, ll) already holds the full stream -- corrupted input then
    targets -- and ll is that stream shifted by one (predicting the next token,
    masked to the target positions). The fixed batch width truncates the tail of
    rows whose stream overruns max_inp (labels lose only tokens that also left
    the input row, so alignment is preserved).
    """
    max_inp = max(r[0].numel() for r in rows)
    max_lab = max_inp - 1
    inp = torch.zeros(len(rows), max_inp, dtype=torch.long)
    lab = torch.full((len(rows), max_lab), -100, dtype=torch.long)
    for bi, (ii, ll) in enumerate(rows):
        inp[bi, :len(ii)] = ii
        lab[bi, :len(ll)] = ll[:max_lab]
    return inp, lab
# Cross-entropy over shifted labels, ignoring -100 (T5, section 3.2 https://arxiv.org/abs/2005.14165)
def span_loss(logits, labels):
    B, T, V = logits.shape
    flat = logits[:, :-1].reshape(-1, V)
    tgt = labels.reshape(-1)
    return F.cross_entropy(flat, tgt, ignore_index=-100)
# Sliding-window batches shuffled every epoch (Attention Is All You Need, section 5.1 https://arxiv.org/abs/1706.03762)
class BatchSampler:

    def __init__(self, ids, block, batch_size, device, rank=0, world=1,
                 labels=None, prefix=None):
        self.block = block
        self.batch_size = batch_size
        self.device = device
        self.rank = rank
        self.world = world
        self.ids = torch.tensor(ids, dtype=torch.long, device=device)
        # Optional per-token labels, -100 where excluded, sliced at the same offsets
        # as ids.
        self.labels = (None if labels is None
                       else torch.tensor(labels, dtype=torch.long, device=device))
        # Persona as a fixed per-window prefix, so every example is conditioned
        # on the fact sheet instead of only the ~0.15% of windows that reach
        # it at the corpus end. Masked out of the loss: it is context the model
        # must read, not text it should emit. Sourced from the corpus, so no
        # token crosses the block boundary and RoPE stays within max_len.
        self.prefix = (None if not prefix
                       else torch.tensor(prefix, dtype=torch.long, device=device))
        self.n_prefix = 0 if self.prefix is None else int(self.prefix.numel())
        self.span = max(1, block - self.n_prefix)
        # A window needs `span` tokens: len - span valid starts (older code kept
        # one window at len == span). len < span has none, so clamp to zero
        # instead of indexing past the tensor.
        spare = len(self.ids) - self.span
        self.num_windows = 1 if self.ids.numel() and spare == 0 else max(spare, 0)
        if self.num_windows == 0 and self.rank == 0:
            print(f"warning: corpus has {len(self.ids):,} tokens, shorter than "
                  f"block {self.span}; no training windows will be produced",
                  flush=True)
        self.offsets = torch.arange(self.span, dtype=torch.long, device=device)
        if self.n_prefix:
            self.prefix_labels = torch.full((self.n_prefix,), -100,
                                            dtype=torch.long, device=device)

    def batches_per_epoch(self):
        return math.ceil(self.num_windows / self.batch_size / self.world)

    def __iter__(self):
        starts = torch.randperm(self.num_windows, device=self.device)
        starts = starts[self.rank::self.world]
        for i in range(0, starts.numel(), self.batch_size):
            picks = starts[i: i + self.batch_size]
            if picks.numel() == 0:
                continue
            idx = picks.unsqueeze(1) + self.offsets.unsqueeze(0)
            # Unshifted window: lm_loss applies the one-ahead shift itself
            # (Attention Is All You Need, section 5.1 https://arxiv.org/abs/1706.03762)
            if self.n_prefix:
                wid = torch.cat([self.prefix.expand(idx.size(0), -1),
                                 self.ids[idx]], dim=1)
                wlab = torch.cat([self.prefix_labels.expand(idx.size(0), -1),
                                  self.ids[idx] if self.labels is None
                                  else self.labels[idx]], dim=1)
                yield wid, wlab
            else:
                yield self.ids[idx], (self.ids[idx] if self.labels is None
                                      else self.labels[idx])
def _opt_float(value, default):
    # Checkpoints store None for metrics a given run never produced.
    return default if value is None else float(value)


def _finite_or_none(value):
    # The train-loss "best" stays inf forever under validation gating, which
    # torch.save would otherwise persist as a bare inf.
    if value is None:
        return None
    value = float(value)
    return None if value == float("inf") else value
def _capture_rng():
    # Resume must continue the RNG stream, not replay the seed: shuffling and
    # dropout both draw from these generators.
    state = {"torch": torch.get_rng_state(), "python": random.getstate()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state):
    rng = state.get("rng")
    if not rng:
        return
    try:
        torch.set_rng_state(rng["torch"])
        if "cuda" in rng and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        random.setstate(rng["python"])
    except (KeyError, TypeError, ValueError, RuntimeError):
        pass
class WeightEMA:
    # Decay-averaged copy of the trainable weights (Polyak averaging, as used
    # by EMA in the timm/MAE lineage). Buffers are left alone; only the raw
    # parameters a step would move are averaged.

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {name: param.detach().clone().float()
                       for name, param in model.named_parameters()
                       if param.requires_grad}

    def update(self, model):
        decay = self.decay
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    self.shadow[name].mul_(decay).add_(
                        param.detach().float(), alpha=1 - decay)

    def load(self, state):
        for name, value in state.items():
            if name in self.shadow:
                self.shadow[name] = value.float().clone()
class ValWindows:
    # stride == block, no shuffle, so every validation token is predicted once.
    # Yields exactly block tokens: block+1 overflows RoPE at max_len == block.
    def __init__(self, ids, block, device, batch_size=8, labels=None):
        # detach().clone() rather than torch.tensor(): ids is already a tensor
        # and torch.tensor() re-wraps it through __array__, which warns
        self.ids = torch.as_tensor(ids, dtype=torch.long).detach().clone().to(device)
        self.labels = (None if labels is None else
                       torch.as_tensor(labels, dtype=torch.long).detach().clone().to(device))
        self.block = block
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.num_windows = max(0, self.ids.numel() // self.block)

    def __len__(self):
        return (self.num_windows + self.batch_size - 1) // self.batch_size

    def _windows(self):
        n = self.ids.numel()
        for start in range(0, n - self.block + 1, self.block):
            chunk = self.ids[start:start + self.block]
            if chunk.numel() < 2:
                break
            yield chunk

    def __iter__(self):
        # Grouped into batches: one (batch, block) pass beats block interleaved ones.
        # start tracks the global window index: _pack needs it, because labels are
        # sliced at absolute offsets and every batch but the first is non-zero.
        pending = []
        start = 0
        for chunk in self._windows():
            pending.append((chunk, start))
            start += self.block
            if len(pending) == self.batch_size:
                yield self._pack(pending)
                pending = []
        if pending:
            yield self._pack(pending)

    def _pack(self, chunks):
        ids = torch.stack([c for c, _ in chunks])
        if self.labels is None:
            return ids, ids
        # Contiguous slices at stride == block: the n-th window uses the n-th
        # absolute offset, not its position inside this batch.
        labels = torch.stack([self.labels[s:s + self.block] for _, s in chunks])
        return ids, labels

class Trainer:
    def __init__(self, args, cfg, model, ids, device, tokenizer, ckpt,
                 start_step=0, rank=0, world=1, val_ids=None, state=None,
                 label_ids=None, val_label_ids=None, prefix_ids=None):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.tokenizer = tokenizer
        self.ckpt = Path(ckpt)
        self.rank = rank
        self.world = world
        self.epochs = args.epochs
        self.epochs_target = self._resolve_epochs_target(args, state)
        self.label_smoothing = args.label_smoothing
        self.log_every = args.log_every
        self.log_best = args.log_best
        self.grad_accum = max(1, args.grad_accum)
        self.objective = args.objective
        self.start_step = start_step
        self.precision = getattr(args, "precision_dtype", None)
        # fp16 gradients can underflow; GradScaler is a no-op unless fp16 on CUDA.
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.precision == torch.float16
            and self.device.type == "cuda")
        self.sample_every_epoch = 4
        self.val_every = max(1, args.val_every)
        self.patience = max(0, args.patience)
        self.spike_factor = max(0.0, args.spike_factor)
        self.spike_patience = max(1, args.spike_patience)
        self.spike_count = 0
        self.diverged = False
        # Spike reference is the best windowed mean, never the best single minibatch:
        # the loss distribution has a long left tail that one lucky batch sets.
        self.best_window = float("inf")
        self.val_windows = self._make_val_windows(args, device, val_ids,
                                                  val_label_ids)
        self.since_best = 0
        self.stopped_early = False
        self.last_train_loss = float("inf")

        # The span objective builds its own sentinel labels, so the completion
        # mask only applies to plain language modelling.
        self._make_sampler(args, device, rank, world, ids, label_ids,
                           prefix_ids)
        self._make_optimizer(args)

        # Restored across resumes so 'best' is not scoped to this process.
        state = state or {}
        self._restore_best(state)
        # Adam moments are not derivable from the weights; the LR schedule is a pure
        # function of the step counter, so it needs no state.
        self._restore_optimizer(state)
        self._restore_scaler(state)
        _restore_rng(state)
        self.ema_decay = max(0.0, getattr(args, "ema", 0.0))
        self.ema = (WeightEMA(self._unwrap(), self.ema_decay)
                    if self.ema_decay else None)
        self._restore_ema(state)
        # True per-step window, sized independently of --log-every, which only sets
        # how often a line prints. Bounded so it cannot grow with the run.
        self.window_n = max(16, min(args.log_every, 256))
        self.recent = deque(maxlen=self.window_n)
        # Pre-clip gradient norms, for clip-threshold diagnosis (report only).
        self.grad_norms = deque(maxlen=2048)
        self.run_start = time.perf_counter()
        self.window_start = self.run_start
        self.last_logged_step = start_step

    @staticmethod
    def _resolve_epochs_target(args, state):
        # args.epochs means 'to run now'; writing it back as the target shrinks the
        # goal on every resume.
        _stored_target = (state or {}).get("epochs_target")
        return int(_stored_target) if _stored_target else int(args.epochs)

    def _make_val_windows(self, args, device, val_ids, val_label_ids):
        if val_ids and len(val_ids) > args.block + 1:
            return ValWindows(val_ids, args.block, device,
                              labels=val_label_ids)
        return None

    def _make_sampler(self, args, device, rank, world, ids, label_ids,
                      prefix_ids):
        self.sampler = BatchSampler(ids, args.block, args.batch, device,
                                    rank=rank, world=world,
                                    labels=label_ids
                                    if self.objective == "lm" else None,
                                    prefix=prefix_ids)
        self.batches_per_epoch = self.sampler.batches_per_epoch()
        self.total_steps = self.start_step + self.epochs * self.batches_per_epoch

    def _decay_split(self):
        # Weight decay belongs on the linear maps, not on 1-D gains
        # (RMSNorm), token embeddings, or biases. LoRA A/B are linear maps and
        # decay like the weights they adapt.
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() < 2 or "norm" in name or "tok_emb" in name:
                no_decay.append(param)
            else:
                decay.append(param)
        return decay, no_decay

    def _make_optimizer(self, args):
        # Adam with beta2 tuned for transformer training (Attention Is All You Need, section 5.3 https://arxiv.org/abs/1706.03762)
        # TODO: REVIEW: the paper's section 5.3 states beta1=0.9, beta2=0.98,
        # eps=1e-9; this call uses beta2=0.95 and adds weight_decay=1e-2
        # (AdamW, not in the paper). The citation is what differs, not the recipe.
        decay, no_decay = self._decay_split()
        self.optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": 1e-2},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=args.lr, betas=(0.9, 0.95), eps=1e-9,
        )
        self.warmup_steps = args.warmup_steps
        self.schedule = getattr(args, "schedule", "invsqrt")
        self.lr = args.lr
        self.base_lr = (args.lr * math.sqrt(args.warmup_steps)
                        if args.warmup_steps > 0 else args.lr)

    def _restore_best(self, state):
        self.best = _opt_float(state.get("best"), float("inf"))
        self.best_step = int(state.get("best_step") or 0)
        self.best_val = _opt_float(state.get("best_val"), float("inf"))
        # Kept for the summary; best and last drifting apart is the expected signal.
        self.last_val = float("nan")

    def _restore_optimizer(self, state):
        if not state.get("optim"):
            return
        stored_groups = state.get("optim_groups")
        if stored_groups is not None and stored_groups != len(self.optimizer.param_groups):
            print(f"warning: checkpoint optimizer has {stored_groups} param "
                  f"group(s), this build uses {len(self.optimizer.param_groups)}; "
                  f"resuming with a fresh optimizer (Adam moments lost)",
                  flush=True)
            return
        try:
            self.optimizer.load_state_dict(state["optim"])
        except (ValueError, KeyError) as exc:
            print(f"warning: could not restore optimizer state ({exc}); "
                  f"resuming with a fresh optimizer (Adam moments lost)",
                  flush=True)

    def _restore_scaler(self, state):
        # fp16 loss-scale state; absent on fp32 checkpoints, where the scaler
        # is disabled and load_state_dict on it would be meaningless.
        if not state.get("scaler"):
            return
        try:
            self.scaler.load_state_dict(state["scaler"])
        except (ValueError, KeyError):
            pass

    def _restore_ema(self, state):
        if self.ema is None or not state.get("ema"):
            return
        try:
            self.ema.load(state["ema"])
        except (KeyError, TypeError, RuntimeError):
            pass

    def _ema_state(self, state_dict):
        # EMA is stored as fp32; cast back to each weight's dtype so the
        # checkpoint loads into the same model it was averaged from.
        return {name: (self.ema.shadow[name].to(value.dtype)
                       if name in self.ema.shadow and value.is_floating_point()
                       else value)
                for name, value in state_dict.items()}

    def _schedule_lr(self, step):
        # LR schedules after linear warmup (Attention, 5.3 https://arxiv.org/abs/1706.03762;
        # T5, 3.4 https://arxiv.org/abs/2005.14165; SGDR, 2.1 https://arxiv.org/abs/1608.03983)
        if self.warmup_steps <= 0:
            return
        # linear warmup ramp applies to every schedule (Attention Is All You Need, section 5.3 https://arxiv.org/abs/1706.03762)
        if step < self.warmup_steps:
            lr = self.lr * step / self.warmup_steps
        elif self.schedule == "cosine":
            # lr(t) = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(pi*t/T)), t = step - warmup,
            # T_total = total_steps - warmup (SGDR, 2.1 https://arxiv.org/abs/1608.03983)
            t = step - self.warmup_steps
            total_sched = max(self.total_steps - self.warmup_steps, 1)
            lr = self.lr * 0.5 * (1.0 + math.cos(math.pi * t / total_sched))
        elif self.schedule == "constant":
            # flat LR at args.lr after warmup (T5, section 3.4 https://arxiv.org/abs/2005.14165)
            lr = self.lr
        else:
            # inverse-square-root decay after warmup peak base_lr (Attention Is All You Need, section 5.3 https://arxiv.org/abs/1706.03762)
            lr = self.base_lr * min(step ** -0.5, step * self.warmup_steps ** -1.5)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def _forward_loss(self, inputs, targets):
        logits = self.model(inputs)
        if self.objective == "span":
            return span_loss(logits, targets)
        return lm_loss(logits, targets, self.label_smoothing)

    def _train_step(self, inputs, targets):
        # mixed-precision autocast for bf16/fp16 with frozen NF4 base (QLoRA, section 3.4 https://arxiv.org/abs/2305.18290)
        precision = self.precision
        with torch.autocast("cuda", dtype=precision, enabled=precision is not None and self.device.type == "cuda"):
            loss = self._forward_loss(inputs, targets)
        loss = loss / self.grad_accum
        self.scaler.scale(loss).backward()
        return loss

    def evaluate(self):
        # Held-out loss with dropout off and no shuffling. Returns None when no
        # validation split was built.
        if self.val_windows is None:
            return None
        model = self._unwrap()
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for batch, labels in self.val_windows:
                batch = batch.to(self.device, non_blocking=True)
                targets = labels.to(self.device, non_blocking=True)
                with torch.autocast("cuda", dtype=self.precision,
                                    enabled=self.precision is not None
                                    and self.device.type == "cuda"):
                    loss = self._forward_loss(batch, targets)
                # weight by batch size so a trailing partial batch is not
                # counted as heavily as a full one
                n = batch.shape[0]
                total += loss.item() * n
                count += n
        model.train()
        return total / count if count else None

    def _maybe_validate(self, steps, epoch):
        # Only rank 0 stops, but the flag must reach every rank or the others keep
        # entering all-reduce with a peer that left, and hang.
        stop = self._validate_on_rank0(steps, epoch)
        if self.world > 1:
            import torch.distributed as dist
            flag = torch.tensor([1 if stop else 0], device=self.device)
            dist.broadcast(flag, src=0)
            stop = bool(flag.item())
            if stop:
                self.stopped_early = True
        return stop

    def _validate_on_rank0(self, steps, epoch):
        # Gate retention on val loss: with dropout off, train loss can only fall.
        if self.val_windows is None or self.rank != 0:
            return False
        val = self.evaluate()
        if val is None:
            return False
        self.last_val = val
        if val < self.best_val - 1e-4:
            self.best_val = val
            self.best_step = steps
            self.since_best = 0
            self.save(self.last_train_loss, steps, val_loss=val)
            print(f"  step {steps:,} epoch {epoch} | val {val:.4f} "
                  f"| best {self.best_val:.4f} -> saved", flush=True)
        else:
            self.since_best += 1
            print(f"  step {steps:,} epoch {epoch} | val {val:.4f} "
                  f"| best {self.best_val:.4f} | stale {self.since_best}"
                  f"/{self.patience}", flush=True)
            if self.patience and self.since_best >= self.patience:
                print(f"  early stop: no val improvement for {self.patience} "
                      f"checks; best val {self.best_val:.4f} at step "
                      f"{self.best_step:,}", flush=True)
                self.stopped_early = True
                return True
        return False

    def fit(self):
        steps = self.start_step
        opt_steps = 0
        for epoch in range(1, self.epochs + 1):
            epoch_start = time.perf_counter()
            result = self._fit_epoch(epoch, opt_steps, steps)
            if result is None:
                return self.last_train_loss, steps
            total, batches, accum, steps, opt_steps = result
            if accum % self.grad_accum != 0:
                self._optimizer_step()
                opt_steps += 1
            avg = total / max(batches, 1)
            self._end_epoch(epoch, steps, avg, epoch_start)
        return self.last_train_loss, steps

    def _fit_epoch(self, epoch, opt_steps, steps):
        total = 0.0
        batches = 0
        accum = 0
        for inputs, targets in self.sampler:
            steps += 1
            self._schedule_lr(steps)
            if self.objective == "span":
                inputs, targets = self._corrupt_span(inputs)
            loss = self._train_step(inputs, targets)
            self._record_loss(loss)
            # Runaway guard: clip_grad_norm_ bounds gradient size but cannot un-NaN the
            # weights, and --keep-last would then save an all-NaN checkpoint.
            if self._diverged(self.last_train_loss, steps, epoch):
                self._abort_diverged(steps, epoch)
                return None
            total += loss.item()
            batches += 1
            accum += 1
            stepped = False
            if accum % self.grad_accum == 0:
                self._optimizer_step()
                opt_steps += 1
                stepped = True
            self._maybe_log(steps, epoch, loss.item())
            # --val-every counts optimizer steps; steps counts micro-batches.
            if self._stop_after_validate(steps, epoch, stepped, opt_steps):
                # last_train_loss, not self.best: best is only set at an epoch boundary.
                return None
        return total, batches, accum, steps, opt_steps

    def _corrupt_span(self, inputs):
        rows = [span_corrupt_pair(row, encoder=self.tokenizer)
                for row in inputs]
        return collate_span(rows, self.tokenizer)

    def _record_loss(self, loss):
        self.last_train_loss = loss.item() * self.grad_accum
        if math.isfinite(self.last_train_loss):
            self.recent.append(self.last_train_loss)

    def _optimizer_step(self):
        # Unscale before clipping so the 1.0 threshold is in real units; the
        # disabled fp32 path skips both calls (GradScaler no-ops them).
        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   GRAD_CLIP)
        if torch.isfinite(grad_norm):
            self.grad_norms.append(float(grad_norm))
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema is not None:
            self.ema.update(self._unwrap())

    def _abort_diverged(self, steps, epoch):
        self.diverged = True
        print(f"\nDIVERGED at step {steps:,} epoch {epoch}: train "
              f"loss {self.last_train_loss}. Aborting without "
              f"saving so the retained checkpoint stays usable.",
              flush=True)

    def _maybe_log(self, steps, epoch, step_loss):
        if self.log_every and steps % self.log_every == 0:
            self._log_step(steps, epoch, step_loss)

    def _stop_after_validate(self, steps, epoch, stepped, opt_steps):
        if stepped and opt_steps % self.val_every == 0:
            if self._maybe_validate(steps, epoch):
                return True
        return False

    def _diverged(self, loss, steps, epoch):
        # Non-finite is a hard stop; the spike check only counts, so one bad
        # minibatch does not end the run.
        if not math.isfinite(loss):
            return True
        # Needs a full window before it has a reference to compare against.
        if len(self.recent) < self.window_n:
            self.spike_count = 0
            return False
        cur = sum(self.recent) / len(self.recent)
        if not self.spike_factor or cur <= self.best_window * self.spike_factor:
            self.best_window = min(self.best_window, cur)
            self.spike_count = 0
            return False
        self.spike_count += 1
        if self.spike_count == 1:
            print(f"  step {steps:,} epoch {epoch} | loss spike "
                  f"windowed mean {cur:.4f} > {self.spike_factor:g}x best "
                  f"{self.best_window:.4f}", flush=True)
        return self.spike_count >= self.spike_patience

    def _unwrap(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _log_step(self, steps, epoch, step_loss):
        if self.rank != 0:
            return
        now = time.perf_counter()
        dt = now - self.window_start
        self.window_start = now
        tokens = (steps - self.last_logged_step) * self.sampler.batch_size * self.sampler.block
        tok_s = tokens / dt if dt > 0 else 0.0
        self.last_logged_step = steps
        elapsed = max(now - self.run_start, 1e-9)
        done = max(steps - self.start_step, 1)
        rate = done / elapsed
        remaining = max(self.total_steps - steps, 0) / rate if rate > 0 else 0.0
        eta = (f" | eta {remaining / 60:.0f}m" if remaining > 60
               else f" | eta {remaining:.0f}s")
        window_avg = sum(self.recent) / len(self.recent)
        cur_lr = self.optimizer.param_groups[0]["lr"]
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        gnorms = list(self.grad_norms)
        p90 = (sorted(gnorms)[min(len(gnorms) - 1, int(len(gnorms) * 0.9))]
               if gnorms else 0.0)
        clip_rate = (sum(1 for g in gnorms if g > GRAD_CLIP) / len(gnorms)
                     if gnorms else 0.0)
        print(
            f"  step {steps:,}/{self.total_steps:,} epoch {epoch}/{self.epochs} "
            f"| window {window_avg:.4f} | last {step_loss:.4f} "
            f"| gnorm {gnorms[-1] if gnorms else 0.0:.2f} p90 {p90:.2f} "
            f"clip {clip_rate:.0%} "
            f"| {tok_s:,.0f} tok/s | params {trainable:,} | lr {cur_lr:.2e}{eta}",
            flush=True,
        )

    def _end_epoch(self, epoch, steps, avg, epoch_start):
        epoch_dt = time.perf_counter() - epoch_start
        remaining_epochs = self.epochs - epoch
        eta = remaining_epochs * epoch_dt
        eta_txt = (f" eta ~{eta / 60:.0f}m" if eta > 60 else f" eta ~{eta:.0f}s")
        if self.rank == 0:
            print(f"  epoch {epoch}/{self.epochs}, steps {steps:,}: loss {avg:.4f}{eta_txt}",
                  flush=True)
        if self.objective == "lm" and epoch % max(1, self.sample_every_epoch) == 0:
            self._sample(epoch)
        self.save_if_best(avg, steps)

    def save_if_best(self, avg, steps):
        # Only used when there is no validation split; with one, _maybe_validate
        # owns which checkpoint gets retained.
        if self.rank != 0 or self.val_windows is not None or avg >= self.best:
            return
        self.best = avg
        self.save(avg, steps)
        if self.log_best:
            print(f"  best loss {avg:.4f} -> saved checkpoint at step {steps:,}",
                  flush=True)

    def _sample(self, epoch):
        if self.rank != 0:
            return
        model = self._unwrap()
        model.eval()
        user_name, her_name = speaker_names()
        seed = self.tokenizer.encode(f"{user_name}: hola como estas\n{her_name}: ")
        out = model.generate(
            torch.tensor([seed], device=self.device),
            max_new=32, temperature=0.8, top_k=20,
            stop_ids=self.tokenizer.stop_ids(),
        )
        model.train()
        print(f"  sample: {self.tokenizer.decode(out)}", flush=True)

    def epochs_done(self, steps):
        # Epochs completed, not requested: an interrupted run stored 20 after 1.9.
        per = max(1, self.batches_per_epoch)
        return int(steps) // per

    def save(self, loss, steps, val_loss=None):
        # loss and step are a matched pair: the train loss at exactly this step.
        # best_val/best_step travel separately so a resume restores them.
        state_dict = self._unwrap().state_dict()
        # Never persist non-finite weights over a good checkpoint.
        if not math.isfinite(loss) or any(
                not torch.isfinite(v).all() for v in state_dict.values()
                if v.is_floating_point()):
            print("  refusing to save: non-finite loss or weights; the "
                  "existing checkpoint is untouched", flush=True)
            return False
        torch.save(
            {"model": state_dict, "loss": float(loss), "cfg": self.cfg,
             "tokenizer": bpe_state(self.tokenizer), "step": int(steps),
             "epochs": self.epochs_done(steps),
             "epochs_target": int(self.epochs_target),
             "steps_per_epoch": int(max(1, self.batches_per_epoch)),
             "best": _finite_or_none(self.best),
             "val_loss": _finite_or_none(
                 val_loss if val_loss is not None else self.best_val),
             "best_val": _finite_or_none(self.best_val),
             "best_step": self.best_step,
             "optim": self.optimizer.state_dict(),
             "optim_groups": len(self.optimizer.param_groups),
             "scaler": self.scaler.state_dict(),
             "rng": _capture_rng(),
             "ema": self._ema_state(state_dict) if self.ema else None},
            self.ckpt,
        )
        return True
