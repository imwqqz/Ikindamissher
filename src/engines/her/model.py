import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# NF4 lookup table, normalized to [-1, 1] (QLoRA 3.1)
NF4_LEVELS = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
])
# Rotary position embedding on interleaved q/k pairs (RoFormer)
def _rotate_half(x):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int, base: float = 10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        pos = torch.arange(max_len).float()
        freq = torch.outer(pos, inv)  # (max_len, dim / 2)
        emb = freq.repeat_interleave(2, dim=-1)  # (max_len, dim) interleaved
        self.register_buffer("cos_cached", torch.cos(emb).unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin_cached", torch.sin(emb).unsqueeze(0).unsqueeze(0))

    def apply(self, q, k, start: int, length: int):
        # cast the fp32 cache down once or autocast promotes q*cos to fp32
        cos = self.cos_cached[:, :, start:start + length].to(q.dtype)
        sin = self.sin_cached[:, :, start:start + length].to(q.dtype)
        return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin

# RMSNorm: pre-norm, no mean subtraction, no bias (LLaMA family)
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight

# Block-wise NF4 quantization with double-quantized scales (QLoRA 3.1-3.2)
def nf4_quantize(weight: torch.Tensor, block: int = 64, scale_block: int = 256):
    weight = weight.float().contiguous().view(-1)
    n = weight.numel()
    pad = (block - n % block) % block
    if pad:
        weight = F.pad(weight, (0, pad))
    blocks = weight.view(-1, block)
    absmax = blocks.abs().amax(1).clamp(min=1e-8)
    # quant constant rounded to the nearest power of two (NF4)
    scale = 2.0 ** torch.round(torch.log2(absmax))
    norm = (blocks / scale.unsqueeze(1)).clamp(-1.0, 1.0)
    idx = (norm.unsqueeze(-1) - NF4_LEVELS).abs().argmin(-1)  # (nb, block) in 0..15
    idx = idx.view(-1)
    # pad right so the element count is even for the byte packing below
    parity = (idx[0::2] << 4) | idx[1::2]
    qweight = parity.to(torch.uint8)

    # double quantization of the scale constants (QLoRA 3.2):
    nb = scale.numel()
    pads = (scale_block - nb % scale_block) % scale_block
    sext = F.pad(scale, (0, pads)) if pads else scale
    sg = sext.view(-1, scale_block)
    sabs = sg.abs().amax(1, keepdim=True).clamp(min=1e-8)
    s_scale = 2.0 ** torch.round(torch.log2(sabs))      # (n_groups, 1)
    scode = torch.round(sg / s_scale * 127).to(torch.int8)
    return qweight, scode, s_scale

# Inverse of nf4_quantize (QLoRA 3.1, Algorithm 1)
def nf4_dequantize(qweight: torch.Tensor, scode: torch.Tensor,
                   s_scale: torch.Tensor, shape: Tuple[int, int],
                   block: int = 64, scale_block: int = 256):
    lohi = torch.stack([qweight.to(torch.int32) >> 4,
                        qweight.to(torch.int32) & 0xF], dim=-1).view(-1)
    nb = lohi.numel() // block
    scales = (scode.float() / 127.0 * s_scale).reshape(-1)[:nb].repeat_interleave(block)
    flat = NF4_LEVELS.to(qweight.device)[lohi.clamp(min=0)] * scales
    n = shape[0] * shape[1]
    return flat[:n].reshape(shape)

# Linear layer: frozen base, optional NF4 + LoRA adapter (QLoRA 3.4; LoRA 4.1)
class ParameterEfficientLinear(nn.Module):

    # output = W0 x + (alpha / r) * (x A) B

    def __init__(self, in_features, out_features, lora=False,
                 lora_r=8, lora_alpha=16, lora_dropout=0.0, quant="none",
                 block=64):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_r = lora_r
        self.alpha = lora_alpha
        self.quant = quant
        self.block = block

        if quant == "nf4":
            # base weights stored only as packed NF4 indices + DQ'd constants
            self.register_buffer("qweight", torch.zeros(0))
            self.register_buffer("scode", torch.zeros(0))
            self.register_buffer("s_scale", torch.zeros(0))
            w = torch.empty(out_features, in_features)
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            self.set_weight(w)
            self.qweight = self.qweight.contiguous()
        else:
            base = nn.Linear(in_features, out_features, bias=False)
            # freeze only when a frozen base is the point; unconditional freezing once stripped every attention gradient at lora=False
            if lora:
                base.weight.requires_grad_(False)
            self.base = base

        self.lora_enabled = lora
        if lora:
            self.lora_dropout = nn.Dropout(lora_dropout)
            self.lora_a = nn.Parameter(torch.empty(in_features, lora_r))
            self.lora_b = nn.Parameter(torch.zeros(lora_r, out_features))
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def set_weight(self, w: torch.Tensor):
        # (re)quantize fp32 weights into the NF4 buffers
        assert self.quant == "nf4"
        qweight, scode, s_scale = nf4_quantize(w, block=self.block)
        self.qweight = qweight.contiguous()
        self.scode = scode.contiguous()
        self.s_scale = s_scale.contiguous()

    def get_weight(self) -> torch.Tensor:
        if self.quant == "nf4":
            return nf4_dequantize(self.qweight, self.scode, self.s_scale,
                                  shape=(self.out_features, self.in_features),
                                  block=self.block)
        return self.base.weight

    def forward(self, x):
        w = self.get_weight()
        out = F.linear(x, w)
        if self.lora_enabled:
            # W = W0 + (alpha/r) * BA (LoRA 4.1)
            scale = self.alpha / max(self.lora_r, 1)
            out = out + scale * (self.lora_dropout(x) @ self.lora_a @ self.lora_b)
        return out

# Multi-head causal attention over scaled dot products (Attention 3.2)
class CausalSelfAttention(nn.Module):

    def __init__(self, cfg, rotary: RotaryEmbedding):
        super().__init__()
        d, nh = cfg["d_model"], cfg["n_heads"]
        assert d % nh == 0
        self.d = d
        self.nh = nh
        self.hd = d // nh
        self.rotary = rotary
        lora = cfg.get("lora", False)
        quant = cfg.get("quant", "none")
        args = dict(lora=lora, lora_r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
                    lora_dropout=cfg["lora_dropout"], quant=quant)
        self.wq = ParameterEfficientLinear(d, d, **args)
        self.wk = ParameterEfficientLinear(d, d, **args)
        self.wv = ParameterEfficientLinear(d, d, **args)
        self.wo = ParameterEfficientLinear(d, d, **args)
        self.dropout = nn.Dropout(cfg["dropout"])
        self._k = None
        self._v = None

    def reset_cache(self):
        self._k = self._v = None

    def forward(self, x, start_pos: int, use_cache: bool):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        q, k = self.rotary.apply(q, k, start_pos, T)
        if use_cache:
            if self._k is None:
                self._k = torch.empty(B, self.nh, 0, self.hd,
                                      dtype=k.dtype, device=k.device)
                self._v = torch.empty_like(self._k)
            self._k = torch.cat([self._k, k], dim=2)
            self._v = torch.cat([self._v, v], dim=2)
            k, v = self._k, self._v
            # causal only when the query covers the whole key range; unconditional False once poisoned the cache
            is_causal = q.shape[2] == k.shape[2]
        else:
            is_causal = True
        y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal,
                                           dropout_p=self.dropout.p if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(y)

# Position-wise FFN with GELU (T5 3.1; Attention 3.3)
class FeedForward(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        d, d_ff = cfg["d_model"], cfg["d_ff"]
        self.w1 = nn.Linear(d, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))

# Pre-norm residual block (Attention 3.1; RMSNorm, see above)
class Block(nn.Module):

    def __init__(self, cfg, rotary: RotaryEmbedding):
        super().__init__()
        self.norm1 = RMSNorm(cfg["d_model"])
        self.attn = CausalSelfAttention(cfg, rotary)
        self.norm2 = RMSNorm(cfg["d_model"])
        self.ffn = FeedForward(cfg)

    def forward(self, x, start_pos, use_cache):
        x = x + self.attn(self.norm1(x), start_pos, use_cache)
        x = x + self.ffn(self.norm2(x))
        return x

# Decoder-only transformer: scaled embeddings + tied head (Attention 3.4)
class GPT(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg["d_model"]
        self.max_len = cfg["max_len"]
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["d_model"])
        self.rotary = RotaryEmbedding(cfg["d_model"] // cfg["n_heads"], cfg["max_len"])
        self.drop = nn.Dropout(cfg["dropout"])
        self.blocks = nn.ModuleList(Block(cfg, self.rotary) for _ in range(cfg["n_layers"]))
        self.norm = RMSNorm(cfg["d_model"])
        # init std must offset the sqrt(d_model) forward scale, else step-0 loss ~128
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=cfg["d_model"] ** -0.5)
        self._pos = 0

    def reset_cache(self):
        for b in self.blocks:
            b.attn.reset_cache()
        self._pos = 0

    def forward(self, idx, use_cache=False):
        B, T = idx.shape
        start = self._pos if use_cache else 0
        x = self.tok_emb(idx) * math.sqrt(self.d_model)
        x = self.drop(x)
        for b in self.blocks:
            if self.cfg.get("gc", False) and self.training and not use_cache:
                # recompute per block to save memory
                x = torch.utils.checkpoint.checkpoint(
                    lambda t, s, u: b(t, s, u), x, start, use_cache, use_reentrant=False)
            else:
                x = b(x, start, use_cache)
        x = self.norm(x)
        logits = F.linear(x, self.tok_emb.weight)
        self._pos = start + T
        return logits

    def generate(self, idx, max_new=64, temperature=0.8, top_k=0, top_p=0.0,
                 repeat_penalty=1.0, stop_ids=()):
        # KV-cache sampling: prefill once, sample one token per forward pass
        if idx.shape[0] != 1:
            # sampling/stop-detection read row 0 only; fail clearly over a silent mismatch
            raise ValueError(
                f"generate supports batch size 1 (got {idx.shape[0]})")
        self.reset_cache()
        device = idx.device
        # read logits from the prefill position, else the last prompt token passes twice
        logits = self(idx, use_cache=True)[0, -1] / max(temperature, 1e-8)
        gen = []
        # RoPE allocates max_len positions, so a prompt at max_len leaves no room
        budget = max(0, self.max_len - idx.shape[1])
        for _ in range(min(max_new, budget)):
            if repeat_penalty != 1.0 and gen:
                # presence penalty on already-generated tokens (Holtzman 5)
                for token_id in set(gen):
                    token_logit = logits[token_id]
                    logits[token_id] = (token_logit / repeat_penalty
                                        if token_logit > 0
                                        else token_logit * repeat_penalty)
            probs = sample_probs(logits, top_k, top_p)
            next_token = int(torch.multinomial(probs, 1).item())
            if next_token in stop_ids:
                break
            gen.append(next_token)
            idx = torch.cat([idx, torch.tensor([[next_token]], device=device)], dim=1)
            # Advance the cache with the token just sampled.
            logits = self(idx[:, -1:], use_cache=True)[0, -1] / max(temperature, 1e-8)
        self.reset_cache()
        return gen

# Nucleus / top-k next-token distribution over raw logits (Holtzman 3.1)
def sample_probs(logits, top_k, top_p):
    probs = F.softmax(logits, dim=-1)
    if top_k:
        # keep only the top-k most probable tokens
        k = min(top_k, probs.numel())
        threshold = torch.topk(probs, k).values[-1]
        probs = probs.clone()
        probs[probs < threshold] = 0.0
    if top_p and top_p < 1.0:
        # nucleus: smallest set with cumulative prob >= top_p (Holtzman 3.1)
        sorted_probs, _ = probs.sort(descending=True)
        cum = sorted_probs.cumsum(0)
        n = int((cum <= top_p).sum()) + 1
        n = min(n, sorted_probs.numel())
        threshold = sorted_probs[n - 1]
        probs = probs.clone()
        probs[probs < threshold] = 0.0
    if probs.sum() <= 0:
        probs = F.softmax(logits, dim=-1)
    return probs / probs.sum()

# Frozen base + trainable LoRA adapters (LoRA 4.1; QLoRA 3.4)
def build_model(cfg):
    model = GPT(cfg)
    if cfg.get("lora", False):
        for name, p in model.named_parameters():
            if "lora_a" not in name and "lora_b" not in name:
                p.requires_grad_(False)
    return model


def trainable_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def total_count(model):
    return sum(p.numel() for p in model.parameters())
