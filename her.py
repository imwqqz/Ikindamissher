import argparse
import json
import math
import random
import re
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_DATA = Path(__file__).parent / "data" / "input" / "wa_out.txt"
DEFAULT_CKPT = Path(__file__).parent / "data" / "her_model.pt"
DEFAULT_DESC = Path(__file__).parent / "data" / "input" / "description.txt"
DEFAULT_BPE = Path(__file__).parent / "data" / "input" / "bpe.json"

USER_SPEAKER = "wqqz"   # canonical name for the user in turns/prompts
HER_SPEAKER = ""    # canonical name for the persona in turns

# GPT-2 style pretokenizer (T5, section 2.1 https://arxiv.org/abs/2005.14165)
BPE_PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z\u00C0-\u1FFF]+| ?\d+| ?[^\sA-Za-z0-9\u00C0-\u1FFF]+|\s+(?!\S)|\s+"""
)

# NF4 lookup table (normalized to [-1, 1]) (https://arxiv.org/abs/2305.18290)
NF4_LEVELS = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
])


# Byte-level BPE encoder, trained on the corpus. (T5, section 3.2 https://arxiv.org/abs/2005.14165).
class BytePairEncoder:

    def __init__(self, n_sentinels=100):
        self.byte_encoder = {i: bytes([i]) for i in range(256)}
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        # token per id: 0..255 bytes, 256 = never-merged word boundary
        self.tokens = [bytes([i]) for i in range(256)] + [b""]
        self.merges: dict = {}                          # (i, j) -> k
        self.n_merges = 0
        self.n_sentinels = n_sentinels

    def train(self, text: str, vocab_size: int):
        pieces = re.findall(BPE_PAT, text)
        byts = [list(p.encode("utf-8")) for p in pieces]
        ids = []
        for b in byts:
            ids.extend(b)
            ids.append(256)  # word-boundary id, kept as a real (empty) token
        ids = ids[: -1] if len(ids) > 1 else ids  # drop trailing boundary marker
        ids = torch.tensor(ids, dtype=torch.long)  # tensor-native merging (no per-merge list conversion)

        target = vocab_size - 256 - 1 - self.n_sentinels
        self.n_merges = 0
        i = 0
        t0 = time.perf_counter()
        while self.n_merges < target:
            (a, b), count = self._best_pair(ids)
            if count < 2:
                break
            k = 257 + self.n_merges
            self.tokens.append(self.tokens[a] + self.tokens[b])
            self.merges[(a, b)] = k
            ids = self._merge_vec(ids, a, b, k)
            self.n_merges += 1
            i += 1
            if i % 500 == 0:
                now = time.perf_counter()
                rate = i / max(now - t0, 1e-9)
                eta = max(target - self.n_merges, 0) / rate
                eta_txt = (f" eta ~{eta / 60:.1f}m" if eta > 60
                           else f" eta ~{eta:.0f}s")
                print(f"  bpe merges {self.n_merges:,}/{target:,}{eta_txt}",
                      flush=True)
        print(f"  bpe done: {self.n_merges:,} merges, vocab {self.vocab_size:,}",
              flush=True)

    @staticmethod
    def _best_pair(ids: torch.Tensor):
        # Vectorized argmax over adjacent pairs:
        #   stats[(a,b)] = sum_t 1[a_t==a and a_{t+1}==b],
        # computed by packing each pair into a scalar key `a*BASE + b`,
        # sorting, and run-length encoding (O(n log n), no per-merge Python loop).
        # Ties on count are broken by first appearance, matching the original
        # pure-Python Counter max() insertion-order semantics.
        n = ids.numel() - 1
        if n <= 0:
            return (0, 0), 0
        BASE = 4096  # largest possible id is 256 + n_merges < 4096 (bytes 0-255, boundary 256, merges 257+)
        # TODO: REVIEW: pair key `a*BASE + b` collides when a/256+n_merges >= BASE (i.e. --vocab-size > 4096)
        pairs = ids[:-1] * BASE + ids[1:]
        # stable sort keeps first-occurrence order for equal pairs, so merges
        # are reproducible across runs (BPE, Neural Machine Translation of
        # Rare Words with Subword Units, section 3 https://arxiv.org/abs/1508.07909)
        pairs_sorted, order = torch.sort(pairs, stable=True)
        keys, counts = torch.unique_consecutive(pairs_sorted, return_counts=True)
        maxc = int(counts.max())
        cand = torch.nonzero(counts == maxc).reshape(-1)
        # first_occurrence = original position of each run's first element
        run_start = torch.cat([torch.zeros(1, dtype=torch.long),
                               counts.cumsum(0)[:-1]])
        first_pos = order[run_start]
        best = cand[int(torch.argmin(first_pos[cand]))]
        p = int(keys[best])
        return (p // BASE, p % BASE), int(counts[best])

    @staticmethod
    def _merge(ids, a, b, k):
        # Greedy left-to-right scan; used on short per-wordpiece lists in encode()
        out, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
                out.append(k)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    @staticmethod
    def _merge_vec(ids: torch.Tensor, a, b, k) -> torch.Tensor:
        # Replace every occurrence of the pair (a,b) with token k in one pass.
        # Greedy left-to-right (no overlapping merges): for runs of adjacent
        # candidate starts, take every other one (`starts[::2]` within a run).
        n = ids.numel()
        is_start = (ids[:-1] == a) & (ids[1:] == b)
        if not bool(is_start.any()):
            return ids
        starts = is_start.nonzero().reshape(-1)
        if starts.numel() >= 2:
            breaks = (starts[1:] - starts[:-1]) > 1
            grp = torch.cat([torch.zeros(1, dtype=torch.long),
                             breaks.cumsum(0)])
            sizes = torch.bincount(grp)
            firsts = starts[torch.cat([torch.tensor([True]), breaks])]
            within = starts - torch.repeat_interleave(firsts, sizes)
            accepted = starts[within % 2 == 0]
        else:
            accepted = starts
        # keep the accepted start (now holding token k); drop only the consumed
        # successor (accepted+1) since the merged pair shrinks the output
        removed = torch.zeros(n, dtype=torch.bool)
        removed[accepted + 1] = True
        vals = ids.clone()
        vals[accepted] = k
        return vals[~removed]

    def encode(self, text: str) -> List[int]:
        pieces = re.findall(BPE_PAT, text)
        ids_all = []
        for p in pieces:
            ids = list(p.encode("utf-8"))
            while len(ids) >= 2:
                stats = Counter(zip(ids, ids[1:]))
                pair = None
                best = None
                for (a, b), rank in stats.items():
                    if (a, b) in self.merges:
                        if best is None or rank < best:
                            best = rank
                            pair = (a, b)
                if pair is None:
                    break
                ids = self._merge(ids, pair[0], pair[1], self.merges[pair])
            ids_all.extend(ids)
        return ids_all

    def decode(self, ids: List[int]) -> str:
        raw = b"".join(self.tokens[i] if i < len(self.tokens) else b"" for i in ids)
        return raw.decode("utf-8", errors="replace")

    @property
    def sentinel_base(self) -> int:
        return 257 + self.n_merges

    @property
    def vocab_size(self) -> int:
        return self.sentinel_base + self.n_sentinels

    def sentinel_id(self, i: int) -> int:
        assert 0 <= i < self.n_sentinels, "not enough sentinels reserved"
        return self.sentinel_base + i

    def stop_ids(self, extra=()):
        nl = self.encode("\n")
        # TODO: REVIEW: halting on the first newline truncates replies at the
        # first line break (newline token boundary ids only, so emoji remain).
        stops = {nl[0]} if nl else set()
        stops.update(self.sentinel_id(i) for i in range(self.n_sentinels))
        stops.update(extra)
        return stops

    def save(self, path: Path):
        payload = {
            "merges": [[int(a), int(b)] for (a, b) in self.merges],
            "tokens_hex": [t.hex() for t in self.tokens],
            "n_sentinels": self.n_sentinels,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BytePairEncoder":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        enc = cls(n_sentinels=payload["n_sentinels"])
        enc.tokens = [bytes.fromhex(t) for t in payload["tokens_hex"]]
        enc.merges = {tuple(pair): 257 + i for i, pair in enumerate(payload["merges"])}
        enc.n_merges = len(enc.merges)
        return enc


# 2. Convert a WhatsApp export into `speaker: message` turns. (T5, section 2.1 https://arxiv.org/abs/2005.14165).
WA_LINE = re.compile(
    r"^\d{1,2}/\d{1,2}/\d{2,4},?.*?- ([^:]+): (.*)$"
)
WA_NAME_BOOTSTRAP = re.compile(r"^[^:\n]+: ")


def parse_wa(text: str, speakers=(USER_SPEAKER, HER_SPEAKER), aliases=None):
    # Normalize the exporter's speaker names to canonical turns. (T5, section 2.1 https://arxiv.org/abs/2005.14165)
    speakers = tuple(s.lower() for s in speakers)
    aliases = {k.lower(): v.lower() for k, v in (aliases or {}).items()}
    lines = []
    for line in text.splitlines():
        match = WA_LINE.match(line)
        if not match:
            continue
        name, msg = match.group(1).strip(), match.group(2).strip()
        name_lower = aliases.get(name.lower(), name.lower())
        if re.match(r"(<media omitted>|media omitted|you deleted this message|this message was deleted)", msg, re.I):
            continue
        if not msg:
            continue
        if name_lower not in speakers:
            continue
        lines.append(f"{name_lower}: {msg}")
    return "\n".join(lines) + "\n"


def detect_user(text: str, label: str = HER_SPEAKER):
    # Pick the most frequent non-her speaker so any WhatsApp contact works. (T5, section 2.1 https://arxiv.org/abs/2005.14165)
    counts = Counter()
    for line in text.splitlines():
        match = WA_LINE.match(line)
        if match:
            name = match.group(1).strip().lower()
            if name and name != label.lower():
                counts[name] += 1
    return counts.most_common(1)[0][0] if counts else USER_SPEAKER


def persona_lines(text: str):
    # Turn the persona description into `name: line` statements, treating the persona as one more text-to-text turn (T5, section 2.1 https://arxiv.org/abs/2005.14165).
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if WA_NAME_BOOTSTRAP.match(line):
            lines.append(line)
        elif len(line) <= 400:
            lines.append("her: " + line)
        else:
            for chunk in re.split(r"(?<=[.!?])\s+", line):
                if chunk.strip():
                    lines.append("her: " + chunk.strip())
    return lines

# (T5, section 2.1 https://arxiv.org/abs/2005.14165).
def build_corpus(data_path: Path, desc_path: Path, label: str = HER_SPEAKER,
                 user: Optional[str] = None):
    raw = data_path.read_text(encoding="utf-8")
    if user is None:
        user = detect_user(raw, label)
    aliases = {}
    if user.lower() != USER_SPEAKER:
        aliases[user.lower()] = USER_SPEAKER
    if label.lower() != HER_SPEAKER:
        aliases[label.lower()] = HER_SPEAKER
    corpus = parse_wa(raw, speakers=(USER_SPEAKER, HER_SPEAKER), aliases=aliases or None)
    corpus = corpus.replace(f"\n{label}: ", f"\n{HER_SPEAKER}: ")
    desc = ""
    if desc_path.exists():
        desc = "\n".join(persona_lines(desc_path.read_text(encoding="utf-8")))
    if desc:
        corpus = corpus.rstrip("\n") + "\n\n" + desc + "\n"
    return corpus


# 3. RoPE, Attention Is All You Need, section 3.5 (positional encoding lineage) https://arxiv.org/abs/1706.03762
def _rotate_half(x):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)

# Rope, Params-Free Attention Is All You Need, sections 3.4-3.5 https://arxiv.org/abs/1706.03762
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
        cos = self.cos_cached[:, :, start:start + length]
        sin = self.sin_cached[:, :, start:start + length]
        return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin

# No-bias normalization used in pre-norm blocks (T5, section 2.1 https://arxiv.org/abs/2005.14165).
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight

    #Block-wise 4-bit NormalFloat quantization with double quantization of the scale constants (QLoRA sections 3.1-3.2 https://arxiv.org/abs/2305.18290).
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
    # TODO: REVIEW: pair packing assumes an even element count; safe while n is
    # padded to a multiple of block (64), breaks for odd block sizes.
    parity = (idx[0::2] << 4) | idx[1::2]
    qweight = parity.to(torch.uint8)

    # double quantization (QLoRA, section 3.2 https://arxiv.org/abs/2305.18290):
    nb = scale.numel()
    pads = (scale_block - nb % scale_block) % scale_block
    sext = F.pad(scale, (0, pads)) if pads else scale
    sg = sext.view(-1, scale_block)
    sabs = sg.abs().amax(1, keepdim=True).clamp(min=1e-8)
    s_scale = 2.0 ** torch.round(torch.log2(sabs))      # (n_groups, 1)
    scode = torch.round(sg / s_scale * 127).to(torch.int8)
    return qweight, scode, s_scale


#Inverse of nf4_quantize (QLoRA, section 3.1, Algorithm 1).
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


# Linear layer: frozen base, optional NF4 + LoRA adapter (QLoRA, section 3.4 https://arxiv.org/abs/2305.18290; LoRA, sections 4.1-4.2 https://arxiv.org/abs/2106.09685)
class ParameterEfficientLinear(nn.Module):

    # output = W0 x + (alpha / r) * (x A) B

    def __init__(self, in_features, out_features, bias=False, lora=False,
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
            base = nn.Linear(in_features, out_features, bias=bias)
            base.weight.requires_grad_(False)
            if base.bias is not None:
                base.bias.requires_grad_(False)
            self.base = base

        self.lora_enabled = lora
        if lora:
            self.lora_dropout = nn.Dropout(lora_dropout)
            self.lora_a = nn.Parameter(torch.empty(in_features, lora_r))
            self.lora_b = nn.Parameter(torch.zeros(lora_r, out_features))
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def set_weight(self, w: torch.Tensor):
        # (Re)quantize fp32 weights into the NF4 buffers (QLoRA, section 3.1 https://arxiv.org/abs/2305.18290)
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
            # W = W0 + (alpha / r) * BA (LoRA, section 4.1
            # https://arxiv.org/abs/2106.09685)
            scale = self.alpha / max(self.lora_r, 1)
            out = out + scale * (self.lora_dropout(x) @ self.lora_a @ self.lora_b)
        return out


# Multi-head causal attention over scaled dot products (Attention Is All You Need, sections 3.2.1-3.2.2 https://arxiv.org/abs/1706.03762)
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
        self.wq = ParameterEfficientLinear(d, d, bias=False, **args)
        self.wk = ParameterEfficientLinear(d, d, bias=False, **args)
        self.wv = ParameterEfficientLinear(d, d, bias=False, **args)
        self.wo = ParameterEfficientLinear(d, d, bias=False, **args)
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
            is_causal = False
        else:
            is_causal = True
        y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal,
                                           dropout_p=self.dropout.p if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(y)


# Position-wise FFN with GELU (T5, section 3.1 https://arxiv.org/abs/2005.14165; 4x d_ff, Attention Is All You Need, section 3.3 https://arxiv.org/abs/1706.03762)
class FeedForward(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        d, d_ff = cfg["d_model"], cfg["d_ff"]
        self.w1 = nn.Linear(d, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))


# Pre-norm residual transformer block (T5, section 3.1 https://arxiv.org/abs/2005.14165; residual + LayerNorm, Attention Is All You Need, section 3.1 https://arxiv.org/abs/1706.03762)
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


# Decoder-only transformer: sqrt(d_model)-scaled embeddings + tied head (Attention Is All You Need, section 3.4 https://arxiv.org/abs/1706.03762)
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
                # gradient checkpointing: trade recompute for memory
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
        # KV-cache autoregressive sampling: prefill the cache with the whole
        # prompt first, then sample one token at a time; without the prefill
        # the first forward would only ever see the prompt's last token and
        # the model could not condition on the question (Attention Is All You Need, section 5 https://arxiv.org/abs/1706.03762)
        self.reset_cache()
        device = idx.device
        self(idx, use_cache=True)
        gen = []
        for _ in range(max_new):
            logits = self(idx[:, -1:], use_cache=True)[0, -1] / max(temperature, 1e-8)
            if repeat_penalty != 1.0 and gen:
                # Penalize already-generated tokens (presence penalty) (The Curious Case of Neural Text Degeneration, section 5 https://arxiv.org/abs/1904.09751)
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
        self.reset_cache()
        return gen


# Nucleus/top-k sampling distribution over the next token from raw logits (The Curious Case of Neural Text Degeneration, section 3.1 https://arxiv.org/abs/1904.09751)
def sample_probs(logits, top_k, top_p):
    probs = F.softmax(logits, dim=-1)
    if top_k:
        # keep only the top-k most probable tokens
        k = min(top_k, probs.numel())
        threshold = torch.topk(probs, k).values[-1]
        probs = probs.clone()
        probs[probs < threshold] = 0.0
    if top_p and top_p < 1.0:
        # nucleus: smallest set with cumulative prob >= top_p (The Curious Case of Neural Text Degeneration, section 3.1 https://arxiv.org/abs/1904.09751)
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


# Causal LM loss with label smoothing (Attention Is All You Need, section 5.4 https://arxiv.org/abs/1706.03762)
def lm_loss(logits, targets, label_smoothing):
    flat = logits[:, :-1].reshape(-1, logits.size(-1))
    tgt = targets[:, 1:].reshape(-1)
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
    spans = []  # (start, end) masked runs, merged across small gaps
    i = 0
    while i < mask.numel():
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < mask.numel() and mask[j]:
            j += 1
        if spans and i - spans[-1][1] <= span_gap:
            spans[-1] = (spans[-1][0], j)
        else:
            spans.append((i, j))
        i = j
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
    """Pad corrupted sequences + labels so logits[:, :-1] aligns with labels."""
    max_inp = max(r[0].numel() for r in rows)
    # TODO: REVIEW: labels longer than max_inp-1 get truncated with ll[:max_lab],
    # which may drop sentinel/target tail tokens for short inputs.
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

    def __init__(self, ids, block, batch_size, device, rank=0, world=1):
        self.block = block
        self.batch_size = batch_size
        self.device = device
        self.rank = rank
        self.world = world
        self.ids = torch.tensor(ids, dtype=torch.long, device=device)
        self.num_windows = max(1, len(self.ids) - block)
        self.offsets = torch.arange(block, dtype=torch.long, device=device)

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
            # unshifted window: lm_loss applies the one-ahead shift itself,
            # so yielding ids[idx + 1] here would double-shift and train the
            # model to predict two tokens ahead (attention is causal) (Attention Is All You Need, section 5.1 https://arxiv.org/abs/1706.03762)
            yield self.ids[idx], self.ids[idx]

class Trainer:
    def __init__(self, args, cfg, model, ids, device, tokenizer, ckpt,
                 start_step=0, rank=0, world=1):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.tokenizer = tokenizer
        self.ckpt = Path(ckpt)
        self.rank = rank
        self.world = world
        self.epochs = args.epochs
        self.label_smoothing = args.label_smoothing
        self.log_every = args.log_every
        self.log_best = args.log_best
        self.grad_accum = max(1, args.grad_accum)
        self.objective = args.objective
        self.start_step = start_step
        self.precision = getattr(args, "precision_dtype", None)
        self.sample_every_epoch = 4

        self.sampler = BatchSampler(ids, args.block, args.batch, device,
                                    rank=rank, world=world)
        self.batches_per_epoch = self.sampler.batches_per_epoch()
        self.total_steps = start_step + self.epochs * self.batches_per_epoch

        # Adam with beta2 tuned for transformer training (Attention Is All You Need, section 5.3 https://arxiv.org/abs/1706.03762)
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable, lr=args.lr, betas=(0.9, 0.95), eps=1e-9,
            weight_decay=1e-2,
        )
        self.warmup_steps = args.warmup_steps
        self.schedule = getattr(args, "schedule", "invsqrt")
        self.lr = args.lr
        self.base_lr = (args.lr * math.sqrt(args.warmup_steps)
                        if args.warmup_steps > 0 else args.lr)

        self.best = float("inf")
        self.recent = deque(maxlen=max(args.log_every, 1))
        self.run_start = time.perf_counter()
        self.window_start = self.run_start
        self.last_logged_step = start_step

    def _schedule_lr(self, step):
        # inverse-square-root / constant / cosine LR schedules after a linear
        # warmup (Attention Is All You Need, section 5.3 https://arxiv.org/abs/1706.03762;
        # T5, section 3.4 https://arxiv.org/abs/2005.14165;
        # SGDR, section 2.1 https://arxiv.org/abs/1608.03983)
        if self.warmup_steps <= 0:
            return
        # linear warmup ramp applies to every schedule (Attention Is All You Need, section 5.3 https://arxiv.org/abs/1706.03762)
        if step < self.warmup_steps:
            lr = self.lr * step / self.warmup_steps
        elif self.schedule == "cosine":
            # lr(t) = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(pi*t/T)),
            # annealed to ~0 over the remaining steps (SGDR, section 2.1 https://arxiv.org/abs/1608.03983):
            # <code> t = step - warmup, T = total_steps - warmup </code>
            t = step - self.warmup_steps
            T = max(self.total_steps - self.warmup_steps, 1)
            lr = self.lr * 0.5 * (1.0 + math.cos(math.pi * t / T))
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
        loss.backward()
        return loss

    def fit(self):
        steps = self.start_step
        for epoch in range(1, self.epochs + 1):
            epoch_start = time.perf_counter()
            total = 0.0
            batches = 0
            accum = 0
            for inputs, targets in self.sampler:
                steps += 1
                self._schedule_lr(steps)
                if self.objective == "span":
                    rows = [span_corrupt_pair(row, encoder=self.tokenizer)
                            for row in inputs]
                    inputs, targets = collate_span(rows, self.tokenizer)
                loss = self._train_step(inputs, targets)
                total += loss.item()
                batches += 1
                accum += 1
                if accum % self.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                if self.log_every and steps % self.log_every == 0:
                    self._log_step(steps, epoch, loss.item())
            if accum % self.grad_accum != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            avg = total / max(batches, 1)
            self._end_epoch(epoch, steps, avg, epoch_start)
        return self.best, steps

    def _unwrap(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _log_step(self, steps, epoch, step_loss):
        if self.rank != 0:
            return
        self.recent.append(step_loss)
        now = time.perf_counter()
        dt = now - self.window_start
        self.window_start = now
        s = max(steps, 1)
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
        print(
            f"  step {steps:,}/{self.total_steps:,} epoch {epoch}/{self.epochs} "
            f"| window {window_avg:.4f} | last {step_loss:.4f} "
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
        if self.rank != 0 or avg >= self.best:
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
        seed = self.tokenizer.encode(f"{USER_SPEAKER}: hola como estas\n{HER_SPEAKER}: ")
        out = model.generate(
            torch.tensor([seed], device=self.device),
            max_new=32, temperature=0.8, top_k=20,
            stop_ids=self.tokenizer.stop_ids(),
        )
        model.train()
        print(f"  sample: {self.tokenizer.decode(out)}", flush=True)

    def save(self, loss, steps):
        state_dict = self._unwrap().state_dict()
        torch.save(
            {"model": state_dict, "loss": loss, "cfg": self.cfg,
             "tokenizer": bpe_state(self.tokenizer), "step": steps,
             "epochs": self.epochs},
            self.ckpt,
        )


def bpe_state(enc: BytePairEncoder):
    return {"merges": [[int(a), int(b)] for (a, b) in enc.merges],
            "tokens_hex": [t.hex() for t in enc.tokens],
            "n_sentinels": enc.n_sentinels}


def bpe_from_state(state):
    enc = BytePairEncoder(n_sentinels=state["n_sentinels"])
    enc.tokens = [bytes.fromhex(t) for t in state["tokens_hex"]]
    enc.merges = {tuple(pair): 257 + i for i, pair in enumerate(state["merges"])}
    enc.n_merges = len(enc.merges)
    return enc


SIZE_PRESETS = {
    "nano": dict(d_model=128, n_layers=4, n_heads=4, d_ff=512, max_len=128),
    "mini": dict(d_model=256, n_layers=6, n_heads=8, d_ff=1024, max_len=256),
    "small": dict(d_model=384, n_layers=8, n_heads=8, d_ff=1536, max_len=512),
}


def build_cfg(args, tokenizer):
    size = SIZE_PRESETS[args.size]
    cfg = {
        "vocab_size": tokenizer.vocab_size,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "d_ff": args.d_ff,
        "max_len": args.block,
        "dropout": args.dropout,
        "objective": args.objective,
        "lora": args.lora or args.qlora,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "quant": "nf4" if args.qlora else "none",
        "gc": getattr(args, "gc", False),
        "tokenizer_arch": "bpe",
    }
    if args.size != "none":
        for k, v in size.items():
            cfg[k] = v
    return cfg


# Frozen base + trainable LoRA adapters (LoRA, section 4.1 https://arxiv.org/abs/2106.09685; QLoRA, section 3.4 https://arxiv.org/abs/2305.18290)
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

def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="her: memorize the chat into a scalable decoder-only transformer",
    )
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--description", default=str(DEFAULT_DESC))
    ap.add_argument("--user", default=None,
                    help="your speaker name in the export (default: auto-detect)")
    ap.add_argument("--tokenizer-file", default=str(DEFAULT_BPE))
    ap.add_argument("--fresh", action="store_true",
                    help="retrain from scratch, ignore checkpoint and BPE cache")
    ap.add_argument("--size", default="nano",
                    choices=["none", "nano", "mini", "small"])

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dropout", type=float, default=0.1,
                    help="dropout probability (default 0.1; 0 for memorization)")
    ap.add_argument("--schedule", default="invsqrt",
                    choices=["invsqrt", "constant", "cosine"],
                    help="LR schedule after warmup: inverse-square-root decay "
                         "(Attention Is All You Need, section 5.3 "
                         "https://arxiv.org/abs/1706.03762), flat constant "
                         "(T5, section 3.4 https://arxiv.org/abs/2005.14165), "
                         "or cosine anneal to ~0 (SGDR, section 2.1 "
                         "https://arxiv.org/abs/1608.03983)")
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="label smoothing eps_ls (Attention Is All You Need, "
                         "section 5.4 https://arxiv.org/abs/1706.03762)")
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="micro-batches per optimizer step (T5, section 3.4 "
                         "https://arxiv.org/abs/2005.14165)")
    ap.add_argument("--precision", default="fp32", choices=["fp32", "bf16", "fp16"])

    ap.add_argument("--objective", default="lm", choices=["lm", "span"])
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--qlora", action="store_true",
                    help="NF4-quantize frozen base + train LoRA adapters "
                         "(QLoRA, section 3.4 https://arxiv.org/abs/2305.18290; "
                         "LoRA, section 4.1 https://arxiv.org/abs/2106.09685)")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.0)

    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=512)
    ap.add_argument("--vocab-size", type=int, default=4096)

    ap.add_argument("--gc", action="store_true", help="gradient checkpointing")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model")
    ap.add_argument("--ddp", action="store_true", help="data-parallel (torchrun)")
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--log-best", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args(argv)


def load_corpus(args):
    data = Path(args.data)
    if not data.exists():
        print(f"training data not found at {data}")
        raise SystemExit(1)
    return build_corpus(data, Path(args.description), user=args.user)


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
            print("[Warning] CUDA is not available. Exiting.")
            sys.exit(1)
    if rank == 0:
        print(f"device: {device} (world {world})")
    return device, rank, world, local


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


def reconcile_epochs(args, stored_epochs, rank):
    # Which epoch count wins on resume: explicit --epochs, else the stored one.
    if args.epochs_explicit:
        _print_resume_override(args, stored_epochs, rank)
        return
    args.epochs = stored_epochs or args.epochs
    if stored_epochs and rank == 0:
        print(f"resume: continuing for {stored_epochs} checkpoint epochs "
              f"(pass --epochs to override)", flush=True)


def resolve_model(args, state, tok, rank):
    # Reuse the checkpoint's architecture when possible; otherwise build from flags.
    if state is None:
        cfg = build_cfg(args, tok)
        return cfg, build_model(cfg)
    cfg = state["cfg"]
    args.objective = cfg["objective"]
    reconcile_epochs(args, cfg.get("epochs"), rank)
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


def maybe_compile(args, model, rank):
    # torch.compile with eager fallback on Windows (no Triton kernels).
    if not args.compile:
        return model
    if sys.platform == "win32":
        # Triton is not packaged for Windows, so Inductor cannot build
        # kernels; degrade to eager instead of aborting on the first forward
        # pass (PyTorch 2 compile docs, https://pytorch.org/docs/stable/torch.compiler.html;
        # torch._dynamo.config.suppress_errors)
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


def run_training(args, cfg, model, ids, device, tok, state, rank, world, local):
    # Optimizer loop + final checkpoint/summary, then teardown the process group.
    trainer = Trainer(args, cfg, model, ids, device, tok, args.ckpt,
                      start_step=state["step"] if state else 0,
                      rank=rank, world=world)
    if rank == 0:
        print(f"training for {args.epochs} epochs (~{trainer.total_steps:,} steps)")
    t0 = time.perf_counter()
    final_loss, steps_done = trainer.fit()
    elapsed = time.perf_counter() - t0

    if rank == 0:
        trainer.save(final_loss, steps_done)
        print(f"saved checkpoint: {args.ckpt} (loss {final_loss:.4f}, {steps_done:,} steps)")
        print_summary(args, cfg, tok, len(ids), trainable_count(model),
                      total_count(model), final_loss, steps_done,
                      state["step"] if state else 0, device, elapsed)
    if args.ddp:
        import torch.distributed as dist
        dist.destroy_process_group()


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    args = parse_args(argv)
    # was --epochs passed on the CLI? resume must not silently override it
    args.epochs_explicit = "--epochs" in (sys.argv[1:] if argv is None else argv)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device, rank, world, local = init_process(args)
    state = load_checkpoint(args, rank, world)
    corpus = load_corpus(args)
    tok = load_tokenizer(args, corpus, state, rank)
    cfg, model = resolve_model(args, state, tok, rank)

    n_train = trainable_count(model)
    n_total = total_count(model)
    report_setup(args, cfg, rank, n_train, n_total)

    model = maybe_compile(args, model, rank)
    args.precision_dtype = apply_precision(args, rank)

    model = model.to(device)
    if args.ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local])

    ids = tok.encode(corpus)
    if rank == 0:
        print(f"corpus: {len(ids):,} tokens, vocab {tok.vocab_size:,}")

    run_training(args, cfg, model, ids, device, tok, state, rank, world, local)


def print_summary(args, cfg, tok, n_tokens, n_train, n_total, final_loss,
                  steps_done, start_step, device, elapsed):
    print("----------------------------------------")
    print("training summary")
    print("----------------------------------------")
    print(f"loss:            {final_loss:.4f}")
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
    print(f"warmup:          {args.warmup_steps}")
    print(f"schedule:        {args.schedule}")
    print(f"dropout:         {cfg['dropout']}")
    print(f"label smoothing: {args.label_smoothing}")
    print(f"device:          {device}")
    print(f"time:            {elapsed / 60:.1f} min")
    print(f"checkpoint:      {args.ckpt}")


if __name__ == "__main__":
    main()