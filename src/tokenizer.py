import json
import re
import time
from pathlib import Path
from typing import List, Tuple

import torch

# GPT-2 style pretokenizer (T5, section 2.1 https://arxiv.org/abs/2005.14165)
BPE_PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z\u00C0-\u1FFF]+| ?\d+| ?[^\sA-Za-z0-9\u00C0-\u1FFF]+|\s+(?!\S)|\s+"""
)
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
        # stats[(a,b)] = sum_t 1[a_t==a and a_(t+1)==b], via key = a*BASE+b, sort, run-length.
        # O(n log n); ties keep first-occurrence order.
        n = ids.numel() - 1
        if n <= 0:
            return (0, 0), 0
        # pair key a*base + b is dense (no collision) and sort-stable for any
        # base > max(b). 4096 keeps ids < 4096 byte-identical to the original;
        # a larger --vocab-size grows the base instead of colliding.
        base = max(4096, int(ids.max()) + 1)
        pairs = ids[:-1] * base + ids[1:]
        # Stable sort keeps merges reproducible across runs
        # (BPE, section 3 https://arxiv.org/abs/1508.07909)
        pairs_sorted, order = torch.sort(pairs, stable=True)
        keys, counts = torch.unique_consecutive(pairs_sorted, return_counts=True)
        # The word-boundary id is never emitted by encode(), so a merge through
        # it could never be applied: keep it out of the ranking entirely.
        counts = counts.clone()
        boundary = (keys // base == 256) | (keys % base == 256)
        counts[boundary] = 0
        maxc = int(counts.max())
        cand = torch.nonzero(counts == maxc).reshape(-1)
        # first_occurrence = original position of each run's first element
        run_start = torch.cat([torch.zeros(1, dtype=torch.long),
                               counts.cumsum(0)[:-1]])
        first_pos = order[run_start]
        best = cand[int(torch.argmin(first_pos[cand]))]
        packed = int(keys[best])
        return (packed // base, packed % base), int(counts[best])

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
        # Greedy left-to-right, no overlapping merges: every other candidate start in a run.
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
        # Keep the accepted start, drop only the consumed successor.
        removed = torch.zeros(n, dtype=torch.bool)
        removed[accepted + 1] = True
        vals = ids.clone()
        vals[accepted] = k
        return vals[~removed]

    def _encode_piece(self, p: str) -> List[int]:
        # Canonical BPE: repeatedly apply the merge with the lowest rank (the
        # id assigned when it was learned). Choosing by frequency here instead
        # applied merges in an order training never used.
        ids = list(p.encode("utf-8"))
        while len(ids) >= 2:
            best = None
            for a, b in zip(ids, ids[1:]):
                rank = self.merges.get((a, b))
                if rank is not None and (best is None or rank < best[1]):
                    best = ((a, b), rank)
            if best is None:
                break
            ids = self._merge(ids, best[0][0], best[0][1], best[1])
        return ids

    def encode(self, text: str) -> List[int]:
        ids_all = []
        for p in re.findall(BPE_PAT, text):
            ids_all.extend(self._encode_piece(p))
        return ids_all

    def encode_masked(self, text: str, char_mask) -> Tuple[List[int], List[bool]]:
        """ids plus a per-token copy of char_mask.

        A character is not a token, so the label is broadcast to every id that the
        piece produced: result is exactly len(ids). The label comes from any scored
        character in the piece, not its first: BPE_PAT lets a piece carry a leading
        space (" pq"), and that space is the last char of the "her: " prefix, so
        reading char_mask[m.start()] unscored the first word of all 5,008 turns.
        """
        ids_all: List[int] = []
        mask_all: List[bool] = []
        for m in re.finditer(BPE_PAT, text):
            ids = self._encode_piece(m.group(0))
            ids_all.extend(ids)
            label = any(char_mask[m.start():m.end()])
            mask_all.extend([label] * len(ids))
        return ids_all, mask_all

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
        # A turn is several "her: " lines, so the newline after the last one and
        # the newlines between them are the same token: the model cannot mark
        # where a turn ends, and stopping here returns her first message only.
        # 64.7% of turns are multi-message, so this is lossy on purpose --
        # returning the whole turn would mean decoding past the newline, into
        # the user's turns, which are never scored and so are unsupervised.
        # That needs a scored turn-end sentinel instead, which changes the
        # corpus format; first message is the faithful option until then.
        nl = self.encode("\n")
        stops = {nl[0]} if nl else set()
        stops.update(self.sentinel_id(i) for i in range(self.n_sentinels))
        stops.update(extra)
        return stops

    def save(self, path: Path):
        path = Path(path)
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
