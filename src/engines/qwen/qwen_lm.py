"""Qwen3 QLoRA: load, train, and serve "her" voice on top of Qwen3-1.7B.

Design rationale: git history (docs/REPORT.md, removed).
"""
import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Windows console: same utf-8 handling as client/repl.py
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# read before import torch at allocator init; measured no-op on torch 2.14, belt-and-braces
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE = "Qwen/Qwen3-1.7B"
ADAPTER_DIR = _ROOT / "data" / "her_qwen_adapter"

_tok_cache = {}


def get_tokenizer(base: str = BASE) -> AutoTokenizer:
    if base not in _tok_cache:
        _tok_cache[base] = AutoTokenizer.from_pretrained(base)
    return _tok_cache[base]


def apply_template(messages, tokenizer=None, add_generation_prompt=False):
    """One place renders the chat template, thinking disabled everywhere."""
    tok = tokenizer or get_tokenizer()
    return tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=add_generation_prompt,
                                   enable_thinking=False)


_markers_cache = {}


def think_markers(tokenizer=None):
    """(open, close) of the empty thinking block, probed at runtime so no
    literal lands in source."""
    if "m" not in _markers_cache:
        probe = apply_template(
            [{"role": "user", "content": "x"},
             {"role": "assistant", "content": "y"}], tokenizer)
        seg = probe.split("<|im_start|>assistant\n", 1)[1]
        open_m, rest = seg.split("\n\n", 1)
        close_m = rest.split("\n\n", 1)[0]
        assert open_m and close_m and open_m != close_m, probe
        _markers_cache["m"] = (open_m, close_m)
    return _markers_cache["m"]


_ASSIST_HEADER = "<|im_start|>assistant\n"
IM_END = "<|im_end|>"


def encode_row(messages, tokenizer=None):
    """(input_ids, labels) for one SFT row, completion-only loss.

    Assistant spans (empty thinking block + reply + <|im_end|>) are scored;
    everything else gets -100. Spans come from char offsets of one full
    render.
    shortcut: literal <|im_end|> inside content splits a span; content is
    chat text, not tokens.
    """
    tok = tokenizer or get_tokenizer()
    text = apply_template(messages, tok)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    spans = []
    pos = 0
    while (s := text.find(_ASSIST_HEADER, pos)) != -1:
        a = s + len(_ASSIST_HEADER)
        e = text.find(IM_END, a)
        assert e != -1, f"assistant turn without <|im_end|>: {text[a:a+60]!r}"
        spans.append((a, e + len(IM_END)))
        pos = e + 1
    labels = [tid if any(o0 < se and o1 > ss for ss, se in spans) else -100
              for tid, (o0, o1) in zip(enc.input_ids, enc.offset_mapping)]
    assert any(l != -100 for l in labels), "row scored nothing"
    return enc.input_ids, labels


def _selfcheck_examples(rows, tok, n=3):
    # decode-as-ids proof: labels must be real token ids of the assistant turn (a mask/ids mix-up once trained to loss 0)
    for row in rows[:n]:
        ids, labels = encode_row(row, tok)
        scored = tok.decode([i for i, l in zip(ids, labels) if l != -100],
                            skip_special_tokens=True)
        labels_decoded = tok.decode([l for l in labels if l != -100],
                                    skip_special_tokens=True)
        for m in row:
            frag = m["content"][:20]
            keep = m["role"] == "assistant"
            assert (frag in scored) == keep or not frag, (m["role"], frag)
            if keep and frag:
                assert frag in labels_decoded, \
                    f"labels are not token ids of {frag!r}"


def load_engine(base: str = BASE, adapter=None, device="cuda"):
    """(tokenizer, model) ready to generate. NF4-quantized base; optional
    LoRA adapter loaded from disk (the sidecar produced by train())."""
    tok = get_tokenizer(base)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=bnb, device_map=device)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter))
        print(f"adapter loaded: {adapter}")
    model.eval()
    return tok, model


def reply(tok, model, messages, max_new_tokens=80, temperature=0.7, top_p=0.9,
          repeat_penalty=1.0):
    """One assistant turn: template -> sample -> visible text (leaked think
    blocks stripped)."""
    prompt = apply_template(messages, tok, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=True,
                      temperature=max(temperature, 1e-8), top_p=top_p)
    if repeat_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = repeat_penalty
    out = model.generate(**ids, **gen_kwargs)
    text = tok.decode(out[0][ids["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip()
    return strip_think(text, tok)


def strip_think(text, tokenizer=None):
    """Remove any thinking block; a block that never closed is a truncated
    leak, so nothing after the open marker is visible text."""
    open_m, close_m = think_markers(tokenizer)
    if open_m in text:
        head, _, tail = text.partition(open_m)
        if close_m in tail:
            _, _, rest = tail.partition(close_m)
            return (head + rest).strip()
        return head.strip().rstrip(" ...") if head.strip() else ""
    return text.strip()


def build_bnb():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def train(args):
    tok = get_tokenizer(args.base)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=build_bnb(), device_map=device)
    # non-reentrant checkpointing once, before the PEFT wrap
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    trainables = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in trainables):,}")

    rows = [json.loads(line) for line in
            Path(args.data).read_text(encoding="utf-8").splitlines()]
    rows = [r["messages"] for r in rows]
    _selfcheck_examples(rows, tok)

    examples = []
    dropped = 0
    for row in rows:
        ids, labels = encode_row(row, tok)
        if len(ids) > args.seq:
            dropped += 1
            continue
        examples.append((len(ids), ids, labels))
    print(f"examples: {len(examples)} (dropped over seq {args.seq}: {dropped})")

    opt = torch.optim.AdamW(trainables, lr=args.lr)

    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    # token-budget micro-batching: padded tokens, not row count, caps VRAM
    examples.sort(key=lambda e: e[0])
    batches = []
    i = 0
    while i < len(examples):
        j = i + 1
        while j < len(examples) and (j - i) * examples[j - 1][0] \
                <= args.token_budget:
            j += 1
        batches.append(examples[i:j])
        i = j
    print(f"micro-batches: {len(batches)} (token budget {args.token_budget})")

    warmup = max(5, int(0.03 * len(batches) * args.epochs / args.grad_accum))
    optimizer_steps = math.ceil(len(batches) / args.grad_accum) * args.epochs

    def lr_at(step):
        if step < warmup:
            return args.lr * (step + 1) / warmup
        p = (step - warmup) / max(1, optimizer_steps - warmup)
        return args.lr * (0.5 * (1 + math.cos(math.pi * min(p, 1.0))))

    def collate(batch):
        width = batch[-1][0]
        ids = torch.tensor([e[1] + [pad] * (width - len(e[1]))
                            for e in batch], device=device)
        attn = torch.tensor([[1] * len(e[1]) + [0] * (width - len(e[1]))
                             for e in batch], device=device)
        labels = torch.tensor([e[2] + [-100] * (width - len(e[2]))
                               for e in batch], device=device)
        return ids, attn, labels

    torch.cuda.reset_peak_memory_stats()
    step = 0
    t0 = __import__("time").perf_counter()
    for epoch in range(args.epochs):
        random.Random(args.seed + epoch).shuffle(batches)
        micro = 0
        opt.zero_grad()
        epoch_loss, n_batches = 0.0, 0
        for batch in batches:
            ids, attn, labels = collate(batch)
            out = model(input_ids=ids, attention_mask=attn, labels=labels)
            (out.loss / args.grad_accum).backward()
            micro += 1
            epoch_loss += out.loss.item()
            n_batches += 1
            if micro % args.grad_accum:
                continue
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            torch.nn.utils.clip_grad_norm_(trainables, 1.0)
            opt.step()
            opt.zero_grad()
            # empty_cache keeps the reserved pool off the WDDM spill edge
            torch.cuda.empty_cache()
            step += 1
            if step % 25 == 0:
                peak = torch.cuda.max_memory_allocated() / 2**30
                print(f"epoch {epoch + 1} step {step}/{optimizer_steps} "
                      f"loss {out.loss.item():.3f} lr {lr_at(step):.2e} "
                      f"peak VRAM {peak:.2f} GB", flush=True)
        print(f"epoch {epoch + 1}: mean loss "
              f"{epoch_loss / max(n_batches, 1):.4f}")
    elapsed = __import__("time").perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"peak VRAM: {peak:.2f} GB | time: {elapsed / 60:.1f} min")

    out_dir = Path(args.out)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"adapter saved: {out_dir}")
    return model


def build_parser():
    ap = argparse.ArgumentParser(
        description="qlora-train a Qwen3 chat model on her SFT data")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--data", default=str(_ROOT / "data" / "sft.json"))
    ap.add_argument("--out", default=str(ADAPTER_DIR))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--token-budget", type=int, default=800,
                    help="max padded tokens per micro-batch; the fp32 CE "
                         "upcast over the ~152k vocab is what eats VRAM")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()

