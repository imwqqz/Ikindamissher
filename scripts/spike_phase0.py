"""Phase 0 spike: prove NF4 QLoRA on Qwen3-1.7B works on the 4060.

Throwaway script (plan Phase 0). Checks, in order:
1. free disk >= 8 GB
2. NF4 load + one generate call
3. chat template keeps an EMPTY thinking block on assistant rows
4. five QLoRA training steps at seq 512, peak VRAM recorded
"""
import shutil
import sys
import time
from pathlib import Path

# Windows console: reflect the repo's own utf-8 stdout handling (her.py)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE = "Qwen/Qwen3-1.7B"
SEQ = 512
# thinking-token markers as rendered by the chat template (verified ASCII)
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def main():
    free = shutil.disk_usage("C:/").free
    print(f"disk free: {free / 2**30:.1f} GB")
    assert free > 8 * 2**30, "need >= 8 GB free disk for the checkpoint cache"

    tok = AutoTokenizer.from_pretrained(BASE)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map="cuda")
    print(f"loaded NF4 in {time.perf_counter() - t0:.0f}s")

    # 2. generate
    msgs = [{"role": "system", "content": "Sos una amiga cercana."},
            {"role": "user", "content": "hola! como estas?"}]
    prompt = tok.apply_chat_template(msgs, tokenize=False,
                                     add_generation_prompt=True,
                                     enable_thinking=False)
    ids = tok(prompt, return_tensors="pt").to("cuda")
    out = model.generate(**ids, max_new_tokens=60, do_sample=True,
                         temperature=0.7, top_p=0.9)
    text = tok.decode(out[0][ids["input_ids"].shape[1]:],
                      skip_special_tokens=True)
    print("generate ok ->", text.replace("\n", " / ")[:120])
    assert THINK_OPEN not in text and THINK_CLOSE not in text, \
        "thinking tokens leaked into generated text"

    # 3. template round-trip: assistant rows carry an EMPTY thinking block
    row = [{"role": "user", "content": "pq no viniste?\ndsp te paso todo"},
           {"role": "assistant", "content": "pq andaba ocupada\ndespues charlamos"}]
    templ = tok.apply_chat_template(row, tokenize=False, enable_thinking=False)
    open_i = templ.index(THINK_OPEN)
    close_i = templ.index(THINK_CLOSE)
    assert close_i > open_i, "thinking block never closed"
    between = templ[open_i + len(THINK_OPEN):close_i]
    assert not between.strip(), f"non-empty thinking block: {between!r}"
    assert templ.count(THINK_OPEN) == 1 and templ.count(THINK_CLOSE) == 1
    rt = tok.decode(tok(templ, return_tensors="pt").input_ids[0],
                    skip_special_tokens=False)
    assert "pq andaba ocupada" in rt, "round-trip lost the reply text"

    # 4. five QLoRA steps
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # ckpt needs an input grad path
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    labels = tok(templ, return_tensors="pt").input_ids[:, :SEQ].to("cuda")
    attn = torch.ones_like(labels)
    labels[:, :-1] = labels[:, 1:].clone()  # shift for LM loss
    labels[:, -1] = -100

    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=2e-4)
    torch.cuda.reset_peak_memory_stats()
    for step in range(5):
        out = model(input_ids=labels.clamp_min(0), attention_mask=attn,
                    labels=labels)
        out.loss.backward()
        opt.step()
        opt.zero_grad()
        print(f"step {step}: loss {out.loss.item():.3f}")
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"peak VRAM: {peak:.2f} GB")
    assert peak < 7.5, "VRAM gate failed"
    print("PHASE 0: OK")


if __name__ == "__main__":
    main()
