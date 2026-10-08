import json
import random
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from s1bench.llm import LLM
from s1bench.prompt import options_of, render

from . import DATA

BASES = {"12b": "google/gemma-4-12b-it", "e4b": "google/gemma-4-E4B-it"}
MIX = DATA / "mix" / "rgg.jsonl"
RUNS = Path(__file__).parent.parent / "runs"
LR = 5e-5
RANK = 32
# the seed sets the option orders, the order of the batches and the LoRA's starting weights
SEED = 13
BATCH_SIZE = 32
TOKENS_PER_PASS = 32768
MAX_PROMPT_TOKENS = 4096
LAYERS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def examples(llm):
    # one random option order per Choice and Noul question; Score stays ascending
    rng = random.Random(SEED)
    out = []
    for line in open(MIX, encoding="utf-8"):
        row = json.loads(line)
        options = options_of(row["question"])
        if row["question"]["type"] != "score":
            rng.shuffle(options)
        gold = str(row["gold"]).lower() if isinstance(row["gold"], bool) else str(row["gold"])
        soft = row.get("soft_gold")
        target = [soft.get(key, 0) if soft else float(key == gold) for key, _ in options]
        ids = llm.tokenizer(llm.prompt(render(row, row["question"], options)), add_special_tokens=False).input_ids
        if sum(target) > 0 and len(ids) <= MAX_PROMPT_TOKENS and len(options) <= len(llm.label_ids):
            out.append((ids, target, row["question"]["type"] == "score"))
    return out


def variants(llm):
    # each letter's token ids in one tensor, padded with its first id and masked
    width = max(len(ids) for ids in llm.label_ids)
    index = torch.tensor([ids + ids[:1] * (width - len(ids)) for ids in llm.label_ids], device="cuda")
    valid = torch.tensor([[i < len(ids) for i in range(width)] for ids in llm.label_ids], device="cuda")
    return index, valid


def letters(model, llm, table, batch):
    # the benchmark's readout: last position, log-sum-exp over each letter's token variants, softmax over the letters shown
    index, valid = table
    padded = llm.tokenizer.pad({"input_ids": [ids for ids, _, _ in batch]}, return_tensors="pt").to("cuda")
    logits = model(**padded, logits_to_keep=1, use_cache=False).logits[:, -1].float()
    shown = torch.arange(len(index), device="cuda") < torch.tensor([len(target) for _, target, _ in batch], device="cuda")[:, None]
    scores = logits[:, index].masked_fill(~valid, float("-inf")).logsumexp(-1)
    return scores.masked_fill(~shown, float("-inf")).log_softmax(-1), shown


def loss_of(logps, shown, batch):
    # cross-entropy against the (soft) gold; Score adds the squared error of the expected level
    targets = torch.zeros(logps.shape)
    for row, (_, target, _) in enumerate(batch):
        targets[row, : len(target)] = torch.tensor(target) / sum(target)
    targets = targets.to(logps.device)
    total = -(targets * logps.masked_fill(~shown, 0)).sum()
    for row, (_, target, score) in enumerate(batch):
        if score:
            levels = torch.arange(len(target), device=logps.device)
            gap = (logps[row, : len(target)].exp() * levels).sum() - (targets[row, : len(target)] * levels).sum()
            total = total + gap**2 / (len(target) - 1) ** 2
    return total


def with_lora(llm):
    llm.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(llm.model, LoraConfig(r=RANK, lora_alpha=RANK, target_modules=LAYERS))
    model.train()
    return model


def train(size):
    torch.manual_seed(SEED)
    llm = LLM(BASES[size])
    data = sorted(examples(llm), key=lambda example: len(example[0]))
    batches = [data[i : i + BATCH_SIZE] for i in range(0, len(data), BATCH_SIZE)]
    random.Random(SEED).shuffle(batches)
    model = with_lora(llm)
    table = variants(llm)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=0)
    warmup = len(batches) // 30
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step + 1) / warmup, (len(batches) - step) / (len(batches) - warmup)))
    print(f"{len(data)} questions, {len(batches)} steps", flush=True)
    start, running = time.time(), 0.0
    for step, batch in enumerate(batches):
        # as many questions per forward pass as fit in the token budget; batches hold questions of similar length
        size = TOKENS_PER_PASS // max(len(ids) for ids, _, _ in batch)
        for i in range(0, len(batch), size):
            micro = batch[i : i + size]
            logps, shown = letters(model, llm, table, micro)
            loss = loss_of(logps, shown, micro) / len(batch)
            loss.backward()
            running += loss.item()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        schedule.step()
        optimizer.zero_grad()
        if (step + 1) % 50 == 0:
            print(f"step {step + 1}/{len(batches)} loss {running / 50:.4f} {time.time() - start:.0f}s", flush=True)
            running = 0.0
    model.save_pretrained(RUNS / f"vistalab-system-one-{size}")


if __name__ == "__main__":
    train(sys.argv[1])
