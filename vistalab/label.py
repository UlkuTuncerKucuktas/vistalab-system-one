import math
import os
import sys

import torch
from s1bench.prompt import LABELS, label_token_ids, options_of, render
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed.device_communicators import shm_broadcast
from vllm.inputs import TokensPrompt

from . import DATA, read, save
from .synthetic import ROUNDS

LABELLERS = {"gemma-4-31b": "google/gemma-4-31B-it", "qwen3.5-35b-a3b": "Qwen/Qwen3.5-35B-A3B", "qwen3.5-397b-a17b": "Qwen/Qwen3.5-397B-A17B-GPTQ-Int4"}
# the letters are read before any sampling, and FlashInfer's sampler compiles a kernel at start-up, which needs a CUDA
# compiler the pod does not have
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
# the pod's /dev/shm holds 64 MiB, and vLLM's queues on two GPUs ask for about 660 MiB, so they get smaller rings, set at
# import for the processes vLLM spawns too
os.environ["VLLM_MQ_MAX_CHUNK_BYTES_MB"] = "2"
os.environ["NCCL_SHM_DISABLE"] = "1"
shm_broadcast.MessageQueue.__init__.__defaults__ = (None, 1 << 20, 4, None)


def logsumexp(values):
    top = max(values)
    return top + math.log(sum(math.exp(v - top) for v in values))


def shifted_orders(question):
    options = options_of(question)
    if question["type"] == "score":
        return [options, options[::-1]]
    shifts = range(len(options)) if len(options) <= 8 else sorted({len(options) * i // 8 for i in range(8)})
    return [options[s:] + options[:s] for s in shifts]


class Labeller:
    # each letter's log-probability over the whole vocabulary, from token ids built as the benchmark builds them
    def __init__(self, repo):
        self.tokenizer = AutoTokenizer.from_pretrained(repo)
        self.label_ids = [label_token_ids(self.tokenizer, label) for label in LABELS]
        self.ids = sorted({i for ids in self.label_ids for i in ids})
        # the 4-bit Qwen3.5-397B-A17B needs both H200s; prefix caching only for Gemma, since the hybrid Qwen models gain
        # almost nothing from it
        gpus = torch.cuda.device_count()
        self.llm = LLM(repo, dtype="bfloat16", tensor_parallel_size=gpus, gpu_memory_utilization=0.94, max_model_len=32768, max_num_seqs=128, enable_prefix_caching="gemma" in repo.lower(), language_model_only=True)
        self.params = SamplingParams(max_tokens=1, temperature=0, logprob_token_ids=self.ids, detokenize=False)

    def tokens(self, text):
        prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return self.tokenizer(prompt, add_special_tokens=False).input_ids

    def letters(self, texts, sizes):
        outputs = self.llm.generate([TokensPrompt(prompt_token_ids=self.tokens(text)) for text in texts], self.params)
        return [[logsumexp([output.outputs[0].logprobs[0][i].logprob for i in ids]) for ids in self.label_ids[:size]] for output, size in zip(outputs, sizes)]


def main(name):
    labeller = Labeller(LABELLERS[name])
    rounds = [folder for folder, settings in ROUNDS.items() if name in settings["labellers"]]
    for folder in ["pool", *rounds]:
        rows = read(DATA / folder / "questions.jsonl")
        jobs = [(row, options) for row in rows for options in shifted_orders(row["question"])]
        logps = labeller.letters([render(row, row["question"], options) for row, options in jobs], [len(options) for _, options in jobs])
        save(DATA / folder / f"{name}.jsonl", [{"id": row["id"], "keys": [key for key, _ in options], "logps": values} for (row, options), values in zip(jobs, logps)])
        print(folder, len(rows), "questions,", len(jobs), "prompts", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
