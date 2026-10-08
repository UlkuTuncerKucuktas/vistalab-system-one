import os
import sys

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.sampling_params import StructuredOutputsParams

from . import DATA, read, save
from .synthetic import ROUNDS

# FlashInfer's top-p sampler compiles a kernel at start-up, which needs a CUDA compiler the pod does not have
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
REPOS = {"gemma": "google/gemma-4-26B-A4B-it", "qwen": "Qwen/Qwen3.6-35B-A3B"}


def main(writer):
    tokenizer = AutoTokenizer.from_pretrained(REPOS[writer])
    llm = LLM(REPOS[writer], dtype="bfloat16", max_model_len=16384, max_num_seqs=256, enable_prefix_caching=writer == "gemma", language_model_only=True)
    for name in ROUNDS:
        jobs = [job for job in read(DATA / name / "jobs.jsonl") if job["writer"] == writer]
        prompts = []
        for job in jobs:
            text = tokenizer.apply_chat_template([{"role": "user", "content": job["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            prompts.append(TokensPrompt(prompt_token_ids=tokenizer(text, add_special_tokens=False).input_ids))
        # no free whitespace in the JSON, since a writer can otherwise fill its whole budget with blank lines
        params = [SamplingParams(temperature=0.8, top_p=0.95, max_tokens=job["max_tokens"], structured_outputs=StructuredOutputsParams(json=job["schema"], disable_any_whitespace=True)) for job in jobs]
        save(DATA / name / f"written_{writer}.jsonl", [{"id": job["id"], "text": output.outputs[0].text} for job, output in zip(jobs, llm.generate(prompts, params))])


if __name__ == "__main__":
    main(sys.argv[1])
