# vistalab-system-one

Training code for two Turkish models that answer System One requests, fine-tuned by VistaLab:

- [vistalab-system-one-12b](https://huggingface.co/UlkuTuncerKucuktas/vistalab-system-one-12b), from Gemma 4 12B
- [vistalab-system-one-e4b](https://huggingface.co/UlkuTuncerKucuktas/vistalab-system-one-e4b), from Gemma 4 E4B

A System One request is a text and a question with named answer options; the model's answer is read from the
probabilities it gives the options' letters in one forward pass.
[system-one-bench-tr](https://github.com/UlkuTuncerKucuktas/system-one-bench-tr) describes the format and is the
benchmark we evaluate on.

## Background

Jev, TypeSafe's System One model, is available through a commercial API. We wanted to see how close an open model can
come on Turkish requests when it is fine-tuned only on public data and on questions written and labelled by open-weight
models.

## What we did

1. **Real questions.** We converted about 40 public Turkish datasets (reviews, complaints, news, social media, spam,
   assistant safety, tool use, fact-checking, exams and others) into System One questions and sampled 50,000 of them,
   with a cap per source so that no single dataset dominates.
2. **Synthetic questions.** In two rounds, Gemma 4 26B-A4B and Qwen3.6-35B-A3B wrote questions about records (reviews,
   complaints, news, posts, messages), about Wikipedia passages, about sentence pairs that keep or change their meaning,
   and about messages that try to manipulate an assistant. Gemma 4 31B, together with Qwen3.5-35B-A3B in round 1 and
   Qwen3.5-397B-A17B in round 2, answered each question. Their answers were combined with weights fitted on real
   labelled questions, and questions whose combined answer disagreed with how they were built were dropped. 163,680
   synthetic questions remained; they are published as
   [vistalab-system-one-synthetic](https://huggingface.co/datasets/UlkuTuncerKucuktas/vistalab-system-one-synthetic).
3. **Benchmark texts.** Training texts that copy a benchmark text were removed, and a final check found none of the
   benchmark's 106,003 dev and test texts in the training data. For a few tasks the models were trained on other splits
   of the same datasets (the training splits of TrGLUE and Lavoir, whose other splits are in the benchmark) or on other
   texts from the same sources (other WebFAQ sites, fact-check reports about other claims).
4. **Training.** A rank-32 LoRA on the attention and MLP layers of the language model, trained for one pass over about
   213,000 questions, with the loss on the probabilities of the answer letters (the readout the benchmark uses). The
   published models have the LoRA merged in.

## Results

On [system-one-bench-tr](https://github.com/UlkuTuncerKucuktas/system-one-bench-tr), test split. Accuracy (%) on the 25
main tasks and by area, raw log-loss on the main tasks (lower is better).

| Model | Accuracy | Raw log-loss | Routing | Opinion | Safety | Fact-checking | Judging | Workflow | Politics | Language | Knowledge |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vistalab-system-one-12b | 77.6 | 0.59 | 88.8 | 65.0 | 92.9 | 54.5 | 83.3 | 89.3 | 82.0 | 80.4 | 74.1 |
| vistalab-system-one-e4b | 75.4 | 0.62 | 87.8 | 63.3 | 92.1 | 57.5 | 79.5 | 89.7 | 76.6 | 76.3 | 66.4 |
| Jev | 77.6 | 0.82 | 87.4 | 62.4 | 90.6 | 51.2 | 81.7 | 87.3 | 86.8 | 78.9 | 86.9 |
| Gemma 4 12B (untrained) | 75.9 | 1.85 | 86.2 | 61.4 | 86.5 | 58.5 | 82.0 | 87.3 | 85.3 | 77.8 | 74.2 |
| Gemma 4 E4B (untrained) | 68.9 | 1.61 | 83.2 | 59.1 | 81.7 | 44.4 | 72.0 | 86.1 | 80.2 | 69.1 | 63.0 |

The 12B reaches the same main-task accuracy as Jev with a lower raw log-loss, and the E4B is 6.5 points above its base
model. Both are weaker than Jev on knowledge and politics, and the 12B is below its own base model on fact-checking and
politics.

## Running it

With [system-one-bench-tr](https://github.com/UlkuTuncerKucuktas/system-one-bench-tr) cloned next to this repository:

```bash
pip install -r requirements.txt -e ../system-one-bench-tr
s1bench build                       # the benchmark, used by the copy checks and for evaluation
python -m vistalab.pull             # the real datasets
python -m vistalab.synthetic pull   # the released synthetic questions
python -m vistalab.mix              # the training mix
python -m vistalab.train 12b        # or e4b
s1bench run runs/vistalab-system-one-12b
s1bench score
```

To build the synthetic questions yourself instead of downloading them, run `seeds`, `pool`, `synthetic jobs`,
`write gemma|qwen`, `synthetic questions`, `label <labeller>` and `synthetic finish` before `mix`.

What each step needs:

- **`s1bench build`:** the benchmark's own requirements (see its README).
- **`pull`:** two gated Hugging Face datasets, `manueltonneau/turkish-hate-speech-superset` and `alibayram/doktorsitesi`.
  Accept their terms on their Hugging Face pages, then run `hf auth login` with a read token. Nine sources are public
  Kaggle datasets downloaded with kagglehub; no account should be needed, but if Kaggle asks for one, set
  `KAGGLE_API_TOKEN`. One source is downloaded from interpress.com.
- **`synthetic pull`:** a public Hugging Face dataset; no token.
- **`train`:** one GPU. On an H200 the 12B took about 5 hours and the E4B about 3 hours, using 70 to 85 GB of memory.
- **`write` and `label`:** vLLM. The writer and labeller models (Gemma 4 and Qwen) are not gated but are large; the
  Qwen3.5-397B-A17B labeller needs two GPUs.

Several training sources are non-commercial, so the models and the synthetic data are released under CC BY-NC 4.0.
