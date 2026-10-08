import json
import random
import sys
from collections import Counter, defaultdict
from functools import cache

from huggingface_hub import snapshot_download

from . import DATA, pool, read, save
from .families import WRITERS, injection, knowledge, pairs, records, traps

# the two rounds of the release, each with the labellers and pooling its questions were labelled with
ROUNDS = {
    "synthetic": {"seeds": "seeds", "labellers": ["gemma-4-31b", "qwen3.5-35b-a3b"], "intercepts": True, "balanced": False},
    "synthetic2": {"seeds": "seeds2", "labellers": ["gemma-4-31b", "qwen3.5-397b-a17b"], "intercepts": False, "balanced": True},
}
# both rounds' writer outputs, questions, teacher labels and kept questions, as released
DATASET = "UlkuTuncerKucuktas/vistalab-system-one-synthetic"


@cache
def read_seeds(name):
    return {source: read(DATA / name / f"{source}.jsonl") for _, _, sources in records.RECORDS.values() for source in sources}


def jobs():
    # each round draws from its own Random(13), in the release's order
    rng, seeds = random.Random(13), read_seeds(ROUNDS["synthetic"]["seeds"])
    first = records.jobs(seeds, rng) + knowledge.passage_jobs(seeds, rng) + traps.jobs(seeds, rng)
    rng, seeds = random.Random(13), read_seeds(ROUNDS["synthetic2"]["seeds"])
    exams = knowledge.exam_jobs(seeds, {job["seed"] for job in first if job["kind"] == "knowledge"}, rng)
    sentences = pairs.sentence_pool(seeds, rng)
    second = exams + pairs.meaning_jobs(sentences, rng) + injection.jobs() + pairs.similarity_jobs(sentences, rng)
    for name, out in zip(ROUNDS, [first, second]):
        save(DATA / name / "jobs.jsonl", out)
        print(name, len(out), "jobs", dict(Counter(f"{job['kind']} {job['writer']}" for job in out)))


def parsed(text):
    # an output cut off at its token limit is not valid JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def questions():
    for name, settings in ROUNDS.items():
        rng, asked, items = random.Random(13), set(), []
        by_id = {job["id"]: job for job in read(DATA / name / "jobs.jsonl")}
        for writer in WRITERS:
            for written in read(DATA / name / f"written_{writer}.jsonl"):
                job, output = by_id[written["id"]], parsed(written["text"])
                if output is None:
                    continue
                if job["kind"] == "schema":
                    items += records.questions(job, output, rng, asked, read_seeds(settings["seeds"]))
                elif job["kind"] == "knowledge":
                    items += knowledge.passage_questions(job, output, rng, asked)
                elif job["kind"] == "trap":
                    items += traps.questions(job, output, rng)
                elif job["kind"] == "sınav":
                    items += knowledge.exam_questions(job, output, rng, asked)
                elif job["kind"] == "anlam":
                    items += pairs.meaning_questions(job, output, rng)
                elif job["kind"] == "kandırma":
                    items += injection.questions(job, output, rng)
                elif job["kind"] == "benzerlik":
                    items += pairs.similarity_questions(job, output, rng)
        save(DATA / name / "questions.jsonl", items)
        print(name, len(items), "questions", dict(Counter(f"{item['meta']['kind']} {item['qid']}" for item in items)))


def agrees(row, p):
    if row["question"]["type"] == "score":
        return abs(sum(int(key) * v for key, v in p.items()) - int(row["meta"]["expected"])) <= 1
    return max(p, key=p.get) == row["meta"]["expected"]


def balanced(rows, ids):
    # within each family, and each kind of edit, as many questions of every answer as of the rarest; exam questions are not
    # balanced across their keys
    cells = defaultdict(list)
    for i in ids:
        meta = rows[i]["meta"]
        cells[meta["kind"], meta.get("edit"), meta["expected"] if meta["kind"] != "sınav" else ""].append(i)
    kept = []
    for group in sorted({key[:2] for key in cells}, key=str):
        members = [cells[key] for key in sorted(cells, key=str) if key[:2] == group]
        n = min(map(len, members))
        kept += [i for m in members for i in m[:n]]
    return kept


def labelled(row, p):
    top = max(p, key=p.get)
    gold = top == "true" if row["question"]["type"] == "noul" else top
    return {**row, "gold": gold, "soft_gold": {key: round(float(v), 4) for key, v in p.items()}}


def finish():
    for name, settings in ROUNDS.items():
        rows = {row["id"]: row for row in read(DATA / name / "questions.jsonl")}
        fitted = pool.fit(*settings["labellers"])
        teachers = [pool.scores(rows, read(DATA / name / f"{labeller}.jsonl")) for labeller in settings["labellers"]]
        pooled = {i: pool.pooled(row, teachers, fitted, settings["intercepts"]) for i, row in rows.items()}
        built = [i for i, row in rows.items() if row["meta"]["kind"] != "schema" and agrees(row, pooled[i])]
        kept = (balanced(rows, built) if settings["balanced"] else built) + records.kept(rows, pooled)
        save(DATA / name / "chunk.jsonl", [labelled(rows[i], pooled[i]) for i in kept])
        print(name, len(kept), "questions kept", dict(Counter(f"{rows[i]['meta']['kind']} {rows[i]['qid']}" for i in kept)))


def pull():
    snapshot_download(DATASET, repo_type="dataset", local_dir=DATA, allow_patterns=[f"{name}/*" for name in ROUNDS])


if __name__ == "__main__":
    {"jobs": jobs, "questions": questions, "finish": finish, "pull": pull}[sys.argv[1]]()
