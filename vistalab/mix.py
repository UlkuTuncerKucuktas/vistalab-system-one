import json
import math
import random
from collections import defaultdict

from . import DATA, read, save
from .overlap import bench_fingerprints, copies_benchmark, holds_benchmark
from .text import cut, encoded

REAL = 50000
PER_QUESTION = 10000
PER_SOURCE = 30000
ENGLISH = 10000
# TrGLUE's STS-B scores are far more generous than the benchmark's STSb-TR: 63% of pairs at levels 4-5, against 27%
LEFT_OUT = [("trglue", "similarity")]
# sources whose question text changes with every item (a tool, an aspect, an Open-Jev question) are capped per type
BY_TYPE = ["absa", "open_jev", "tools_atasoglu", "tools_hermes", "tools_when2call"]


def train_questions():
    for path in sorted((DATA / "train").glob("*.jsonl")):
        for line in open(path, encoding="utf-8"):
            item = json.loads(line)
            source = item["source"] + ("_en" if item.get("meta", {}).get("language") == "en" else "")
            for qid, question in item["questions"].items():
                if (item["source"], qid) not in LEFT_OUT:
                    row = {"id": item["id"], "source": item["source"], "qid": qid, "state": cut(item["state"]), "question": question, "gold": item["gold"][qid]}
                    if qid in item.get("soft_gold", {}):
                        row["soft_gold"] = item["soft_gold"][qid]
                    yield source, question["type"] if item["source"] in BY_TYPE else question["instructions"], row


def sample(rows, n, rng, weight):
    # a label with fewer rows than its share gives all it has, so a skewed question under its quota stays more skewed than
    # its weights say
    by_label = defaultdict(list)
    for row in rows:
        by_label[json.dumps(row["gold"])].append(row)
    weights = {label: weight(len(group)) for label, group in by_label.items()}
    total = sum(weights.values())
    return [row for label, group in by_label.items() for row in rng.sample(group, min(len(group), round(n * weights[label] / total)))]


def capped(groups):
    rng = random.Random(13)
    rows = []
    for source, by_key in sorted(groups.items()):
        wanted = {key: min(PER_QUESTION, len(group)) for key, group in by_key.items()}
        scale = min(1, (ENGLISH if source.endswith("_en") else PER_SOURCE) / sum(wanted.values()))
        for key, group in sorted(by_key.items()):
            rows += sample(group, int(wanted[key] * scale), rng, math.sqrt)
    rng.shuffle(rows)
    return rows


def swap_copies(rows, groups, key_of):
    # a row that holds a benchmark text gives its place to another row of the same question, with the same answer when one
    # is left
    taken = {(row["id"], row["qid"]) for row in rows}
    rng = random.Random(13)
    for i, row in enumerate(rows):
        if holds_benchmark(row["state"]):
            source, key = key_of[row["id"], row["qid"]]
            pool = groups[source][key]
            others = sorted(rng.sample(pool, len(pool)), key=lambda r: r["gold"] != row["gold"])
            rows[i] = next(r for r in others if (r["id"], r["qid"]) not in taken and not holds_benchmark(r["state"]))
            taken.add((rows[i]["id"], rows[i]["qid"]))


def flatten_stars(rows, groups):
    # star and point questions drawn again with flatter level shares into the places of the old ones; a place the draw
    # leaves empty keeps its row
    rng = random.Random(13)
    for source, by_key in sorted(groups.items()):
        for key, pool in sorted(by_key.items()):
            if pool[0]["question"]["type"] == "score" and source not in BY_TYPE:
                places = [i for i, row in enumerate(rows) if (row["source"], row["question"]["instructions"]) == (source, key)]
                drawn = [row for row in sample(pool, len(places), rng, lambda count: count**0.125) if not holds_benchmark(row["state"])]
                for i, row in zip(places, rng.sample(drawn, len(drawn))):
                    rows[i] = row


def real():
    groups, key_of = defaultdict(lambda: defaultdict(list)), {}
    for source, key, row in train_questions():
        groups[source][key].append(row)
        key_of[row["id"], row["qid"]] = source, key
    rows = capped(groups)[:REAL]
    swap_copies(rows, groups, key_of)
    flatten_stars(rows, groups)
    return rows


def main():
    # synthetic questions that hold or copy a benchmark text are dropped, and round 1's undecoded e-mails, which its seeds
    # kept
    seen = bench_fingerprints()
    first = [row for row in read(DATA / "synthetic" / "chunk.jsonl") if not encoded(row["state"]) and not holds_benchmark(row["state"]) and not copies_benchmark(row["state"], seen)]
    second = [row for row in read(DATA / "synthetic2" / "chunk.jsonl") if not holds_benchmark(row["state"]) and not copies_benchmark(row["state"], seen)]
    rows = real() + first + second
    random.Random(13).shuffle(rows)
    save(DATA / "mix" / "rgg.jsonl", rows)
    print(len(rows), "questions,", len(first), "from round 1 and", len(second), "from round 2")


if __name__ == "__main__":
    main()
