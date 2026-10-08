import json
import multiprocessing
import random
import sys
from collections import Counter
from functools import cache

from . import DATA
from .families.records import RECORDS
from .overlap import SHARED, bench_windows, copies_benchmark, fingerprints, holds_benchmark, holds_dev_test, texts
from .text import cut, encoded, mask

PER_SOURCE = 20000
# all of Turkish Wikipedia, since only about one article in seven is long enough for a passage; round 1's seeds are the
# first ROUND_ONE of that sample
LARGER = {"wikipedia": 10**6}
ROUND_ONE = 100000
# touche_tr's speeches come from ParlaMint-TR, which transcribes the TBMM from 2011 (term 24) on
PARLAMINT_TERM = 24


@cache
def held_out():
    # the tool sets repeat their tool lists across splits, so only their requests are compared
    prints = set()
    for path in sorted((DATA / "heldout").glob("*.jsonl")):
        for line in open(path, encoding="utf-8"):
            state = json.loads(line)["state"]
            for text in texts(state["istek"] if isinstance(state, dict) and "istek" in state else state):
                prints |= fingerprints(text)
    return prints


def check(line):
    item = json.loads(line)
    item["state"] = cut(item["state"])
    if item["source"] == "parliament" and item["meta"]["term"] >= PARLAMINT_TERM:
        return item, "parlamint", "parlamint"
    if copies_benchmark(item["state"], held_out()):
        return item, "held out", "held out"
    return item, holds_dev_test(item["state"]), "encoded" if encoded(item["state"]) else holds_benchmark(item["state"])


def save_seeds(folder, name, checked):
    reasons = Counter()
    with open(DATA / folder / f"{name}.jsonl", "w", encoding="utf-8") as kept, open(DATA / folder / f"{name}.dropped.jsonl", "w", encoding="utf-8") as dropped:
        for item, reason in checked:
            if reason:
                dropped.write(json.dumps({"id": item["id"], "reason": reason}, ensure_ascii=False) + "\n")
                # a benchmark copy's reason is the benchmark item's id, counted by task
                reasons[reason.split("/")[0]] += 1
            else:
                state = mask(item["state"]) if isinstance(item["state"], str) else {k: mask(v) if isinstance(v, str) else v for k, v in item["state"].items()}
                kept.write(json.dumps({"id": item["id"], "source": name, "state": state}, ensure_ascii=False) + "\n")
    print(folder, name, sum(reasons.values()), "dropped", dict(reasons.most_common()), flush=True)


def main():
    # the held-out texts and both benchmark indexes are loaded before the workers fork, so that they share them
    held_out()
    bench_windows(SHARED)
    bench_windows(())
    for folder in ["seeds", "seeds2"]:
        (DATA / folder).mkdir(exist_ok=True)
    with multiprocessing.get_context("fork").Pool() as pool:
        for name in sys.argv[1:] or sorted({source for _, _, sources in RECORDS.values() for source in sources}):
            path = next(p for p in [DATA / "seed" / f"{name}.jsonl", DATA / "train" / f"{name}.jsonl"] if p.exists())
            lines = open(path, encoding="utf-8").readlines()
            sample = random.Random(name).sample(lines, min(LARGER.get(name, PER_SOURCE), len(lines)))
            checked = list(pool.imap(check, sample, chunksize=256))
            save_seeds("seeds", name, [(item, first) for item, first, _ in checked[:ROUND_ONE]])
            save_seeds("seeds2", name, [(item, second) for item, _, second in checked])


if __name__ == "__main__":
    main()
