import json
import sys

from . import DATA, save
from .overlap import bench_fingerprints, copies_benchmark, fingerprint, words
from .sources import SOURCES


def pull(name):
    # heldout sorts before train, so training items whose text is held out are dropped too
    seen = bench_fingerprints()
    keys, held_out = set(), set()
    for split, items in sorted(SOURCES[name]().items()):
        kept = []
        for item in items:
            text = fingerprint(words(json.dumps(item["state"], ensure_ascii=False)))
            key = fingerprint(words(json.dumps([item["state"], item.get("questions")], ensure_ascii=False)))
            if key not in keys and text not in held_out and not copies_benchmark(item["state"], seen):
                keys.add(key)
                kept.append(item)
                if split == "heldout":
                    held_out.add(text)
        save(DATA / split / f"{name}.jsonl", [{"id": f"{name}/{split}/{i:06d}", "source": name, **item} for i, item in enumerate(kept)])
        print(f"{name} {split}: {len(kept)} of {len(items)}", flush=True)


if __name__ == "__main__":
    for name in sys.argv[1:] or SOURCES:
        pull(name)
