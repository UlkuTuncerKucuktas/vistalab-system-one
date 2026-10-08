import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from functools import cache

from s1bench import DATA as BENCH

# a copy typed without Turkish letters ("var mi") still matches
FOLD = str.maketrans("çğıöşüâîû", "cgiosuaiu")
QUOTED = re.compile(r"[\"“](.+?)[\"”]")
# the benchmark tasks whose rest shares a corpus with a training source, or whose claims fact-check reports quote: their
# rest texts are benchmark texts here too
SHARED = (
    "buyuksinema_tr", "checkthat_tr", "jev_noul_tr", "lavoir_zeroshot_tr", "mide22_tr", "musteri_yorumlari_tr", "news_topic_tr", "stsb_tr",
    "trclaim19_tr", "webfaq_tr", "xfact_tr", "xfact_teyit_tr",
)


def words(text):
    # composed first: TurkishMMLU stores its letters decomposed ("g" and a breve), which splits every word that has one
    text = unicodedata.normalize("NFKC", text).replace("İ", "i").replace("I", "ı").lower().translate(FOLD)
    return re.findall(r"[^\W_]+", re.sub(r"https?://\S+|@\w+", " ", text))


def fingerprint(w):
    return hashlib.blake2b(" ".join(w).encode(), digest_size=8).hexdigest()


def fingerprints(text):
    w = words(text)
    return {fingerprint(w)} | {fingerprint(w[i : i + 8]) for i in range(len(w) - 7)}


def windows(text):
    w = words(text)
    return [fingerprint(w[i : i + 8]) for i in range(len(w) - 7)]


def stems(text):
    return frozenset(word[:5] for word in words(text) if len(word) > 2)


def texts(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, list):
        return [text for v in value for text in texts(v)]
    return []


def item_texts(item):
    return texts(item["state"]) + [q for text in texts(item["questions"]) for q in QUOTED.findall(text)]


@cache
def bench_fingerprints():
    seen = set()
    for path in sorted(BENCH.glob("*/*.jsonl")):
        for line in open(path, encoding="utf-8"):
            for text in item_texts(json.loads(line)):
                if path.stem == "rest":
                    seen.add(fingerprint(words(text)))
                else:
                    seen |= fingerprints(text)
    return seen


def bench_states(task):
    return [json.loads(line)["state"] for path in sorted((BENCH / task).glob("*.jsonl")) for line in open(path, encoding="utf-8")]


def copies(text, seen):
    # a copy matches a whole benchmark text of 4 or more words, half of its 8-word windows, or 10 windows in a
    # row (17 words) that make up a fifth of it; very short texts such as "tamam", scattered windows and runs
    # inside long texts, such as quoted laws, are shared by unrelated texts
    w = words(text)
    hits = "".join("x" if fingerprint(w[i : i + 8]) in seen else "." for i in range(len(w) - 7))
    return (len(w) >= 4 and fingerprint(w) in seen) or hits.count("x") * 2 > len(hits) or ("x" * 10 in hits and hits.count("x") * 5 > len(hits))


def copies_benchmark(state, seen):
    # each text on its own and all of them joined, since a benchmark text can be split over several fields
    parts = texts(state)
    return any(copies(text, seen) for text in {*parts, " ".join(parts)})


@cache
def bench_windows(shared):
    owners, sizes = defaultdict(list), {}
    for path in sorted(BENCH.glob("*/*.jsonl")):
        if path.stem != "rest" or path.parent.name in shared:
            for line in open(path, encoding="utf-8"):
                item = json.loads(line)
                for n, text in enumerate(item_texts(item)):
                    found = set(windows(text))
                    for f in found:
                        owners[f].append((item["id"], n))
                    sizes[item["id"], n] = len(found)
    return owners, sizes


def holds_benchmark(state):
    # the benchmark text's side of copies(): a training text that holds half the windows of one benchmark text, or 10 of
    # them that make up a fifth of it, however long the training text is (a fact-check report quoting a whole post); a
    # passage two long texts share, such as a quoted law, does not count
    owners, sizes = bench_windows(SHARED)
    for text in {*texts(state), " ".join(texts(state))}:
        hits = Counter(key for f in set(windows(text)) for key in owners.get(f, ()))
        found = [key for key, n in hits.items() if n * 2 >= sizes[key] or n >= 10 and n * 5 >= sizes[key]]
        if found:
            return found[0][0]
    return None


def holds_dev_test(state):
    # the rule round 1's seeds were drawn with: 10 dev or test windows in a row at any length, or half the windows of one
    # dev or test text; a text under 17 words never makes a run of 10
    owners, sizes = bench_windows(())
    for text in {*texts(state), " ".join(texts(state))}:
        prints = windows(text)
        hits = Counter(key for f in set(prints) for key in owners.get(f, ()))
        halves = [key for key, n in hits.items() if n * 2 >= sizes[key]]
        if halves or "x" * 10 in "".join("x" if f in owners else "." for f in prints):
            return (halves or [max(hits, key=hits.get)])[0][0]
    return None


@cache
def bench_questions():
    seen, asked, options = set(), Counter(), set()
    for path in sorted(BENCH.glob("*/*.jsonl")):
        for line in open(path, encoding="utf-8"):
            for question in json.loads(line)["questions"].values():
                for text in texts(question):
                    seen |= fingerprints(text)
                asked[stems(question["instructions"]), path.parent.name] += 1
                if isinstance(question["criteria"], dict) and len(question["criteria"]) > 2:
                    options.add(stems(" ".join(question["criteria"])))
    return seen, {s for (s, _), n in asked.items() if n >= 20 and len(s) >= 3}, options


def clone(question):
    # a question that copies benchmark question text, asks a task's own question in nearly the same words, or offers
    # mostly a task's options (three or more)
    seen, templates, options = bench_questions()
    if any(copies(text, seen) for text in [question["instructions"], *question["criteria"].values()]):
        return True
    s = stems(question["instructions"])
    names = stems(" ".join(question["criteria"]))
    return any(len(s & t) * 5 >= len(s | t) * 3 for t in templates) or any(len(names & o) >= max(3, len(names) / 2) for o in options)
