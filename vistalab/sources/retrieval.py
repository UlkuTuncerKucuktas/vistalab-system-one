import re
from collections import defaultdict

from ..download import hf_rows
from ..overlap import bench_fingerprints, copies, words
from ..text import lower, mask

FITS = {
    "type": "noul",
    "instructions": "Bu cevap soruyu yanıtlıyor mu?",
    "criteria": {"true": "Cevap sorulan şeyi yanıtlıyor.", "false": "Cevap başka bir soruya ait ya da sorulanı yanıtlamıyor."},
}
BETTING = re.compile(r"bet|bahis|casino|slot|iddaa")
# gambling spam also sits on unrelated domains
GAMBLING = re.compile(r"bahis|casino|kumar|slot oyun|slot makine|rulet|poker|deneme bonusu|free ?spin|iddaa|1xbet")
PER_SITE = 30


def similar(a, b):
    a, b = set(words(a)), set(words(b))
    return len(a & b) * 5 >= len(a | b)


def fits_item(question, answer, fits, site):
    return {"state": {"soru": mask(question), "cevap": mask(answer)}, "questions": {"fits": FITS}, "gold": {"fits": fits}, "meta": {"site": site}}


def build_webfaq():
    bench = bench_fingerprints()
    sites = defaultdict(list)
    for r in hf_rows("PaDaS-Lab/webfaq", "tur", "default"):
        sites[r["origin"]].append(r)
    items = []
    for site, rows in sites.items():
        # a site that hosts a benchmark question is left out whole, since its other questions are templated
        # near-copies; so are betting sites, and a few questions per site keep the set from being site templates
        gambling = sum(bool(GAMBLING.search(lower(r["question"] + " " + r["answer"]))) for r in rows)
        if BETTING.search(site) or gambling > 1 or any(copies(r["question"], bench) or copies(r["answer"], bench) for r in rows):
            continue
        rows = rows[:PER_SITE]
        for i, r in enumerate(rows):
            # every second question gets the answer of the question before it, unless the two questions are near-copies
            other = rows[i - 1]
            if i % 2 == 0:
                items.append(fits_item(r["question"], r["answer"], True, site))
            elif other["answer"] != r["answer"] and not similar(r["question"], other["question"]):
                items.append(fits_item(r["question"], other["answer"], False, site))
    return {"train": items}
