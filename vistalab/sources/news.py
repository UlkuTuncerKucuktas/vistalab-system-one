import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

from ..download import download, hf_rows, kaggle_file, read_tsv, read_zip
from ..overlap import bench_states, words
from ..text import lower
from .questions import choice

VERDICTS = {
    "true": "doğru",
    "dogru": "doğru",
    "doğru": "doğru",
    "mixed": "karma",
    "misleading": "yanıltıcı",
    "false": "yanlış",
    "yanliş": "yanlış",
    "yanliþ": "yanlış",
}
VERDICT = {
    "type": "choice",
    "instructions": "Doğruluk kontrolü raporuna göre bu iddia doğru mu?",
    "criteria": {
        "doğru": "İddia doğru.",
        "karma": "İddianın bir kısmı doğru, bir kısmı yanlış.",
        "yanıltıcı": "İddia gerçek bir şeye dayanıyor ama bağlamından koparılmış ya da yanıltıcı biçimde sunulmuş.",
        "yanlış": "İddia yanlış.",
    },
}
VERDICT_WORDS = re.compile(r"\b(doğru|yanlış|yanıltıcı|yalan|asılsız)(dur|dür|dır|dir|tır|tir|dı|tı|ı|lar|ları)?\b|\bkarma\b|\bsonuç\b|gerçeği yansıtm")
# FACTurk re-checks some claims of the benchmark's fact-checking tasks, often in other words
FACT_CHECK_TASKS = ["xfact_tr", "xfact_teyit_tr", "trclaim19_tr", "mide22_tr", "checkthat_tr"]
TOPIC = "Bu haber hangi kategoriye giriyor?"
INTERPRESS = {
    "aktuel": "güncel",
    "bilisim": "bilişim",
    "egitim": "eğitim",
    "ekonomi": "ekonomi",
    "gida": "gıda",
    "iletisim": "iletişim",
    "kultursanat": "kültür sanat",
    "magazin": "magazin",
    "saglik": "sağlık",
    "savunma": "savunma",
    "seyahat": "seyahat",
    "siyasi": "siyaset",
    "spor": "spor",
    "teknoloji": "teknoloji",
    "ticaret": "ticaret",
    "turizm": "turizm",
    "yasam": "yaşam",
}
KEMIK = {
    "dunya": "dünya",
    "ekonomi": "ekonomi",
    "kultur-sanat": "kültür sanat",
    "magazin": "magazin",
    "saglik": "sağlık",
    "siyaset": "siyaset",
    "spor": "spor",
    "teknoloji": "teknoloji",
    "turkiye": "Türkiye",
    "yasam": "yaşam",
}
BILCAT = {
    "Dunya": "dünya",
    "Ekonomi": "ekonomi",
    "KulturSanat": "kültür sanat",
    "Politika": "siyaset",
    "Saglik": "sağlık",
    "Spor": "spor",
    "Turkiye": "Türkiye",
}
EMOTIONS = {"Anger": "kızgın", "Disgust": "iğrenmiş", "Fear": "korkmuş", "Happy": "mutlu", "Sadness": "üzgün", "Surprise": "şaşırmış"}
EMOTION = choice("Bu metin hangi duyguyu anlatıyor?", EMOTIONS.values())


def without_verdict(text):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(s for s in sentences if not VERDICT_WORDS.search(lower(s)))


def stems(text):
    return {word[:5] for word in words(text) if len(word) > 2}


def benchmark_claims():
    claims = [stems(state["iddia"] if isinstance(state, dict) else state) for task in FACT_CHECK_TASKS for state in bench_states(task)]
    index = defaultdict(list)
    for i, claim in enumerate(claims):
        for stem in claim:
            index[stem].append(i)
    return claims, index


def repeats_claim(claim, claims, index):
    # half or more of the word stems of both claims together are shared
    shared = Counter(i for stem in claim for i in index.get(stem, ()))
    return any(n * 2 >= len(claim | claims[i]) for i, n in shared.items())


def build_facturk():
    claims, index = benchmark_claims()
    items = {"heldout": [], "train": []}
    for r in hf_rows("ealtuncu/FACTurk", None, "train"):
        verdict = VERDICTS.get(str(r["normalised_rating"]).lower())
        report = without_verdict(r["content"])
        # Teyit's claims up to 2020 are all in X-FACT
        in_benchmark = r["organisation"] == "Teyit" and str(r["date_published"]) < "2021" or repeats_claim(stems(r["claim"]), claims, index)
        if verdict and report and not in_benchmark and not VERDICT_WORDS.search(lower(r["claim"])):
            split = "heldout" if str(r["date_published"])[:4] in ("2025", "2026") else "train"
            items[split].append({
                "state": {"iddia": r["claim"], "rapor": report},
                "questions": {"verdict": VERDICT},
                "gold": {"verdict": verdict},
                "meta": {"date": r["date_published"], "organisation": r["organisation"]},
            })
    return items


def build_interpress():
    path = download("https://www.interpress.com/downloads/interpress_news_category_tr_270k.zip")
    question = choice(TOPIC, INTERPRESS.values())
    items = [
        {"state": {"başlık": r["Title"], "metin": r["Content"]}, "questions": {"topic": question}, "gold": {"topic": INTERPRESS[r["Category"]]}}
        for name in ["_train.tsv", "_test.tsv"]
        for r in read_tsv(read_zip(path, name))
    ]
    return {"train": items}


def build_kemik_news():
    folder = kaggle_file("oktayozturk010/42000-news-text-in-13-classes", "42bin_haber/news")
    question = choice(TOPIC, KEMIK.values())
    items = []
    for category, topic in KEMIK.items():
        for path in sorted((folder / category).glob("*.txt")):
            title, _, text = path.read_text(encoding="utf-8").strip().partition("\n")
            items.append({"state": {"başlık": title.strip(), "metin": text.strip()}, "questions": {"topic": question}, "gold": {"topic": topic}})
    return {"train": items}


def build_bilcat():
    question = choice(TOPIC, BILCAT.values())
    items = [
        {"state": r["Text"], "questions": {"topic": question}, "gold": {"topic": BILCAT[r["Class"]]}}
        for r in hf_rows("ctoraman/BilCat-news-classification", None, "train")
        if r["Class"] in BILCAT
    ]
    return {"train": items}


def build_tremo():
    items = []
    for doc in ET.parse(kaggle_file("mansuralp/tremo", "TREMODATA.xml")).getroot():
        votes = [EMOTIONS[e.text] for e in doc.find("VoteDistribution") if e.text in EMOTIONS]
        if doc.findtext("ValidatedEmotion") in EMOTIONS:
            items.append({
                "state": doc.findtext("Entry"),
                "questions": {"emotion": EMOTION},
                "gold": {"emotion": EMOTIONS[doc.findtext("ValidatedEmotion")]},
                "soft_gold": {"emotion": {emotion: votes.count(emotion) / len(votes) for emotion in EMOTIONS.values()}},
            })
    return {"train": items}
