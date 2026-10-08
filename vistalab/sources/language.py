import re

from ..download import hf_rows
from .questions import yes_no

TRGLUE = "turkish-nlp-suite/TrGLUE"
POSITIVE = yes_no("Bu yorum olumlu mu?")
PARAPHRASE = yes_no("Bu iki cümle aynı anlama mı geliyor?")
DUPLICATE = yes_no("Bu iki soru aynı şeyi mi soruyor?")
ENTAILS = yes_no("Birinci cümle doğruysa ikinci cümle de kesinlikle doğru olur mu?")
ANSWERS = yes_no("Bu cümle sorunun cevabını içeriyor mu?")
HAS_ERROR = yes_no("Bu cümlede yazım ya da dilbilgisi hatası var mı?")
SIMILARITY = {
    "type": "score",
    "instructions": "İki cümle anlamca ne kadar benzer?",
    "criteria": [
        "0: İki cümle tamamen farklı konular hakkında.",
        "1: Anlamları farklı ama aynı konudan bahsediyorlar.",
        "2: Anlamları farklı ama bazı ayrıntıları paylaşıyorlar.",
        "3: Kabaca aynı anlamdalar ama önemli bir bilgi farklı ya da eksik.",
        "4: Büyük ölçüde aynı anlamdalar; yalnızca önemsiz ayrıntılar farklı.",
        "5: Tamamen aynı anlama geliyorlar.",
    ],
}


def pair(r, a, b):
    return {"cümle_1": r[a], "cümle_2": r[b]}


def build_trglue():
    # no MNLI or CoLA: the benchmark's jev_noul tasks are made from TrGLUE's validation and test sets, including MNLI,
    # and CoLA's minimal pairs cross its splits
    rows = {config: hf_rows(TRGLUE, config, "train") for config in ["sst2", "mrpc", "qqp", "rte", "qnli", "stsb"]}
    items = [{"state": r["sentence"], "questions": {"positive": POSITIVE}, "gold": {"positive": r["label"] == 1}} for r in rows["sst2"]]
    items += [{"state": pair(r, "sentence1", "sentence2"), "questions": {"paraphrase": PARAPHRASE}, "gold": {"paraphrase": r["label"] == 1}} for r in rows["mrpc"]]
    items += [{"state": pair(r, "question1", "question2"), "questions": {"duplicate": DUPLICATE}, "gold": {"duplicate": r["label"] == 1}} for r in rows["qqp"]]
    items += [{"state": pair(r, "sentence1", "sentence2"), "questions": {"entails": ENTAILS}, "gold": {"entails": r["label"] == 0}} for r in rows["rte"]]
    items += [
        {"state": {"soru": re.sub(r"^\d+\.\s*", "", r["question"]), "cümle": r["sentence"]}, "questions": {"answers": ANSWERS}, "gold": {"answers": r["label"] == 0}}
        for r in rows["qnli"]
    ]
    items += [
        {"state": pair(r, "sentence1", "sentence2"), "questions": {"similarity": SIMILARITY}, "gold": {"similarity": int((r["label"] - 1) * 1.25 + 0.5)}, "meta": {"score": r["label"]}}
        for r in rows["stsb"]
    ]
    return {"train": items}


def build_gecturk():
    items = [
        {"state": " ".join(r["tokens"]), "questions": {"error": HAS_ERROR}, "gold": {"error": any(r["labels"])}}
        for r in hf_rows("GGLab/GECTurk", None, "train")
    ]
    return {"train": items}


def build_plu_steps():
    items = []
    for r in hf_rows("gglab-ku/turkish-plu-step-inference", None, "train"):
        options = dict(zip("ABCD", [r["ending0"], r["ending1"], r["ending2"], r["ending3"]]))
        question = {"type": "choice", "instructions": "Bu amaca ulaşmak için atılması gereken adım hangisidir?", "criteria": options}
        items.append({"state": r["sent2"], "questions": {"step": question}, "gold": {"step": "ABCD"[r["label"]]}})
    return {"train": items}
