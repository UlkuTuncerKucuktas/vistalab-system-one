import re

from ..download import hf_sample
from ..text import lower

APPEAL = {
    "type": "choice",
    "instructions": "Yüksek mahkeme bu temyiz incelemesinde alt mahkemenin kararını onamış mıdır, bozmuş mudur?",
    "criteria": {"onama": "Alt mahkemenin kararı onandı.", "bozma": "Alt mahkemenin kararı bozuldu."},
}
VERDICTS = {"onanmasına": "onama", "bozulmasına": "bozma"}
DISSENT = re.compile(r"\n[^\n]{0,12}(KARŞI ?OY|MUHALEFET ŞERHİ|AZLIK OYU)")
# the reporting judge's and the prosecutor's opinions name the outcome, which the court then mostly follows ("onama isteyen
# tebliğname", "onanması gerektiği düşünülmektedir")
OPINION = re.compile(r"^.*(DÜŞÜNCESİ|[Dd]üşüncesi|TEBLİĞNAME GÖRÜŞÜ|[Tt]ebliğname görüşü|düşünülmektedir|(onama|bozma) isteyen).*\n?", re.M)


def part_of_verdict(paragraph):
    # headings ("KARAR SONUCU:", "HÜKÜM") and numbered orders ("1. temyiz isteminin reddine,") before the verdict sentence
    p = lower(paragraph).strip()
    return len(p.split()) < 5 or p.endswith(("ne,", "na,", "ne;", "na;")) or p.startswith(("açıklanan", "yukarıda açıklanan")) or "karar sonucu" in p


def split_verdict(text):
    # a dissent follows the verdict and may argue for the other outcome, so it is cut off first; the outcome is read from the
    # whole verdict block, so a decision that upholds one part and overturns another has both
    paragraphs = DISSENT.split(text)[0].strip().split("\n")
    for i in reversed(range(len(paragraphs))):
        if any(word in lower(paragraphs[i]) for word in VERDICTS):
            while i and part_of_verdict(paragraphs[i - 1]):
                i -= 1
            block = lower("\n".join(paragraphs[i:]))
            return "\n".join(paragraphs[:i]).strip(), {verdict for word, verdict in VERDICTS.items() if word in block}
    return "", set()


def build_court_outcome():
    items = []
    for court in ["yargitay", "danistay"]:
        for r in hf_sample("mrfg/turkish-court-decisions", court, "train", 100000):
            reasoning, verdicts = split_verdict(r["text"])
            reasoning = OPINION.sub("", reasoning)
            # a short text is only the header; a verdict word left in the reasoning comes from the case history
            if len(reasoning.split()) >= 80 and len(verdicts) == 1 and not any(word in lower(reasoning) for word in VERDICTS):
                items.append({
                    "state": reasoning,
                    "questions": {"outcome": APPEAL},
                    "gold": {"outcome": verdicts.pop()},
                    "meta": {"court": r["court"], "date": r["karar_tarihi"]},
                })
    return {"train": items}
