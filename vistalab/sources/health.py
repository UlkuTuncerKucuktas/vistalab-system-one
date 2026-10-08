from ..download import hf_rows
from ..text import mask
from .questions import with_distractors

INSTRUCTIONS = "Bu hastanın sorusunu hangi uzmanlık alanındaki bir doktor yanıtlamalı?"


def questions(split):
    return [(r["question_content"], r["doctor_speciality"].replace("-", " ")) for r in hf_rows("alibayram/doktorsitesi", None, split)]


def build_doktorsitesi():
    train, test = questions("train"), questions("test")
    specialties = sorted({specialty for _, specialty in train})

    def items(pairs):
        return [
            {"state": mask(text), "questions": {"specialty": with_distractors(INSTRUCTIONS, specialty, specialties, text)}, "gold": {"specialty": specialty}}
            for text, specialty in pairs
            if specialty in specialties
        ]

    return {"heldout": items(test), "train": items(train)}
