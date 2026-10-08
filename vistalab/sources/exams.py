from ..download import hf_rows
from .questions import with_distractors

def build_thesis():
    rows = hf_rows("boun-tabilab/Thesis-Abstract-Classification-11K", None, "train")
    fields = sorted({r["label"] for r in rows})
    items = [
        {"state": r["text"], "questions": {"field": with_distractors("Bu tez özeti hangi bilim alanına ait?", r["label"], fields, r["text"])}, "gold": {"field": r["label"]}}
        for r in rows
    ]
    return {"train": items}
