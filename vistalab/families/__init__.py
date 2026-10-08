import re

from ..overlap import texts

WRITERS = ["gemma", "qwen"]
SAME_ASKS = ["Bu iki cümle aynı anlamı mı taşıyor?", "İkinci cümle birincisiyle aynı şeyi mi söylüyor?", "İki cümle de aynı bilgiyi mi veriyor?"]
SAME = {"true": "Evet: İki cümle aynı bilgiyi veriyor; yalnızca söyleyiş farklı.", "false": "Hayır: Cümlelerden biri farklı ya da çelişen bir bilgi veriyor."}


def strict(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


EDIT = strict({"eski": {"type": "string", "minLength": 1, "maxLength": 80}, "yeni": {"type": "string", "minLength": 1, "maxLength": 80}})


def sentences_in(state):
    for text in texts(state):
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            if 10 <= len(sentence.split()) <= 30 and sentence[0].isupper() and sentence.endswith(".") and '"' not in sentence:
                yield sentence


def item(job, k, qid, state, question, **meta):
    return {"id": f"{job['id']}/{k}", "source": "synthetic", "qid": qid, "state": state, "question": question,
            "meta": {"kind": job["kind"], "writer": job["writer"], "job": job["id"], **meta}}
