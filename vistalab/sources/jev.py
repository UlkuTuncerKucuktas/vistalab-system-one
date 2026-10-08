import json

from ..download import hf_rows

CONTROLS = [
    "amount-extraction-control-v1",
    "citation-control-v1",
    "context-retention-control-v1",
    "email-selection-control-v1",
    "entity-alignment-control-v1",
    "ir-control-v1",
    "mailroom-control-v1",
    "phone-extraction-control-v1",
    "silent-failure-control-v1",
    "sponsor-segment-control-v1",
]


def key_and_description(option):
    key, _, description = option.partition(": ")
    return key, description


def to_item(r):
    kind, options, target = r["kind"], r["options"], r["target"]
    if kind == "noul":
        criteria = {"true": "Yes", "false": "No"}
        soft = {"true": target[options.index("yes")], "false": target[options.index("no")]}
        gold = soft["true"] > 0.5
    elif kind == "score":
        criteria = options
        soft = {str(level): p for level, p in enumerate(target)}
        gold = target.index(max(target))
    else:
        criteria = dict(key_and_description(option) for option in options)
        soft = dict(zip(criteria, target))
        gold = max(soft, key=soft.get)
    return {
        "state": json.loads(r["state_json"]),
        "questions": {"answer": {"type": kind, "instructions": r["question"], "criteria": criteria}},
        "gold": {"answer": gold},
        "soft_gold": {"answer": soft},
        "meta": {"control": r["source"], "language": json.loads(r["metadata_json"]).get("language", "en")},
    }


def build_open_jev():
    items = [
        to_item(r)
        for control in CONTROLS
        for r in hf_rows("OpenAGILab/Jev-dataset", control, "train")
        if json.loads(r["metadata_json"]).get("language") != "zh"
    ]
    return {"train": items}
