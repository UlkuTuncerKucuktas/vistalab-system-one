import csv

from ..download import hf_file

STANCE = {
    "type": "choice",
    "instructions": "Bu yorumu yazan editör, Vikipedi'deki silme tartışmasında hangi görüşü savunuyor?",
    "criteria": {
        "silinsin": "Sayfanın silinmesini istiyor.",
        "kalsın": "Sayfanın kalmasını istiyor.",
        "aktarılsın": "Sayfanın başka bir maddeye aktarılmasını ya da onunla birleştirilmesini istiyor.",
        "yorum": "Oy vermeden yalnızca yorum yapıyor.",
    },
}


def build_wiki_stance():
    items = [
        {"state": {"tartışma": r["topic"], "yorum": r["comment"]}, "questions": {"stance": STANCE}, "gold": {"stance": r["decision"]}}
        for split in ["train", "val", "test"]
        for r in csv.DictReader(open(hf_file("copenlu/wiki-stance", f"tr-{split}.tsv"), encoding="utf-8"), delimiter="\t")
    ]
    return {"train": items}
