from ..download import hf_rows, hf_sample
from ..text import mask


def build_yorumbudur():
    rows = hf_sample("alibayram/yorumbudur", None, "train", 200000)
    return {"seed": [{"state": {"başlık": r["Baslik"], "yorum": mask(r["Yorum"])}, "meta": {"rating": r["Puan"]}} for r in rows]}


def build_financial_news():
    rows = hf_rows("ituperceptron/turkish-financial-sentiment-256k", None, "train")
    return {"seed": [{"state": {"başlık": r["konu"], "metin": r["text"]}, "meta": {"date": r["tarih"]}} for r in rows]}


def build_kap():
    return {"seed": [{"state": r["messages"][0]["content"]} for r in hf_rows("finansai/kap-turkish-financial-sentiment", None, "train")]}


def build_thy_tweets():
    return {"seed": [{"state": mask(" ".join(r["text"].split()))} for r in hf_rows("trmteb/thy_sa", None, "test")]}


def build_tr_news():
    rows = [r for split in ["train", "validation", "test"] for r in hf_rows("batubayk/TR-News", None, split)]
    return {"seed": [{"state": {"başlık": r["title"], "metin": r["content"]}, "meta": {"topic": r["topic"], "source": r["source"]}} for r in rows]}


def build_parliament():
    rows = hf_sample("boun-tabilab/turkish_parliamentary_data", "pages", "train", 200000)
    items = [
        {"state": r["text"], "meta": {"term": r["term"], "year": r["year"], "document": r["document_id"]}}
        for r in rows
        if r["language"] != "ottoman_turkish"
    ]
    return {"seed": items}


def build_wikipedia():
    return {"seed": [{"state": {"başlık": r["title"], "metin": r["text"]}} for r in hf_rows("wikimedia/wikipedia", "20231101.tr", "train")]}
