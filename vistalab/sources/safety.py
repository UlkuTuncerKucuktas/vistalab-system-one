import csv

import openpyxl

from ..download import hf_rows, kaggle_file
from ..text import mask
from .questions import choice, yes_no

HATEMAP_TOPICS = [
    "animals",
    "cities",
    "ethnicity",
    "lgbt",
    "misogyny",
    "occupations",
    "politics",
    "political-orientation",
    "refugees",
    "religion",
    "sects",
    "veganism",
]
HATEMAP_LABELS = ["saldırgan", "nefret", "nötr", "medeni"]
HATEMAP = choice("Bu yorum saldırgan mı, nefret söylemi mi, nötr mü, yoksa medeni mi?", HATEMAP_LABELS)
HATEFUL = {
    "type": "noul",
    "instructions": "Bu metin nefret söylemi içeriyor mu?",
    "criteria": {
        "true": "Birini ya da bir grubu kökeni, dini, cinsiyeti, cinsel yönelimi ya da siyasi görüşü gibi bir kimliği yüzünden aşağılıyor ya da ona saldırıyor.",
        "false": "Nefret söylemi içermiyor.",
    },
}
OFFENSIVE = {
    "type": "noul",
    "instructions": "Bu tweet saldırgan (offensive) bir ifade içeriyor mu?",
    "criteria": {
        "true": "Hakaret, küfür, tehdit, aşağılama ya da hedef gösterme içeriyor.",
        "false": "Saldırgan bir ifade içermiyor.",
    },
}
OFFENSE_TYPES = {"Cinsiyetçilik": "cinsiyetçilik", "Irkçılık": "ırkçılık", "Kızdırma": "kızdırma", "Nötr": "nötr"}
OFFENSE_TYPE = choice("Bu paylaşım cinsiyetçi mi, ırkçı mı, birini kızdırmaya mı çalışıyor, yoksa nötr mü?", OFFENSE_TYPES.values())
HARMS = {
    "Küfür": yes_no("Bu paylaşım küfür içeriyor mu?"),
    "Tehdit": yes_no("Bu paylaşım birine yönelik bir tehdit içeriyor mu?"),
    "Hakaret": yes_no("Bu paylaşım hakaret içeriyor mu?"),
    "Dolandırıcılık": yes_no("Bu paylaşım dolandırıcılık amaçlı mı?"),
    "Cinsel İçerik": yes_no("Bu paylaşım cinsel içerik taşıyor mu?"),
    "Irkçılık": yes_no("Bu paylaşım ırkçı bir ifade içeriyor mu?"),
}
SPAM_EMAIL = {
    "type": "noul",
    "instructions": "Bu e-posta istenmeyen (spam) bir ileti mi?",
    "criteria": {
        "true": "Reklam, kampanya, dolandırıcılık ya da toplu gönderilmiş istenmeyen bir ileti.",
        "false": "Kişiye ya da işe dair normal bir ileti.",
    },
}
SPAM_SMS = {
    "type": "noul",
    "instructions": "Bu SMS istenmeyen (spam) bir mesaj mı?",
    "criteria": {
        "true": "Reklam, kampanya, dolandırıcılık ya da toplu gönderilmiş istenmeyen bir mesaj.",
        "false": "Birine yazılmış normal bir mesaj.",
    },
}
# the labels mark what a tweet is about, not whether it is itself a scam
TOPICS = {
    "dolandırıcılık": yes_no("Bu paylaşım dolandırıcılıkla ilgili mi?"),
    "yasa dışı yatırım tavsiyesi": yes_no("Bu paylaşım yatırım tavsiyesiyle ilgili mi?"),
    "kumar": yes_no("Bu paylaşım kumar ya da bahisle ilgili mi?"),
}


def hatemap_items(split):
    return [
        {"state": {"başlık": r["baslik"].replace("-", " "), "yorum": mask(r["text"])}, "questions": {"hate": HATEMAP}, "gold": {"hate": HATEMAP_LABELS[r["label"]]}}
        for topic in HATEMAP_TOPICS
        for r in hf_rows("turkish-nlp-suite/TurkishHateMap", topic, split)
    ]


def build_hatemap():
    return {"heldout": hatemap_items("test"), "train": hatemap_items("train") + hatemap_items("validation")}


def build_hate_superset():
    items = [
        {"state": mask(r["text"]), "questions": {"hateful": HATEFUL}, "gold": {"hateful": r["labels"] == 1}}
        for r in hf_rows("manueltonneau/turkish-hate-speech-superset", None, "train")
    ]
    return {"train": items}


def build_offensive_nanelimon():
    items = [
        {"state": mask(r["text"]), "questions": {"type": OFFENSE_TYPE}, "gold": {"type": OFFENSE_TYPES[r["label"]]}}
        for r in hf_rows("nanelimon/turkish-social-media-offensive-dataset", None, "train")
    ]
    return {"train": items}


def build_offensive_eymaahner():
    path = kaggle_file("eymaahner/trke-saldrgan-dil-derlemi", "saldirgan_saldirgandegil.csv")
    items = [
        {"state": mask(r["Text"].strip(" ,")), "questions": {"offensive": OFFENSIVE}, "gold": {"offensive": r["Class"] == "1"}}
        for r in csv.DictReader(open(path, encoding="utf-8", newline=""))
    ]
    return {"train": items}


def build_harmful_tweets():
    path = kaggle_file("smailsariteke/trke-tweetlerin-6-snfa-gre-etiketlenmesi", "Veri_Seti_10000.xlsx")
    header, *rows = openpyxl.load_workbook(path, read_only=True).active.iter_rows(values_only=True)
    rows = [dict(zip(header, row)) for row in rows]
    # rows with no label at all, not even Tespit (none of the harms), are left out; _x0092_ is an Excel escape for an apostrophe
    items = [
        {"state": mask(str(r["Paylaşım"]).replace("_x0092_", "'")), "questions": HARMS, "gold": {harm: r[harm] == 1 for harm in HARMS}}
        for r in rows
        if any(r[column] == 1 for column in ["Tespit", *HARMS])
    ]
    return {"train": items}


def build_spam_email_aydemir():
    folder = kaggle_file("emrahaydemr/turkish-mail-dataset-normalspam", "Mails")
    items = [
        {"state": mask(path.read_text(encoding="utf-8").strip()), "questions": {"spam": SPAM_EMAIL}, "gold": {"spam": path.name.startswith("spam")}}
        for path in sorted(folder.glob("*.txt"))
    ]
    return {"train": items}


def build_spam_email_orvile():
    path = kaggle_file("orvile/turkish-spam-v01", "trspam.csv")
    items = [
        {"state": mask(r["Text"].strip()), "questions": {"spam": SPAM_EMAIL}, "gold": {"spam": r["Classification"] == "spam"}}
        for r in csv.DictReader(open(path, encoding="utf-8-sig", newline=""))
        if r["Classification"] in ("ham", "spam")
    ]
    return {"train": items}


def build_spam_sms():
    rows = hf_rows("akuysal/turkishSMS-ds", None, "train") + hf_rows("akuysal/turkishSMS-ds", None, "validation")
    items = [{"state": mask(r["text"]), "questions": {"spam": SPAM_SMS}, "gold": {"spam": r["label"] == "spam"}} for r in rows]
    return {"train": items}


def build_fraud_gambling():
    path = kaggle_file("dorukglen/trke-kumar-and-yatrm-tavsiyesi-veri-seti", "datasetV1.xls")
    items = [
        {"state": mask(r["text"]), "questions": TOPICS, "gold": {topic: r[topic] == "1" for topic in TOPICS}}
        for r in csv.DictReader(open(path, encoding="utf-8", newline=""))
    ]
    return {"train": items}
