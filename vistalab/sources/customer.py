import csv
import re

import pyarrow.parquet as pq

from ..download import hf_file, hf_rows, kaggle_file
from ..overlap import fingerprint
from ..text import lower, mask
from .questions import choice, stars

SECTORS = {
    "alisveris": "alışveriş",
    "anne-bebek": "anne ve bebek",
    "beyaz-esya": "beyaz eşya",
    "bilgisayar": "bilgisayar",
    "cep-telefon-kategori": "cep telefonu",
    "egitim": "eğitim",
    "elektronik": "elektronik",
    "emlak-ve-insaat": "emlak ve inşaat",
    "enerji": "enerji",
    "etkinlik-ve-organizasyon": "etkinlik ve organizasyon",
    "finans": "finans",
    "gida": "gıda",
    "giyim": "giyim",
    "hizmet-sektoru": "hizmet sektörü",
    "icecek": "içecek",
    "internet": "internet",
    "kamu-hizmetleri": "kamu hizmetleri",
    "kargo-nakliyat": "kargo ve nakliyat",
    "kisisel-bakim-ve-kozmetik": "kişisel bakım ve kozmetik",
    "kucuk-ev-aletleri": "küçük ev aletleri",
    "medya": "medya",
    "mekan-ve-eglence": "mekân ve eğlence",
    "mobilya-ev-tekstili": "mobilya ve ev tekstili",
    "mucevher-saat-gozluk": "mücevher, saat ve gözlük",
    "mutfak-arac-gerec": "mutfak araç gereçleri",
    "otomotiv": "otomotiv",
    "saglik": "sağlık",
    "sigortacilik": "sigortacılık",
    "spor": "spor",
    "temizlik": "temizlik",
    "turizm": "turizm",
    "ulasim": "ulaşım",
}
SECTOR = choice("Bu şikâyet hangi sektördeki bir firmayla ilgili?", SECTORS.values())
PLACE = {
    "rating": stars("Bu yorumu yazan kişi mekâna kaç yıldız vermiştir?"),
    "food": stars("Bu yorumu yazan kişi yemeklere kaç yıldız vermiştir?"),
    "service": stars("Bu yorumu yazan kişi hizmete kaç yıldız vermiştir?"),
    "atmosphere": stars("Bu yorumu yazan kişi ortama kaç yıldız vermiştir?"),
}
PRODUCT = stars("Bu yorumu yazan müşteri ürüne kaç yıldız vermiştir?")
TRENDYOL_CATEGORY = choice("Bu yorum hangi kategorideki bir ürün hakkında?", ["Aksesuar", "Giyim", "Kozmetik & Kişisel Bakım", "Elektronik", "Kitap"])
POLARITY = {"positive": "olumlu", "negative": "olumsuz", "neutral": "nötr"}
LAVOIR_WORKFLOWS = [
    "banka_destek",
    "belediye_basvuru",
    "bt_destek",
    "e_devlet_destek",
    "e_ticaret_iade",
    "ik_talepleri",
    "kvkk_talepleri",
    "operator_destek",
    "seyahat_degisiklik",
    "sgk_islemleri",
    "sigorta_hasar",
]
SPEAKERS = {"user": "Müşteri", "system": "Asistan"}


def ten_points(instructions):
    return {"type": "score", "instructions": instructions, "criteria": [f"{n}/10" for n in range(1, 11)]}


DELIVERY = {
    "speed": ten_points("Bu yorumu yazan müşteri teslimat hızına 10 üzerinden kaç puan vermiştir?"),
    "service": ten_points("Bu yorumu yazan müşteri servise 10 üzerinden kaç puan vermiştir?"),
    "taste": ten_points("Bu yorumu yazan müşteri lezzete 10 üzerinden kaç puan vermiştir?"),
}


def held_out_company(text):
    # the first word of a complaint title is nearly always the company ("Pegasus'ta", "Cinemaximum"), so 1 in 20
    # companies is held out
    company = re.split(r"[\s'’]", lower(text.strip()))[0]
    return int(fingerprint([company]), 16) % 20 == 0


def build_tc32():
    items = {"heldout": [], "train": []}
    path = kaggle_file("savasy/multiclass-classification-data-for-turkish-tc32", "ticaret-yorum.csv")
    for r in csv.DictReader(open(path, encoding="utf-8", newline="")):
        text = mask(re.sub(r'\s*Devamını oku"?$', "", r["text"]))
        split = "heldout" if held_out_company(text) else "train"
        items[split].append({"state": text, "questions": {"sector": SECTOR}, "gold": {"sector": SECTORS[r["category"]]}})
    return items


def build_google_maps():
    items = []
    path = hf_file("opdullah/turkish-google-maps-reviews", "train.csv")
    for r in csv.DictReader(open(path, encoding="utf-8", newline="")):
        scores = {"rating": r["review_rating"], "food": r["food_rating"], "service": r["service_rating"], "atmosphere": r["atmosphere_rating"]}
        scores = {q: int(float(score)) - 1 for q, score in scores.items() if score}
        items.append({
            "state": mask(r["review_text"]),
            "questions": {q: PLACE[q] for q in scores},
            "gold": scores,
            "meta": {"place_category": r["place_category"]},
        })
    return {"train": items}


def build_yemeksepeti():
    path = kaggle_file("bilgejinnen/yemeksepeti-comments", "yorumsepeti.parquet")
    items = [
        {
            "state": mask(r["yorum"]),
            "questions": DELIVERY,
            "gold": {"speed": int(r["hiz"]) - 1, "service": int(r["servis"]) - 1, "taste": int(r["lezzet"]) - 1},
        }
        for r in pq.read_table(path).to_pylist()
    ]
    return {"train": items}


def build_trendyol():
    items = [
        {
            "state": mask(r["comment"]),
            "questions": {"rating": PRODUCT, "category": TRENDYOL_CATEGORY},
            "gold": {"rating": r["point"] - 1, "category": r["category"]},
        }
        for r in hf_rows("Doukan/trendyol-turkish-product-reviews", None, "train")
    ]
    return {"train": items}


def build_vitamins():
    items = [
        {"state": mask(r["text"]), "questions": {"rating": PRODUCT}, "gold": {"rating": r["star"] - 1}}
        for r in hf_rows("turkish-nlp-suite/vitamins-supplements-reviews", None, "train")
    ]
    return {"train": items}


def build_absa():
    items = []
    for r in hf_rows("ytu-ce-cosmos/absa-tr", None, "train"):
        polarity = {a["aspect"]: POLARITY[a["polarity"]] for a in r["aspects"]}
        questions = {aspect: choice(f"Bu yorum {aspect} hakkında olumlu mu, olumsuz mu, yoksa nötr mü?", POLARITY.values()) for aspect in polarity}
        if polarity:
            items.append({"state": r["text"], "questions": questions, "gold": polarity, "meta": {"domain": r["domain"]}})
    return {"train": items}


def build_product_category():
    products = []
    for r in hf_rows("Hulusiaa/turkish-product-category-finetune-openai", None, "train"):
        product = re.search(r"\{([^}]*)\}", r["messages"][1]["content"]).group(1).strip()
        path = re.search(r"kategori \{([^}]*)\}", r["messages"][2]["content"]).group(1)
        products.append((product, path))
    question = choice("Bu ürün hangi ana kategoriye girer?", sorted({path.split(">")[0].strip() for _, path in products}))
    items = [
        {"state": product, "questions": {"category": question}, "gold": {"category": path.split(">")[0].strip()}, "meta": {"path": path}}
        for product, path in products
    ]
    return {"train": items}


def lavoir_items(split):
    return [
        {
            "state": "\n".join(f"{SPEAKERS[turn['role']]}: {turn['text']}" for turn in r["state"]),
            "questions": {"route": r["question"]},
            # the dataset's gold is the route of one hidden customer profile; the text only supports the most likely route
            "gold": {"route": max(r["target"], key=r["target"].get)},
            "soft_gold": {"route": r["target"]},
            "meta": {"workflow": workflow, "kind": r["kind"]},
        }
        for workflow in LAVOIR_WORKFLOWS
        for r in hf_rows("moganai/lavoir-dialogues-tr", workflow, split)
    ]


def build_lavoir():
    return {"heldout": lavoir_items("test_seen"), "train": lavoir_items("train")}
