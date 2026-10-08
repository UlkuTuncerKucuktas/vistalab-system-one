import re

from ..overlap import words
from . import EDIT, SAME, SAME_ASKS, WRITERS, item, sentences_in, strict

TRAP_SENTENCES = 7000
TRAP_PROMPT = """Türkçe için dilbilgisi ve anlam alıştırmaları hazırlıyorsun. Doğal, düzgün Türkçe yaz. Yalnızca istenen JSON'u yaz.

Cümle: {sentence}

1. "aynı_anlam": Cümleyi aynı anlamı koruyarak yeniden yaz: sözcüklerin bir kısmını eş anlamlılarıyla değiştir ya da cümleyi başka bir yapıyla kur. Bilgi ekleme, çıkarma ya da değiştirme.
2. "anlam_değişikliği": Cümlenin anlamını en fazla iki yeri değiştirerek değiştir: kimin kime ne yaptığını yer değiştir, bir sayıyı, zamanı ya da yeri değiştir, olumluyu olumsuz yap. Cümle yine doğal ve dilbilgisi açısından doğru kalsın.
3. "dilbilgisi_hatası": Cümlede tek bir yeri değiştirerek bir dilbilgisi hatası yap: yanlış hâl eki, kişi ya da sayı uyumsuzluğu, yanlış zaman eki ya da yanlış sözcük sırası. Yazım hatası yapma; hata dilbilgisinde olsun.
Değişikliklerde her "eski", cümleden harfi harfine kopyalanmış, cümlede yalnızca bir kez geçen ve tam sözcüklerden oluşan bir parçadır; "yeni" onun yerine gelir.

Çıktı:
{{"aynı_anlam":"…","anlam_değişikliği":[{{"eski":"…","yeni":"…"}}],"dilbilgisi_hatası":[{{"eski":"…","yeni":"…"}}]}}"""
PAIR_ASKS = ["Hangi cümle Türkçenin dilbilgisi kurallarına uygun?", "Dilbilgisi hatası olmayan cümle hangisi?"]
CORRECT = ("Bu cümle Türkçenin dilbilgisi kurallarına uyuyor mu?", {"true": "Evet: Cümlede dilbilgisi hatası yok.", "false": "Hayır: Cümlede yanlış bir ek, uyumsuzluk ya da yanlış bir sözcük sırası var."})
WRONG = ("Bu cümlede bir dilbilgisi hatası var mı?", {"true": "Evet: Cümlede yanlış bir ek, uyumsuzluk ya da yanlış bir sözcük sırası var.", "false": "Hayır: Cümlede dilbilgisi hatası yok."})


def jobs(seeds, rng):
    pool = [s for source in ["wikipedia", "tr_news", "kemik_news", "bilcat"] for row in rng.sample(seeds[source], min(3000, len(seeds[source]))) for s in sentences_in(row["state"])]
    schema = strict({"aynı_anlam": {"type": "string", "minLength": 10, "maxLength": 500}, "anlam_değişikliği": {"type": "array", "items": EDIT, "minItems": 1, "maxItems": 2},
                     "dilbilgisi_hatası": {"type": "array", "items": EDIT, "minItems": 1, "maxItems": 1}})
    out = []
    for n, sentence in enumerate(rng.sample(pool, TRAP_SENTENCES)):
        out.append({"id": f"trap/{n}", "kind": "trap", "writer": WRITERS[n % 2], "prompt": TRAP_PROMPT.format(sentence=sentence), "schema": schema, "max_tokens": 600, "sentence": sentence})
    return out


def edited(text, changes):
    # each change replaces a span that occurs once and starts and ends at word edges; the changes are made in turn, each in
    # the sentence the ones before it left, so two names or numbers cannot swap places
    for change in changes:
        start = text.find(change["eski"])
        end = start + len(change["eski"])
        whole = start >= 0 and (start == 0 or not re.match(r"[\w']", text[start - 1])) and (end == len(text) or not re.match(r"[\w']", text[end]))
        if text.count(change["eski"]) != 1 or not whole or change["eski"] == change["yeni"]:
            return None
        text = text.replace(change["eski"], change["yeni"])
    return text


def questions(job, output, rng):
    sentence = job["sentence"]
    same, changed, broken = output["aynı_anlam"].strip(), edited(sentence, output["anlam_değişikliği"]), edited(sentence, output["dilbilgisi_hatası"])
    pairs = [(same, "true")] if words(same) != words(sentence) else []
    pairs += [(changed, "false")] if changed else []
    for other, expected in pairs:
        first, second = (sentence, other) if rng.random() < 0.5 else (other, sentence)
        question = {"type": "noul", "instructions": rng.choice(SAME_ASKS), "criteria": SAME}
        yield item(job, expected, "anlam", {"birinci cümle": first, "ikinci cümle": second}, question, expected=expected)
    if broken and rng.random() < 0.5:
        first, second = (sentence, broken) if rng.random() < 0.5 else (broken, sentence)
        question = {"type": "choice", "instructions": rng.choice(PAIR_ASKS), "criteria": {"birinci": "Birinci cümle", "ikinci": "İkinci cümle"}}
        yield item(job, "pair", "dilbilgisi", {"birinci cümle": first, "ikinci cümle": second}, question, expected="birinci" if first == sentence else "ikinci")
    elif broken:
        shown, grammatical = rng.choice([(sentence, True), (broken, False)])
        ask = rng.choice([CORRECT, WRONG])
        instructions, criteria = ask
        yield item(job, "single", "dilbilgisi", shown, {"type": "noul", "instructions": instructions, "criteria": criteria}, expected=str(grammatical == (ask is CORRECT)).lower())
