import re

from ..overlap import words
from . import EDIT, SAME, SAME_ASKS, WRITERS, item, sentences_in, strict

MEANING_SENTENCES = 4000
SIMILARITY_SENTENCES = 1200
MEANING_PROMPT = """Türkçe için anlam alıştırmaları hazırlıyorsun. Doğal, düzgün Türkçe yaz. Yalnızca istenen JSON'u yaz.

Cümle: {sentence}

1. "küçük_aynı": Cümlede bir ya da iki sözcüğü eş anlamlısıyla değiştir; anlam tamamen aynı kalsın.
2. "küçük_farklı": Cümlede bir ya da iki yeri değiştirerek anlamı değiştir: iki adın ya da sayının yerini değiştir, kimin kime ne yaptığını tersine çevir, bir sayıyı, zamanı ya da yeri değiştir. Cümle yine doğal ve dilbilgisi açısından doğru kalsın.
3. "yeni_aynı": Cümleyi baştan yeniden yaz: başka sözcükler ve başka bir yapı kullan, ama hiçbir bilgiyi ekleme, çıkarma ya da değiştirme.
4. "yeni_farklı": Cümleyi baştan yeniden yaz ve yalnızca tek bir bilgiyi değiştir: bir sayıyı, bir adı, bir yeri, bir zamanı ya da kimin ne yaptığını.
Değişikliklerde her "eski", cümleden harfi harfine kopyalanmış, cümlede yalnızca bir kez geçen ve tam sözcüklerden oluşan bir parçadır; "yeni" onun yerine gelir. İki parçanın yerini değiştirmek için iki değişiklik yaz ("Ahmet" yerine "Mehmet" ve "Mehmet" yerine "Ahmet").

Çıktı:
{{"küçük_aynı":[{{"eski":"…","yeni":"…"}}],"küçük_farklı":[{{"eski":"…","yeni":"…"}}],"yeni_aynı":"…","yeni_farklı":"…"}}"""
SIMILARITY_PROMPT = """Türkçe için anlam benzerliği alıştırmaları hazırlıyorsun. Doğal, düzgün Türkçe yaz. Yalnızca istenen JSON'u yaz.

Cümle: {sentence}

Bu cümleye göre beş yeni cümle yaz:
"5": aynı bilgiyi başka sözcüklerle veren bir cümle;
"4": aynı bilgiyi veren ama önemsiz bir ayrıntısı değişmiş bir cümle;
"3": aynı olayı anlatan ama önemli bir bilgisi değişmiş bir cümle;
"2": aynı konudan, yalnızca bir ayrıntıyı paylaşan başka bir cümle;
"1": aynı konudan ama başka bir şey anlatan bir cümle.

Çıktı:
{{"5":"…","4":"…","3":"…","2":"…","1":"…"}}"""
SIMILARITY_ASKS = ["İki cümlenin verdiği bilgi ne kadar örtüşüyor?", "Cümleler anlam bakımından birbirine ne kadar benziyor?"]
SIMILARITY = [
    "0: Cümleler başka konulardan söz ediyor.",
    "1: Cümleler aynı konudan söz ediyor ama başka şeyler anlatıyor.",
    "2: Cümleler yalnızca bir ayrıntıyı paylaşıyor.",
    "3: Cümleler aynı olayı anlatıyor ama önemli bir bilgide ayrılıyor.",
    "4: Cümleler aynı bilgiyi veriyor; yalnızca önemsiz bir ayrıntı farklı.",
    "5: Cümleler aynı bilgiyi veriyor.",
]
SENTENCE = {"type": "string", "minLength": 10, "maxLength": 500}


def sentence_pool(seeds, rng):
    # round 2's own order of the sources, not round 1's
    return [s for source in ["tr_news", "kemik_news", "bilcat", "wikipedia"] for row in rng.sample(seeds[source], min(3000, len(seeds[source]))) for s in sentences_in(row["state"])]


def meaning_jobs(sentences, rng):
    schema = strict({"küçük_aynı": {"type": "array", "items": EDIT, "minItems": 1, "maxItems": 2}, "küçük_farklı": {"type": "array", "items": EDIT, "minItems": 1, "maxItems": 2},
                     "yeni_aynı": SENTENCE, "yeni_farklı": SENTENCE})
    out = []
    for n, sentence in enumerate(rng.sample(sentences, MEANING_SENTENCES)):
        out.append({"id": f"anlam/{n}", "kind": "anlam", "writer": WRITERS[n % 2], "prompt": MEANING_PROMPT.format(sentence=sentence), "schema": schema, "max_tokens": 1000, "sentence": sentence})
    return out


def similarity_jobs(sentences, rng):
    # the writer's sentences are levels 5 to 1, and "other", a sentence drawn for each job, is level 0
    schema = strict({level: SENTENCE for level in "54321"})
    out = []
    for n, sentence in enumerate(rng.sample(sentences, SIMILARITY_SENTENCES)):
        out.append({"id": f"benzerlik/{n}", "kind": "benzerlik", "writer": WRITERS[n % 2], "prompt": SIMILARITY_PROMPT.format(sentence=sentence), "schema": schema,
                    "max_tokens": 1000, "sentence": sentence, "other": rng.choice(sentences)})
    return out


def edited(text, changes):
    # each change replaces a span that occurs once in the sentence and starts and ends at word edges; the spans are found in
    # the sentence as written and replaced together, so two names or numbers can swap places, and they may not overlap
    spans = []
    for change in changes:
        start = text.find(change["eski"])
        end = start + len(change["eski"])
        whole = start >= 0 and (start == 0 or not re.match(r"[\w']", text[start - 1])) and (end == len(text) or not re.match(r"[\w']", text[end]))
        if text.count(change["eski"]) != 1 or not whole or change["eski"] == change["yeni"]:
            return None
        spans.append((start, end, change["yeni"]))
    spans.sort()
    if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
        return None
    for start, end, new in reversed(spans):
        text = text[:start] + new + text[end:]
    return text


def meaning_questions(job, output, rng):
    sentence = job["sentence"]
    built = [
        ("küçük", edited(sentence, output["küçük_aynı"]), "true"),
        ("küçük", edited(sentence, output["küçük_farklı"]), "false"),
        ("yeni", output["yeni_aynı"].strip(), "true"),
        ("yeni", output["yeni_farklı"].strip(), "false"),
    ]
    for k, (edit, other, expected) in enumerate(built):
        if other is not None and words(other) != words(sentence):
            first, second = (sentence, other) if rng.random() < 0.5 else (other, sentence)
            question = {"type": "noul", "instructions": rng.choice(SAME_ASKS), "criteria": SAME}
            yield item(job, k, "anlam", {"birinci cümle": first, "ikinci cümle": second}, question, expected=expected, edit=edit)


def similarity_questions(job, output, rng):
    pairs = [(output[level].strip(), level) for level in "54321"] + [(job["other"], "0")]
    for k, (other, level) in enumerate(pairs):
        if words(other) != words(job["sentence"]):
            first, second = (job["sentence"], other) if rng.random() < 0.5 else (other, job["sentence"])
            question = {"type": "score", "instructions": rng.choice(SIMILARITY_ASKS), "criteria": SIMILARITY}
            yield item(job, k, "benzerlik", {"birinci cümle": first, "ikinci cümle": second}, question, expected=level)
