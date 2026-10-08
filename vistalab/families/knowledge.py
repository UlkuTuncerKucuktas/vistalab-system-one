import re
from collections import Counter, defaultdict
from functools import cache

from .. import read
from ..overlap import BENCH, bench_fingerprints, copies, stems, texts
from . import WRITERS, item, strict

KNOWLEDGE_PASSAGES = 17000
QUESTIONS_PER_PASSAGE = 3
EXAM_WINDOWS = 12000
# article starts about a living person, a place, a work, a sport, a contest, a song or a game, whose questions turn into dates and
# trivia
TRIVIA = re.compile(r"\(d\. |doğumlu|ilçesidir|köyüdür|mahallesidir|filmidir|albümü|dizisi|oyuncu|şarkı|futbol|basketbol|voleybol|hentbol|tenis|güreş|boksör|atlet|sporcu|sezonu|ligi|turnuva|"
                    r"şampiyona|yarışması|olimpiyat|kupası|grand prix|video oyunu", re.IGNORECASE)
# where an article's references, notes and links start
REFERENCES = re.compile(r"\n(?:Kaynakça|Kaynaklar|Dış bağlantılar|Ayrıca bakınız|Notlar|Dipnotlar|Bibliyografya)\n")
EXAMS = ["global_mmlu_tr", "turkishmmlu_tr", "include_tr", "tus21_tr", "jevbench_mmlu_tr", "jevbench_arc_challenge_tr"]
NUMERALS = ["I", "II", "III"]
COMBINATIONS = {"Yalnız I": {0}, "Yalnız II": {1}, "Yalnız III": {2}, "I ve II": {0, 1}, "I ve III": {0, 2}, "II ve III": {1, 2}, "I, II ve III": {0, 1, 2}}
KNOWLEDGE_PROMPT = """Türkçe genel kültür ve okul sınavları için çoktan seçmeli sorular yazıyorsun. Doğal, düzgün Türkçe yaz. Yalnızca istenen JSON'u yaz.

Aşağıdaki metinden {n} soru yaz.
<metin>
{passage}
</metin>

Kurallar
1. Her soru, iyi eğitimli birinin bilmesi beklenen ve metinde açıkça yazan bir bilgiyi sorsun: önemli olaylar, kişiler, yerler, tarihler, kavramlar, nedenler ve sonuçlar.
2. Soru metni görmeden anlaşılsın: "metne göre", "yukarıdaki", "bu maddede" gibi ifadeler kullanma; özel adları ve bağlamı soruya yaz.
3. Çok ayrıntılı ya da önemsiz bilgileri sorma (küçük bir yerin nüfusu, az bilinen birinin doğum günü, bir listenin kaçıncı maddesi olduğu).
4. Dört seçenek yaz. Biri doğru, üçü aynı türden, akla yatkın ama yanlış olsun (hepsi yıl, hepsi şehir ya da hepsi kişi). "Hepsi", "hiçbiri" gibi seçenekler kullanma; doğru seçenek ötekilerden uzun ya da ayrıntılı olmasın.
5. "doğru" doğru seçeneğin sırasıdır (0'dan başlayarak). "kanıt" cevabı veren parçadır, metinden harfi harfine kopyalanmış.

Çıktı:
{{"sorular":[{{"soru":"…","seçenekler":["…","…","…","…"],"doğru":0,"kanıt":"…"}}]}}"""
EXAM_PROMPT = """Türkiye'deki sınavlar (YKS, KPSS, ALES, TUS) için çoktan seçmeli sorular yazıyorsun. Doğal, düzgün Türkçe yaz. Yalnızca istenen JSON'u yaz.

Aşağıdaki metindeki bilgiyle, metni görmeyen ama konuyu bilen birinin cevaplayabileceği üç soru yaz.
<metin>
{passage}
</metin>

1. "düz": Tek doğru cevabı olan bir soru; "doğru" o cevap, "yanlışlar" akla yatkın ama yanlış üç ya da dört seçenek.
2. "değildir": "Aşağıdakilerden hangisi ... söylenemez?" ya da "... değildir?" biçiminde bir soru; "doğrular" metne göre doğru üç ya da dört ifade, "yanlış" metne göre yanlış tek ifade.
3. "öncüller": "... hangileri doğrudur?" biçiminde bir soru ve üç önerme; her önermenin doğru olup olmadığını "doğru" alanında belirt. En az bir önerme doğru olsun.

Kurallar
1. Kavramları, ilkeleri, süreçleri, neden-sonuç ilişkilerini ve önemli olayları sor. Doğum tarihi, nüfus, bölüm adı, maç skoru gibi ezber ayrıntılarını sorma.
2. Soru metni görmeden anlaşılsın: "metne göre", "yukarıdaki", "bu maddede" gibi ifadeler kullanma; gereken bağlamı soruya yaz.
3. Seçenekler aynı türden, akla yatkın ve birbirinden açıkça ayrılabilir olsun. "Hepsi", "hiçbiri" gibi seçenekler kullanma; doğru seçenek ötekilerden uzun ya da ayrıntılı olmasın.
4. Doğru cevap metindeki bilgiyle kesinleşsin; tartışmalı ya da yoruma açık soru yazma.

Çıktı:
{{"düz":{{"soru":"…","doğru":"…","yanlışlar":["…","…","…"]}},"değildir":{{"soru":"…","doğrular":["…","…","…"],"yanlış":"…"}},"öncüller":{{"soru":"…","önermeler":[{{"metin":"…","doğru":true}},{{"metin":"…","doğru":false}},{{"metin":"…","doğru":true}}]}}}}"""
KNOWLEDGE_ASKS = ["Doğru seçenek hangisi?", "Hangisi doğrudur?", "Doğru cevabı seç.", "Bu soruyu hangi seçenek doğru yanıtlıyor?"]
CONTEXT = re.compile(r"metne göre|metinde|yukarıdaki|bu maddede|bu metin", re.I)
STEM = {"type": "string", "minLength": 15, "maxLength": 400}
OPTION = {"type": "string", "minLength": 1, "maxLength": 200}


def passage(text, size):
    # the start of a text, ending at a sentence
    head = text[:size]
    return head[: head.rfind(".") + 1] or head


def body(text):
    match = REFERENCES.search(text)
    return text[: match.start()] if match else text


def prose(text):
    # the share of a text in lines of 100 characters or more, which headings, lists and tables lack
    return sum(len(line) for line in text.split("\n") if len(line) >= 100) / len(text)


def window(text, rng):
    # a stretch of prose from the start of a random paragraph, ending at a sentence; None when every stretch is mostly headings,
    # lists or tables
    starts = [0] + [m.end() for m in re.finditer(r"\n\n", text) if len(text) - m.end() >= 750]
    rng.shuffle(starts)
    return next((p for p in (passage(text[start:], 1500) for start in starts) if prose(p) >= 0.8), None)


def passage_jobs(seeds, rng):
    articles = [row for row in seeds["wikipedia"] if len(row["state"]["metin"]) >= 2500]
    question = strict({"soru": STEM, "seçenekler": {"type": "array", "items": {"type": "string", "minLength": 1, "maxLength": 150}, "minItems": 4, "maxItems": 4},
                       "doğru": {"type": "integer", "minimum": 0, "maximum": 3}, "kanıt": {"type": "string", "minLength": 5, "maxLength": 500}})
    schema = strict({"sorular": {"type": "array", "items": question, "minItems": QUESTIONS_PER_PASSAGE, "maxItems": QUESTIONS_PER_PASSAGE}})
    out = []
    for n, row in enumerate(rng.sample(articles, KNOWLEDGE_PASSAGES * 3 // 4) + rng.sample(seeds["tr_news"], KNOWLEDGE_PASSAGES // 4)):
        text = passage(f"{row['state']['başlık']}. {row['state']['metin']}", 1500)
        out.append({"id": f"knowledge/{n}", "kind": "knowledge", "writer": WRITERS[n % 2], "prompt": KNOWLEDGE_PROMPT.format(n=QUESTIONS_PER_PASSAGE, passage=text), "schema": schema, "max_tokens": 1500, "seed": row["id"]})
    return out


def exam_jobs(seeds, used_articles, rng):
    # a window is drawn for every article of the shuffle before the first EXAM_WINDOWS are taken, as the release drew them
    articles = [row for row in seeds["wikipedia"] if len(body(row["state"]["metin"])) >= 2500 and row["id"] not in used_articles and not TRIVIA.search(row["state"]["metin"][:300])]
    schema = strict({
        "düz": strict({"soru": STEM, "doğru": OPTION, "yanlışlar": {"type": "array", "items": OPTION, "minItems": 3, "maxItems": 4}}),
        "değildir": strict({"soru": STEM, "doğrular": {"type": "array", "items": OPTION, "minItems": 3, "maxItems": 4}, "yanlış": OPTION}),
        "öncüller": strict({"soru": STEM, "önermeler": {"type": "array", "items": strict({"metin": OPTION, "doğru": {"type": "boolean"}}), "minItems": 3, "maxItems": 3}}),
    })
    out = []
    windows = [(row, window(body(row["state"]["metin"]), rng)) for row in rng.sample(articles, len(articles))]
    for n, (row, text) in enumerate([(row, text) for row, text in windows if text][:EXAM_WINDOWS]):
        out.append({"id": f"sınav/{n}", "kind": "sınav", "writer": WRITERS[n % 2], "prompt": EXAM_PROMPT.format(passage=f"{row['state']['başlık']}. {text}"), "schema": schema,
                    "max_tokens": 2000, "seed": row["id"]})
    return out


def passage_questions(job, output, rng, asked):
    for k, q in enumerate(output["sorular"]):
        options = [o.strip() for o in q["seçenekler"]]
        stem = q["soru"].strip()
        if len({o.lower() for o in options}) < 4 or CONTEXT.search(stem) or stem.lower() in asked or copies(stem, bench_fingerprints()):
            continue
        asked.add(stem.lower())
        question = {"type": "choice", "instructions": rng.choice(KNOWLEDGE_ASKS), "criteria": dict(zip("abcd", options))}
        yield item(job, k, "bilgi", stem, question, expected="abcd"[q["doğru"]])


@cache
def exam_items():
    # the word stems of every knowledge-exam item of the benchmark, all splits, with an index of the rarer stems: a re-translated
    # copy shares most stems without sharing 8-word windows
    items = []
    for task in EXAMS:
        for path in sorted((BENCH / task).glob("*.jsonl")):
            for entry in read(path):
                items.append(stems(" ".join(texts(entry["state"]) + [text for question in entry["questions"].values() for text in texts(question["criteria"])])))
    counts = Counter(stem for entry in items for stem in entry)
    index = defaultdict(list)
    for i, entry in enumerate(items):
        for stem in entry:
            if counts[stem] <= 200:
                index[stem].append(i)
    return items, index


def exam_copy(text):
    # half or more of the word stems of both texts together are shared
    items, index = exam_items()
    s = stems(text)
    return any(len(s & items[i]) * 2 >= len(s | items[i]) for i in {i for stem in s for i in index.get(stem, ())})


def exam(correct, wrong, rng):
    # a Choice question whose options are shuffled under the keys a to e, so a writer never picks the letter
    options = [correct, *wrong]
    rng.shuffle(options)
    keys = "abcde"[: len(options)]
    question = {"type": "choice", "instructions": rng.choice(KNOWLEDGE_ASKS), "criteria": dict(zip(keys, options))}
    return question, keys[options.index(correct)]


def exam_questions(job, output, rng, asked):
    plain, negative, premises = output["düz"], output["değildir"], output["öncüller"]
    statements = [p["metin"].strip() for p in premises["önermeler"]]
    right = {i for i, p in enumerate(premises["önermeler"]) if p["doğru"]}
    answer = next((name for name, members in COMBINATIONS.items() if members == right), None)
    built = [
        (plain["soru"], plain["doğru"], plain["yanlışlar"]),
        (negative["soru"], negative["yanlış"], negative["doğrular"]),
        (premises["soru"] + "\n" + "\n".join(f"{numeral}. {text}" for numeral, text in zip(NUMERALS, statements)), answer, rng.sample([c for c in COMBINATIONS if c != answer], 4)),
    ]
    for k, (stem, correct, wrong) in enumerate(built):
        stem, wrong = stem.strip(), [w.strip() for w in wrong]
        if correct is None or len({o.lower() for o in [correct.strip(), *wrong]}) < len(wrong) + 1 or CONTEXT.search(stem):
            continue
        if stem.lower() in asked or copies(stem, bench_fingerprints()) or exam_copy(" ".join([stem, correct, *wrong])):
            continue
        asked.add(stem.lower())
        question, key = exam(correct.strip(), wrong, rng)
        yield item(job, k, "sınav", stem, question, expected=key)
