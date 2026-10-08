import json
import math
import re
from collections import defaultdict

from ..overlap import clone, words
from . import WRITERS, item, strict

SCHEMA_CALLS = 32
SLOTS = 6
RECORDS_PER_QUESTION = 60
KEPT_PER_QUESTION = 40
# a schema question is kept when its answers depend on the record by at least this much, in bits, and when answers
# other than the most common one cover at least this many records
MIN_BITS = 0.2
MIN_MINORITY = 10
FAMILIES = {
    "talep": "kaydı yazanın ne istediği ya da ne beklediği",
    "yönlendirme": "kaydı hangi ekibin ya da birimin ele alması gerektiği",
    "sonraki adım": "kaydı işleyenin şimdi ne yapması gerektiği: cevap vermek, eksik bilgiyi sormak, bir araç kullanmak, bir insana devretmek ya da kapatmak",
    "değer": "kayıtta açıkça yazan bir değer: tarih, tutar, yer, ürün, kurum, sayı ya da bir kişinin rolü",
    "olgu": "kayıtta açıkça yazan ya da yazmayan bir durum",
    "kural": "kaydın, soruda yazan bir kurum kuralına uyup uymadığı",
    "tutum": "kaydı yazanın ya da konuşmacının açıkça savunduğu tutum",
    "konu": "kaydın ana konusu",
}
# no health or legal records: nobody can check questions about them and they hold special categories of personal data
RECORDS = {
    "ürün yorumu": ("bir e-ticaret sitesinin müşteri deneyimi ekibi", ["talep", "değer", "olgu", "konu", "yönlendirme"], ["yorumbudur", "trendyol", "vitamins"]),
    "mekân yorumu": ("bir restoran ve kafe zincirinin müşteri ilişkileri ekibi", ["talep", "değer", "olgu", "konu"], ["google_maps"]),
    "yemek siparişi yorumu": ("bir yemek sipariş uygulamasının destek ekibi", ["talep", "olgu", "yönlendirme", "değer"], ["yemeksepeti"]),
    "şikâyet": ("şikâyet edilen firmanın müşteri hizmetleri", ["talep", "yönlendirme", "sonraki adım", "değer", "olgu"], ["tc32"]),
    "haber": ("bir haber ajansının editörleri", ["değer", "olgu", "konu"], ["tr_news", "financial_news", "kemik_news", "bilcat"]),
    "şirket duyurusu": ("bir aracı kurumun yatırım analistleri", ["değer", "olgu", "konu"], ["kap"]),
    "sosyal medya paylaşımı": ("bir sosyal medya platformunun içerik denetim ekibi", ["kural", "olgu", "tutum", "konu"], ["thy_tweets", "hate_superset", "offensive_eymaahner", "harmful_tweets", "tremo"]),
    "e-posta": ("bir şirketin e-posta güvenliği ekibi", ["kural", "talep", "olgu", "yönlendirme"], ["spam_email_aydemir", "spam_email_orvile"]),
    "kısa mesaj": ("bir operatörün mesaj güvenliği ekibi", ["kural", "olgu", "talep"], ["spam_sms"]),
    "meclis konuşması": ("bir gazetenin meclis muhabirleri", ["tutum", "değer", "olgu", "konu"], ["parliament"]),
    "vikipedi maddesi": ("bir ansiklopedinin editörleri", ["değer", "olgu", "konu"], ["wikipedia"]),
    "yapay zekâ asistanına gelen mesaj": ("yapay zekâ asistanının güvenlik ekibi", ["kural", "talep", "sonraki adım", "olgu"], ["guardrail", "injection"]),
    "araç kullanan bir asistanla yazışma": ("asistanı geliştiren ekip", ["sonraki adım", "talep", "değer", "olgu"], ["tools_atasoglu", "tools_hermes", "tools_when2call"]),
    "doğruluk kontrolü raporu": ("bir doğruluk kontrolü platformunun editörleri", ["değer", "olgu", "konu"], ["facturk"]),
    "soru ve cevap": ("bir sıkça sorulan sorular sayfasının editörleri", ["olgu", "değer", "konu"], ["webfaq"]),
    "tez özeti": ("bir üniversite kütüphanesinin katalog ekibi", ["konu", "değer", "olgu"], ["thesis"]),
}
SCHEMA_PROMPT = """Türkçe bir "System One" modeli için karar soruları tasarlıyorsun. Bu model bir kaydı (mesaj, belge, form, yazışma) okur ve bir sorunun her seçeneğine bir olasılık verir. Yazdığın her soru bu türdeki bütün kayıtlara sorulacak; bu yüzden soruyu tek bir kayda göre değil, kayıt türüne göre yaz. Birçok kayıtta doğru cevap "hayır", "diğer" ya da "belirtilmemiş" olacak ve bu doğaldır. Doğal, günlük Türkçe yaz; çeviri kokan kalıplar kullanma. Yalnızca istenen JSON'u yaz.

Kayıt türü: {record_type}
Soruları soran: {asker}

Bu türden beş örnek kayıt aşağıda. Bunlar yalnızca fikir vermek için: sorularda bu kayıtlardaki adları, tutarları, tarihleri ya da olayları kullanma.
{examples}

Aileler:
{families}

Aşağıdaki her yuva için bir soru yaz. Yuvadaki aileyi, türü ve seçenek sayısını değiştirme.
{slots}

Kurallar
1. Her soru kaydın tek bir özelliğini sorsun ve kısa olsun; "Kayıttan anlaşılan üzere" gibi gereksiz girişler kullanma. Soru kalıplarını çeşitlendir: soru cümlesi, "... belirle" gibi bir yönerge ya da "Bu kayıt için hangisi doğru?" gibi bir kalıp.
2. Yuvada "ters": true varsa soruyu, "evet" cevabı ailenin aradığı durumun olmadığını gösterecek biçimde kur. Olumsuz soru kullanma ("... yok mu?", "... içermiyor mu?"): bu sorulara verilen "evet" iki anlama gelebilir. Karşıt durumu olumlu bir soruyla sor (ör. "Mesaj kaba sözlerden uzak mı?", "Sipariş eksiksiz teslim edilmiş mi?").
3. Cevap yalnızca kayıttan ve seçeneklerin ölçütlerinden çıkabilsin. Dış bilgi, kaydın kaynağı ya da yazanın kim olduğu gerekmesin.
4. Her seçeneğe bir ölçüt yaz: o seçeneğin ne zaman doğru olduğunu anlatan, tek başına anlaşılır bir cümle. Ölçüt seçeneğin adıyla başlasın ("İade: ..."). Karışabilecek iki seçenek arasındaki sınırı açıkça yaz ("... ancak ... yoksa"). Seçenekler örtüşmesin ve birlikte her kaydı kapsasın.
5. Seçenekler karışık sırayla gösterilir. Bir ölçüt başka bir seçeneğe sırasıyla ya da harfiyle gönderme yapmasın ("yukarıdakiler", "A şıkkı", "önceki seçenek" olmaz).
6. choice: Liste her kaydı kapsamıyorsa "diğer" seçeneğini ekle ("Diğer: Kayıt bu seçeneklerin hiçbirine uymuyor."). Kayıt bu konuda hiç bilgi vermeyebiliyorsa "belirtilmemiş" seçeneğini ekle ("Belirtilmemiş: Kayıtta bu konuda bilgi yok."). Bu seçenekler yuvadaki seçenek sayısına dahildir.
   noul: Ölçütler "Evet:" ve "Hayır:" ile başlasın ve iki durumu açıkça ayırsın (ör. "Evet: Müşteri parasının geri ödenmesini istiyor." / "Hayır: Müşteri para iadesi istemiyor ya da başka bir çözüm istiyor.").
7. Yuvada "kurum_kuralı": true varsa, cevabı sağduyudan farklı yapabilecek bir kurum kuralını ilgili ölçütün içine yaz (ör. "Acil: Yalnızca para kaybı ya da hukuki işlem söz konusuysa. Bu ekip için gecikme tek başına acil değildir.").
8. Kayıtta açıkça yazmayan bir hastalığı, tanıyı ya da kişisel bir özelliği (etnik köken, din, siyasi görüş, cinsel yaşam, suç geçmişi) tahmin ettiren soru sorma. Kaydı yazanın ne istediğini, neye ihtiyaç duyduğunu ya da konuşmacının kayıtta açıkça savunduğu tutumu sormak serbesttir (ör. "Konuşmacı bu kanun teklifini destekliyor mu?").

Çıktı:
{{"sorular":[{{"aile":"…","type":"choice|noul","instructions":"…","options":[{{"key":"…","criterion":"…"}}]}}]}}
noul anahtarları "true" ve "false"; choice anahtarları kısa adlardır ("diğer" ve "belirtilmemiş" dahil). Bir değeri adlandıran seçenekte anahtar değerin kendisi olabilir ("250 TL", "1923").

Örnek:
{example}"""
EXAMPLES = [
    {"aile": "talep", "type": "choice", "instructions": "Müşteri bu mesajla ne istiyor?", "options": [
        {"key": "iade", "criterion": "İade: Müşteri ürünü geri gönderip parasını almak istiyor."},
        {"key": "değişim", "criterion": "Değişim: Müşteri ürünün başka bir beden, renk ya da sağlam bir örnekle değiştirilmesini istiyor."},
        {"key": "bilgi", "criterion": "Bilgi: Müşteri yalnızca bilgi istiyor, bir işlem talep etmiyor."},
        {"key": "diğer", "criterion": "Diğer: Müşteri bu seçeneklerin hiçbirine uymayan bir şey istiyor ya da bir şey istemiyor."}]},
    {"aile": "olgu", "type": "noul", "instructions": "Yazışmada bir sipariş ya da başvuru numarası veriliyor mu?", "options": [
        {"key": "true", "criterion": "Evet: Yazışmada işlemi tanımlayan bir numara ya da kod yazıyor."},
        {"key": "false", "criterion": "Hayır: Yazışmada böyle bir numara ya da kod yok."}]},
    {"aile": "yönlendirme", "type": "choice", "instructions": "Bu kaydı hangi birim ele almalı?", "options": [
        {"key": "faturalama", "criterion": "Faturalama: Kayıt bir ödeme, fatura ya da ücret sorunuyla ilgili."},
        {"key": "teknik destek", "criterion": "Teknik destek: Kayıt bir cihazın, uygulamanın ya da bağlantının çalışmamasıyla ilgili."},
        {"key": "lojistik", "criterion": "Lojistik: Kayıt bir gönderinin gecikmesi, kaybolması ya da hasar görmesiyle ilgili."},
        {"key": "diğer", "criterion": "Diğer: Kayıt bu birimlerin hiçbirinin işine girmiyor."}]},
]
POSITIONAL = re.compile(r"yukarıdaki|aşağıdaki|önceki seçenek|sonraki seçenek|\b[A-H] (şıkkı|seçeneği)", re.I)
TEXT = {"type": "string", "minLength": 8, "maxLength": 300}


def show(state):
    return "\n".join(f"{k}: {v}" for k, v in state.items()) if isinstance(state, dict) else state


def slot_schema(slot):
    if slot["type"] == "noul":
        key, count = {"enum": ["true", "false"]}, 2
    else:
        key, count = {"type": "string", "pattern": '^[^\\s"][^"]{0,39}$'}, slot["seçenek"]
    option = strict({"key": key, "criterion": TEXT})
    return strict({"aile": {"const": slot["aile"]}, "type": {"const": slot["type"]}, "instructions": TEXT, "options": {"type": "array", "items": option, "minItems": count, "maxItems": count}})


def slots(families, rng):
    out = []
    for family in rng.choices(families, k=SLOTS):
        slot = {"aile": family, "type": "noul" if family == "olgu" or rng.random() < 0.3 else "choice"}
        if slot["type"] == "noul":
            slot["ters"] = rng.random() < 0.3
        else:
            slot["seçenek"] = rng.randint(6, 12) if family == "yönlendirme" else rng.randint(3, 7)
        if family == "kural" or (family in ["yönlendirme", "sonraki adım"] and rng.random() < 0.5):
            slot["kurum_kuralı"] = True
        out.append(slot)
    return out


def jobs(seeds, rng):
    out = []
    for record_type, (asker, families, sources) in sorted(RECORDS.items()):
        pool = [row for source in sources for row in seeds[source]]
        for call in range(SCHEMA_CALLS):
            examples = rng.sample(pool, 5)
            wanted = slots(families, rng)
            prompt = SCHEMA_PROMPT.format(
                record_type=record_type,
                asker=asker,
                examples="\n".join(f"<örnek>{show(row['state'])[:1000]}</örnek>" for row in examples),
                families="\n".join(f"- {family}: {FAMILIES[family]}" for family in sorted({slot["aile"] for slot in wanted})),
                slots="\n".join(json.dumps(slot, ensure_ascii=False) for slot in wanted),
                example=json.dumps(rng.choice(EXAMPLES), ensure_ascii=False),
            )
            questions = {"type": "array", "prefixItems": [slot_schema(slot) for slot in wanted], "items": False, "minItems": SLOTS, "maxItems": SLOTS}
            out.append({"id": f"schema/{record_type}/{call}", "kind": "schema", "writer": WRITERS[call % 2], "prompt": prompt, "schema": strict({"sorular": questions}), "max_tokens": 4096,
                        "record_type": record_type, "sources": sources, "examples": [row["id"] for row in examples]})
    return out


def questions(job, output, rng, asked, seeds):
    pool = [row for source in job["sources"] for row in seeds[source] if row["id"] not in job["examples"]]
    for k, q in enumerate(output["sorular"]):
        keys = [o["key"].strip() for o in q["options"]]
        question = {"type": q["type"], "instructions": q["instructions"].strip(), "criteria": dict(zip(keys, [o["criterion"].strip() for o in q["options"]]))}
        norm = " ".join(words(question["instructions"]))
        if len(set(keys)) < len(keys) or any(POSITIONAL.search(t) for t in [question["instructions"], *question["criteria"].values()]) or norm in asked or clone(question):
            continue
        asked.add(norm)
        for row in rng.sample(pool, RECORDS_PER_QUESTION):
            yield item(job, f"{k}/{row['id']}", q["aile"], row["state"], question, record_type=job["record_type"], schema=f"{job['id']}/{k}")


def information(rows):
    # how much a question's answers depend on the record, in bits; 0 when every record gets the same distribution
    mean = {key: sum(p[key] for p in rows) / len(rows) for key in rows[0]}

    def entropy(p):
        return -sum(v * math.log2(v) for v in p.values() if v > 0)

    return entropy(mean) - sum(entropy(p) for p in rows) / len(rows)


def kept(rows, pooled):
    by_schema = defaultdict(list)
    for i, row in rows.items():
        if row["meta"]["kind"] == "schema":
            by_schema[row["meta"]["schema"]].append(i)
    out = []
    for ids in by_schema.values():
        answers = defaultdict(list)
        for i in ids:
            answers[max(pooled[i], key=pooled[i].get)].append(i)
        majority = max(answers, key=lambda a: len(answers[a]))
        others = [i for a, group in answers.items() if a != majority for i in group]
        if information([pooled[i] for i in ids]) >= MIN_BITS and len(others) >= MIN_MINORITY:
            others = others[: KEPT_PER_QUESTION * 2 // 3]
            out += others + answers[majority][: min(2 * len(others), KEPT_PER_QUESTION - len(others))]
    return out
