import json
import random
from collections import Counter, defaultdict

import numpy as np
import torch
from s1bench.prompt import options_of

from . import DATA, read, save
from .mix import train_questions

POOL = DATA / "pool"
# native text with human or site labels that a reader can recover from the text: not TC32, HateMap or doktorsitesi, and
# no model labels, translated or synthetic text
SOURCES = [
    "bilcat", "court_outcome", "facturk", "fraud_gambling", "google_maps", "harmful_tweets", "hate_superset", "interpress",
    "kemik_news", "offensive_eymaahner", "offensive_nanelimon", "product_category", "spam_email_aydemir", "spam_email_orvile",
    "spam_sms", "thesis", "tremo", "trendyol", "vitamins", "webfaq", "wiki_stance", "yemeksepeti",
]
PER_QUESTION = 400
OTHER = "diğer"
FLOOR = np.log(1e-4)
# which questions share weights: one pair for all, one per question type, or one per type and order path
STRUCTURES = {
    "one": lambda kind, size: "all",
    "type": lambda kind, size: kind,
    "path": lambda kind, size: "choice >8" if kind == "choice" and size > 8 else kind,
}


def answer(row):
    return str(row["gold"]).lower() if isinstance(row["gold"], bool) else str(row["gold"])


def yes_no(row, rng):
    # "is this the answer?" about an option drawn at random; drawn whatever the answer is, as a written question would be,
    # so that the intercept measures the teachers and not the draw
    key, text = rng.choice(options_of(row["question"]))
    question = {"type": "noul", "instructions": f"{row['question']['instructions']}\n\nŞu cevap doğru mu: {text}", "criteria": {"true": "Evet", "false": "Hayır"}}
    out = {**row, "id": row["id"] + "#yes_no", "kind": "yes_no", "question": question, "gold": key == answer(row)}
    if "soft_gold" in row:
        out["soft_gold"] = {"true": row["soft_gold"].get(key, 0), "false": 1 - row["soft_gold"].get(key, 0)}
    return out


def other(row, rng):
    # half the options, drawn at random whatever the answer is, and "diğer" for the rest
    options = options_of(row["question"])
    drawn = rng.sample(options, (len(options) + 1) // 2)
    kept = [option for option in options if option in drawn]
    question = {**row["question"], "criteria": {**dict(kept), OTHER: "Diğer seçeneklerin hiçbiri"}}
    out = {**row, "id": row["id"] + "#other", "kind": "other", "question": question, "gold": answer(row) if answer(row) in dict(kept) else OTHER}
    if "soft_gold" in row:
        shown = {key: row["soft_gold"].get(key, 0) for key, _ in kept}
        out["soft_gold"] = {**shown, OTHER: 1 - sum(shown.values())}
    return out


def build():
    groups = defaultdict(list)
    for _, key, row in train_questions():
        if row["source"] in SOURCES or row["source"] == "lavoir":
            groups[row["source"], key].append(row)
    rng = random.Random(13)
    rows = []
    for (source, _), group in sorted(groups.items()):
        for i, row in enumerate(rng.sample(group, min(PER_QUESTION, len(group)))):
            row = {**row, "id": f"{row['id']}#{row['qid']}", "kind": "real"}
            rows.append(row)
            # each Choice item of the pool also gives one constructed question, for the "true" and "diğer" intercepts
            if source in SOURCES and row["question"]["type"] == "choice":
                rows.append(yes_no(row, rng) if i % 2 == 0 else other(row, rng))
    save(POOL / "questions.jsonl", rows)
    print(len(rows), "questions", dict(Counter(f"{row['source']} {row['kind']}" for row in rows)))


def normalised(logps):
    logps = np.asarray(logps)
    return logps - np.logaddexp.reduce(logps)


def letter_priors(rows, labels):
    # the teacher's own preference for each letter, per number of options, from the shifted orders of the Choice questions:
    # log p = option + letter + prompt, fitted by alternating means; it needs no answers. It is fitted on the floored
    # log-probabilities: below the floor a confident teacher's letters spread over tens of nats, which would drive the fit
    by_size = defaultdict(list)
    for label in labels:
        question = rows[label["id"]]["question"]
        if question["type"] == "choice":
            keys = [key for key, _ in options_of(question)]
            by_size[len(keys)].append((label["id"], keys.index(label["keys"][0]), np.maximum(normalised(label["logps"]), FLOOR)))
    priors = {}
    for size, prompts in by_size.items():
        index = {i: j for j, i in enumerate(sorted({i for i, _, _ in prompts}))}
        question = np.array([index[i] for i, _, _ in prompts])[:, None]
        shown = (np.array([shift for _, shift, _ in prompts])[:, None] + np.arange(size)) % size
        logps = np.array([values for _, _, values in prompts])
        option, letter = np.zeros((len(index), size)), np.zeros(size)
        for _ in range(30):
            prompt = (logps - option[question, shown] - letter).mean(axis=1, keepdims=True)
            sums, counts = np.zeros_like(option), np.zeros_like(option)
            np.add.at(sums, (question, shown), logps - prompt - letter)
            np.add.at(counts, (question, shown), 1)
            option = sums / counts
            letter = (logps - prompt - option[question, shown]).mean(axis=0)
            letter -= letter.mean()
        priors[size] = letter
    return priors


def scores(rows, labels):
    # each option's mean log-probability over the orders, floored at 1e-4; above 8 options and for Score the teacher's
    # letter prior comes off first, since those orders do not put every option at every letter
    priors = letter_priors(rows, labels)
    shown = defaultdict(lambda: defaultdict(list))
    for label in labels:
        logps = normalised(label["logps"])
        if len(logps) > 8 or rows[label["id"]]["question"]["type"] == "score":
            logps = normalised(logps - priors.get(len(logps), 0))
        for key, value in zip(label["keys"], np.maximum(logps, FLOOR)):
            shown[label["id"]][key].append(value)
    return {i: {key: np.mean(values) for key, values in by_key.items()} for i, by_key in shown.items()}


def intercepts(row, fitting):
    # while fitted, real questions bring one intercept per class, so that a source's label prior stays out of the weights;
    # otherwise only the constructed "true" and "diğer" intercepts apply
    if fitting and row["kind"] == "real":
        return {key: f"{row['source']} {row['question']['instructions']} {key}" for key, _ in options_of(row["question"])}
    return {"true": "true", OTHER: OTHER}


def tensors(rows, teachers, structure, groups, names, fitting):
    size = max(len(row["question"]["criteria"]) for row in rows)
    x = np.zeros((len(teachers), len(rows), size))
    target = np.zeros((len(rows), size))
    slot = np.zeros((len(rows), size), dtype=int)
    padding = np.ones((len(rows), size), dtype=bool)
    group = np.zeros(len(rows), dtype=int)
    for i, row in enumerate(rows):
        keys = [key for key, _ in options_of(row["question"])]
        soft = row.get("soft_gold") or {answer(row): 1}
        named = intercepts(row, fitting)
        group[i] = groups.index(STRUCTURES[structure](row["question"]["type"], len(keys)))
        for j, key in enumerate(keys):
            x[:, i, j] = [teacher[row["id"]][key] for teacher in teachers]
            target[i, j] = soft.get(key, 0)
            slot[i, j] = names.get(named.get(key), 0)
        padding[i, :len(keys)] = False
    return [torch.tensor(a) for a in (x, target / target.sum(axis=1, keepdims=True), slot, padding, group)]


def logits(data, weights, values):
    x, _, slot, padding, group = data
    z = (weights[group].T[:, :, None] * x).sum(0) + torch.cat([torch.zeros(1, dtype=values.dtype), values])[slot]
    return z.masked_fill(padding, -torch.inf)


def losses(data, weights, values):
    _, target, _, padding, _ = data
    return -(target * logits(data, weights, values).log_softmax(-1).masked_fill(padding, 0)).sum(-1)


class Model:
    # p = softmax(sum of each teacher's weight times its log-probabilities, plus intercepts), by maximum likelihood
    def __init__(self, rows, teachers, structure):
        self.teachers, self.structure = teachers, structure
        self.groups = sorted({STRUCTURES[structure](row["question"]["type"], len(row["question"]["criteria"])) for row in rows})
        self.names = {name: k + 1 for k, name in enumerate(sorted({n for row in rows for n in intercepts(row, True).values()}))}
        data = tensors(rows, teachers, structure, self.groups, self.names, True)
        self.weights = torch.full((len(self.groups), len(teachers)), 1 / len(teachers), dtype=torch.float64, requires_grad=True)
        self.values = torch.zeros(len(self.names), dtype=torch.float64, requires_grad=True)
        optimiser = torch.optim.LBFGS([self.weights, self.values], max_iter=1000, line_search_fn="strong_wolfe")

        def closure():
            optimiser.zero_grad()
            # a Gaussian prior on the intercepts keeps those of rare classes near zero
            loss = (losses(data, self.weights, self.values).sum() + (self.values**2).sum()) / len(rows)
            loss.backward()
            return loss

        optimiser.step(closure)
        self.weights, self.values = self.weights.detach(), self.values.detach()

    def loss(self, rows):
        return losses(tensors(rows, self.teachers, self.structure, self.groups, self.names, False), self.weights, self.values).numpy()


def loso(rows, teachers, structure):
    # each source in turn is left out of the fit and scored as a new family would be: without intercepts of its own
    means = []
    for source in SOURCES:
        held = [row for row in rows if row["source"] == source and row["kind"] == "real"]
        means.append(Model([row for row in rows if row["source"] != source], teachers, structure).loss(held).mean())
    return np.mean(means)


def fit(first, second):
    rows = {row["id"]: row for row in read(POOL / "questions.jsonl")}
    teachers = [scores(rows, read(POOL / f"{name}.jsonl")) for name in [first, second]]
    # Lavoir's questions count in the letter priors, but the weights are never fitted on them
    pool = [row for row in rows.values() if row["source"] != "lavoir"]
    results = {structure: loso(pool, teachers, structure) for structure in STRUCTURES}
    for structure, result in results.items():
        print(f"leave one source out, weights {structure}: log-loss {result:.4f}")
    chosen = min(results, key=results.get)
    model = Model(pool, teachers, chosen)
    weights = {group: [round(float(w), 4) for w in model.weights[k]] for k, group in enumerate(model.groups)}
    constructed = {name: round(float(model.values[model.names[name] - 1]), 4) for name in ["true", OTHER]}
    fitted = {"teachers": [first, second], "structure": chosen, "weights": weights, "intercepts": constructed}
    json.dump(fitted, open(POOL / f"{first}+{second}.json", "w"), ensure_ascii=False, indent=1)
    print("weights", weights, "intercepts", constructed)
    return fitted


def pooled(row, teachers, fitted, with_intercepts):
    keys = [key for key, _ in options_of(row["question"])]
    weights = fitted["weights"][STRUCTURES[fitted["structure"]](row["question"]["type"], len(keys))]
    shift = fitted["intercepts"] if with_intercepts else {}
    z = np.array([sum(w * t[row["id"]][key] for w, t in zip(weights, teachers)) + shift.get(key, 0) for key in keys])
    p = np.exp(z - z.max())
    return dict(zip(keys, p / p.sum()))


if __name__ == "__main__":
    build()
