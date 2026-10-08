import re

from .overlap import texts

CUT = 8000


def mask(text):
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[e-posta]", text)
    text = re.sub(r"\bTR ?\d{2}(?: ?\d{4}){5} ?\d{2}\b", "[iban]", text, flags=re.I)
    text = re.sub(r"(?<!\d)[1-9]\d{10}(?!\d)", "[kimlik]", text)
    text = re.sub(r"(?<!\d)(\+?90[\s.-]?|0)?\(?5\d{2}\)?[\s.-]?\d{3}[\s.-]?\d{2}[\s.-]?\d{2}(?!\.?\d)", "[telefon]", text)
    return re.sub(r"@\w+", "@USER", text)


def lower(text):
    return text.replace("İ", "i").replace("I", "ı").lower()


def cut(state):
    return state[:CUT] if isinstance(state, str) else {k: v[:CUT] if isinstance(v, str) else v for k, v in state.items()}


def encoded(state):
    # mostly base64 or a similar encoding: e-mail bodies that were never decoded (MIME lines are 76 characters long)
    text = " ".join(texts(state))
    return sum(map(len, re.findall(r"[A-Za-z0-9+/=]{60,}", text))) * 2 > len(text)
