import random


def yes_no(instructions):
    return {"type": "noul", "instructions": instructions, "criteria": {"true": "Evet", "false": "Hayır"}}


def stars(instructions):
    criteria = ["1 yıldız: Çok olumsuz", "2 yıldız: Olumsuz", "3 yıldız: Ne iyi ne kötü", "4 yıldız: Olumlu", "5 yıldız: Çok olumlu"]
    return {"type": "score", "instructions": instructions, "criteria": criteria}


def choice(instructions, options):
    return {"type": "choice", "instructions": instructions, "criteria": dict.fromkeys(options)}


def with_distractors(instructions, gold, options, text):
    # a question with too many options shows the right one among nine others, picked by the text so that a
    # repeated text gets the same options
    others = random.Random(text).sample([option for option in options if option != gold], 9)
    return choice(instructions, sorted(others + [gold]))
