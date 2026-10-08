import json
import re

from ..download import hf_rows


def call_now(name):
    # the labels say whether the tool is called in reply to this message, not whether it is ever needed
    return {
        "type": "noul",
        "instructions": f"Asistan bu mesaja yanıt olarak şimdi {name} aracını çağırmalı mı?",
        "criteria": {
            "true": "İstek bu araçla şimdi karşılanabilir ve aracın ihtiyaç duyduğu bilgiler mesajda var.",
            "false": "Bu araç gerekmiyor, ya da önce kullanıcıya eksik bilgi sorulmalı, ya da araç daha sonraki bir adımda çağrılmalı.",
        },
    }


def tool_item(tools, request, called):
    return {
        "state": {"araçlar": tools, "istek": request},
        "questions": {tool["name"]: call_now(tool["name"]) for tool in tools},
        "gold": {tool["name"]: tool["name"] in called for tool in tools},
    }


def when2call_items(split):
    # only the When2Call rows, translated with DeepSeek; the xLAM rows were translated with the OpenAI API
    items = []
    for r in hf_rows("bilalabic/turkish-tool-calling", "default", split):
        tools = json.loads(r["tools"])
        called = {call["function"]["name"] for m in r["messages"] if m["tool_calls"] for call in m["tool_calls"]}
        if tools and r["id"].startswith("when2call"):
            items.append(tool_item(tools, r["messages"][1]["content"], called))
    return items


def build_tools_when2call():
    return {"heldout": when2call_items("test"), "train": when2call_items("train")}


def build_tools_atasoglu():
    items = []
    for r in hf_rows("atasoglu/turkish-function-calling-20k", None, "train"):
        tools = [tool["function"] for tool in json.loads(r["tools"])]
        called = {answer["function"]["name"] for answer in json.loads(r["answers"] or "[]")}
        items.append(tool_item(tools, r["query"], called))
    return {"train": items}


def build_tools_hermes():
    items = []
    for r in hf_rows("Tuguberk/turkish-hermes-function-calling", None, "train"):
        turns = r["conversations"]
        if "func_calling" in r["_subset"] and len(turns) > 2 and turns[2]["from"] == "gpt":
            tools = [tool.get("function", tool) for tool in json.loads(r["tools"] or "[]") or []]
            # some calls are not valid JSON, so look for the tool names inside the call blocks instead of parsing them
            calls = " ".join(re.findall(r"<tool_call>(.*?)</tool_call>", turns[2]["value"], re.S))
            called = {tool["name"] for tool in tools if f'"{tool["name"]}"' in calls}
            if tools:
                items.append(tool_item(tools, turns[1]["value"], called))
    return {"train": items}
