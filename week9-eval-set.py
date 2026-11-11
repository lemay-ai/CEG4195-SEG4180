# container: pytorch
# needs: network
# Build your own eval set: 45 support tickets, each with the expected label (billing,
# account or technical), in data/week9-eval-set.csv. Run two small local LLMs and two
# prompt versions on every ticket, then compare accuracy, tokens per task and time per
# task, and print the failures to read. No API key: the models download from Hugging Face
# (about 3.8 GB in all) and run on a CPU. For an API model, the cost per 1,000 tasks is
# the tokens per task times the price per token, times 1,000.
# Tested November 2026: transformers 5.22.2, torch 2.9.
import csv
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SMOKE = os.environ.get("COURSE_SMOKE") == "1"  # the course smoke test: one model, 9 tickets
torch.manual_seed(0)

# region: eval-set
rows = list(csv.DictReader(open(Path(__file__).parent / "data" / "week9-eval-set.csv")))
LABELS = ["billing", "account", "technical"]
PROMPTS = {
    "v1": ("Classify this customer support ticket as billing, account or technical. "
           "Answer with one word.\n\nTicket: {text}"),
    "v2": ("Classify this customer support ticket. Answer with one word:\n"
           "billing: charges, refunds, prices, invoices, payment methods\n"
           "account: login, password, profile, users, security, personal data\n"
           "technical: the app or website does not work as it should\n\nTicket: {text}"),
}
# endregion
if SMOKE:
    rows = rows[::5]

models = ["HuggingFaceTB/SmolLM2-360M-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]
if SMOKE:
    models = models[:1]


# region: run
def run(tok, model, prompt):
    results = []
    for row in rows:
        messages = [{"role": "user", "content": prompt.format(text=row["text"])}]
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True)
        start = time.perf_counter()
        out = model.generate(**ids, max_new_tokens=4, do_sample=False)
        seconds = time.perf_counter() - start
        answer = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        words = answer.lower().replace(".", " ").split()
        predicted = next((w for w in words if w in LABELS), "invalid")
        results.append({**row, "answer": answer.strip(), "predicted": predicted,
                        "tokens": out.shape[1], "seconds": seconds})
    return results
# endregion


print(f"{len(rows)} tickets")
print(f"{'Model':38s} {'Prompt':6s} {'Accuracy':>8s} {'Invalid':>7s} {'Tokens/task':>11s} {'s/task (CPU)':>12s}")
failures = {}
for name in models:
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    for version, prompt in PROMPTS.items():
        r = run(tok, model, prompt)
        accuracy = sum(x["predicted"] == x["label"] for x in r) / len(r)
        invalid = sum(x["predicted"] == "invalid" for x in r)
        tokens = sum(x["tokens"] for x in r) / len(r)
        seconds = sum(x["seconds"] for x in r) / len(r)
        print(f"{name:38s} {version:6s} {accuracy:8.0%} {invalid:7d} {tokens:11.0f} {seconds:12.2f}")
        failures[f"{name}, prompt {version}"] = [x for x in r if x["predicted"] != x["label"]]

# Read the failures: they show what the prompt or the labels need
for name, wrong in failures.items():
    print(f"\n{name}: {len(wrong)} wrong")
    for x in wrong[:4]:
        print(f"  expected {x['label']:9s} got {x['answer']!r:14s} {x['text']}")
