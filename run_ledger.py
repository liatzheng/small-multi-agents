"""Run the single agent, ensemble, and hierarchies on ledger aggregation problems through OpenRouter."""

import argparse
import functools
import json
import math
import os
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from generate_ledger import CUSTOMERS
from run_wide import SAMPLING, OpenRouterChat, resume_log, run_problem, tokenizer_file

# Gemma 4B slips when it adds a dozen numbers in one expression, so every role adds stepwise.
# Stated as a general rule, it made workers keep one grand total even when asked to filter or
# group, so it applies only to adding a list.
ADDING = "When you add a list of numbers, add them one at a time, writing each running total."
ANSWER_PROMPT = ("Work through the problem step by step. " + ADDING + " End your response with "
                 "the final answer on its own line in the form {form}")
WORK_PROMPT = (
    "You are a worker on a team. You see one store's ledger and an instruction. Follow the "
    "instruction step by step. " + ADDING + " Then write a line starting with REPORT: "
    "followed by exactly what the instruction asks you to report. Your team sees only what "
    "comes after REPORT:."
)
PLAN_PROMPT = (
    "You are the orchestrator of a team. The data is the sales ledgers of several stores; each "
    "row is customer, category, amount. Each worker will receive one store's ledger and your "
    "instruction, nothing else. A reconstructor will then see the question, your instruction, "
    "and every worker's report, but not the ledgers, and must answer the question from the "
    "reports alone. Write the instruction for the workers: what to compute from their store's "
    "ledger and exactly what to report so that the question can be answered from the reports. "
    "Do not answer the question. Return only the instruction."
)
RECONSTRUCT_PROMPT = (
    "You are the reconstructor on a team. Each worker followed the instruction below on one "
    "store's ledger and wrote a report. You cannot see the ledgers. Combine the reports to "
    "answer the question, step by step. " + ADDING + " End your response with the final "
    "answer on its own line in the form {form}"
)
MERGE_PROMPT = (
    "You are a merger on a team. Workers each followed the instruction below on one store's "
    "ledger and wrote a report. Each input below is either such a report or a merged report "
    "covering several stores. You cannot see the ledgers. Combine the inputs into one report in "
    "the same format that covers all of their stores, step by step. " + ADDING + " Then write a "
    "line starting with REPORT: followed by the combined report. Your team sees only what comes "
    "after REPORT:."
)
# For linked rows, every worker also reports its store's total in a fixed form, so code can pass
# it to the stores that refer to it. Only this line is not left to the planner.
STORE_TOTAL_LINE = "Also report one line: store total = <the sum of all amounts in your ledger>"
SCHEDULE_PROMPT = (
    "You are the orchestrator of a team. The data is the sales ledgers of several stores; each "
    "row is customer, category, amount. Each worker will receive one store's ledger and your "
    "instruction, nothing else, and will also report its store's total, the sum of all its "
    "amounts. A reconstructor will then see the question, your instruction, and every worker's "
    "report, but not the ledgers, and must answer the question from the reports alone. If a "
    "store's ledger cannot be completed without another store's total, write a line "
    "'Store <a> needs Store <b>'; the worker for store a will then run after store b's worker "
    "and be given its total. Write the instruction for the workers: what to compute from their "
    "store's ledger and exactly what to report so that the question can be answered from the "
    "reports. Then write the needs lines, if any. Do not answer the question. Return only the "
    "instruction and the needs lines."
)
DISCOVER_RULE = (
    " If a row's amount refers to another store's total and you were not given that total, do "
    "not guess: write only a line NEED: Store <number>, naming that store, and no REPORT. You "
    "will be run again with its total."
)
CEO_PLAN_PROMPT = (
    "You lead a company that answers a question about the sales ledgers of several stores; each "
    "row is customer, category, amount. The stores are split into groups. Each group has a "
    "manager and one worker per store. Each manager will receive the question and your "
    "instruction, plan the work for its workers, and send you one group report. You will then "
    "see only the group reports, not the ledgers, and must answer the question from them. Write "
    "the instruction for the managers: exactly what each group report must contain so that the "
    "question can be answered from the group reports. Do not answer the question. Return only "
    "the instruction."
)
MANAGER_PLAN_PROMPT = (
    "You are a manager in a company. Each store in your group goes to its own worker, who "
    "receives the store's ledger and your instruction, nothing else; each row is customer, "
    "category, amount. After the workers report, you will combine their reports, without "
    "seeing the ledgers, into the group report your leader asks for. Write the instruction for "
    "your workers: what to compute from their store's ledger and exactly what to report so that "
    "you can build the group report from their reports. Do not answer the question. Return "
    "only the instruction."
)
MANAGER_MERGE_PROMPT = (
    "You are a manager in a company. Each of your workers followed your instruction below on "
    "one store's ledger and wrote a report. You cannot see the ledgers. Combine the reports "
    "into the group report your leader's instruction asks for, step by step. " + ADDING + " Then "
    "write a line starting with REPORT: followed by the group report. Your leader sees only "
    "what comes after REPORT:."
)
CEO_MERGE_PROMPT = (
    "You lead a company. Each manager followed your instruction below for a group of stores and "
    "sent a group report. You cannot see the ledgers. Combine the group reports to answer the "
    "question, step by step. " + ADDING + " End your response with the final answer on its own "
    "line in the form {form}"
)
# The CEO's merge sees one report per group rather than per store, so it holds back less.
CEO_MERGE_SHARE = 8
# A manager's planning call carries its prompt, so it takes a quarter of the group's budget, and
# the same is held back for its merge.
MANAGER_SHARE = 4
SYSTEMS = ("single_agent", "ensemble", "hierarchy_oracle", "hierarchy_reconstruct", "hierarchy",
           "hierarchy_tree_reconstruct", "hierarchy_tree", "hierarchy_schedule",
           "hierarchy_discover", "hierarchy_ceo")
# The reconstructor's share is held back from the workers so they cannot starve it.
RECONSTRUCT_SHARE = 4
PLAN_SHARE = 8
# A tree merge gives each merger at most this many reports, so no call adds a wide list.
FAN_IN = 4

REPORT_LINE = re.compile(r"^[\W_]*report[\W_]*:[*_\s]*", re.I | re.M)
ANSWER_LINE = re.compile(r"answer[*_\s]*:[*_\s]*(.+)$", re.I | re.M)
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
PAIR = re.compile(r"([A-Za-z]+)\**\s*[=:]\s*\**\s*(-?\d+)")
TOTAL = re.compile(r"total\s*=\s*\**\s*(-?\d+)", re.I)
STORE_TOTAL = re.compile(r"store[*_\s]*total\s*=\s*\**\s*(-?\d+)", re.I)
NEEDS_LINE = re.compile(r"^[\W_]*Store\s+(\d+)\s+needs\s+Store\s+(\d+)\W*$", re.I | re.M)
NEED = re.compile(r"^[\W_]*NEED[\W_]*:[*_\s]*Store\s+(\d+)", re.I | re.M)


def customer_values(text):
    """Read 'Name = value' pairs for known customers; a later pair for a name wins."""
    return {name: int(value) for name, value in PAIR.findall(text) if name in CUSTOMERS}


def parse_answer(text, problem):
    """Read the last 'Answer:' as a list aligned with the targets, None where a value is missing."""
    matches = list(ANSWER_LINE.finditer(text))
    if not matches:
        return None, False
    lines = [l for l in text.splitlines() if l.strip()]
    format_ok = bool(ANSWER_LINE.search(lines[-1]))
    if problem["question_type"] == "customer_totals":
        # Models sometimes put one customer per line after "Answer:", so read to the end.
        values = customer_values(text[matches[-1].start():])
        prediction = [values.get(t) for t in problem["targets"]]
        return (None if all(v is None for v in prediction) else prediction), format_ok
    numbers = [float(n.replace(",", "")) for n in NUMBER.findall(matches[-1].group(1))]
    # An average given with decimals is rounded down as the question asks.
    return ([math.floor(numbers[-1])] if numbers else None), format_ok


def extract_report(text):
    """Return everything after the first REPORT: marker, dropping markers that repeat per line."""
    match = REPORT_LINE.search(text)
    if not match:
        return None
    report = REPORT_LINE.sub("", text[match.end():]).strip().strip("*`").strip()
    return report or None


def merge_reports(problem, reports):
    """Merge the oracle instruction's reports in code, or return None if any is unreadable."""
    kind = problem["question_type"]
    if kind == "total":
        totals = [re.search(r"total\s*=\s*\**\s*(-?\d+)", r or "", re.I) for r in reports]
        return None if None in totals else [sum(int(m.group(1)) for m in totals)]
    if kind == "average":
        sums, counts = 0, 0
        for r in reports:
            total = re.search(r"total\s*=\s*\**\s*(-?\d+)", r or "", re.I)
            count = re.search(r"count\s*=\s*\**\s*(-?\d+)", r or "", re.I)
            if not (total and count):
                return None
            sums, counts = sums + int(total.group(1)), counts + int(count.group(1))
        return [sums // counts] if counts else None
    spent = {}
    for r in reports:
        if r is None:
            return None
        for name, value in customer_values(r).items():
            spent[name] = spent.get(name, 0) + value
    return [spent.get(t) for t in problem["targets"]] if spent else None


def finish(problem, budget, calls, prediction, **fields):
    """Score a prediction with None for missing values and total every call's tokens."""
    answer = problem["answer"]
    hits = [p == a for p, a in zip(prediction or [None] * len(answer), answer)]
    if "status" not in fields:
        fields["status"] = "correct" if all(hits) else "wrong_answer"
    if len(answer) == 1:
        # One slip anywhere makes a single number wrong, so the error size shows how close it was.
        value = prediction[0] if prediction else None
        fields["abs_error"] = None if value is None else abs(value - answer[0])
    input_tokens = sum(c["input_tokens"] for c in calls)
    output_tokens = sum(c["output_tokens"] for c in calls)
    return {
        **problem, **fields, "prediction": prediction, "correct": all(hits),
        "target_accuracy": sum(hits) / len(hits), "calls": calls, "budget": budget,
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def answer_messages(problem):
    return [{"role": "system", "content": ANSWER_PROMPT.format(form=problem["answer_format"])},
            {"role": "user", "content": problem["prompt"]}]


def reply_status(reply, prediction):
    if reply["stop_reason"] != "stop":
        return reply["stop_reason"]
    return "format_error" if prediction is None else "answered"


def single_steps(problem, budget, seed):
    (reply,) = yield [{"messages": answer_messages(problem), "cap": budget, "seed": seed}]
    prediction, format_ok = parse_answer(reply["final_text"], problem)
    status = reply_status(reply, prediction)
    fields = {"format_ok": format_ok}
    if status != "answered":
        fields["status"] = status
    return finish(problem, budget, [reply], prediction, **fields)


def ensemble_steps(problem, budget, seed, attempts=3):
    base, remainder = divmod(budget, attempts)
    caps = [base] * (attempts - remainder) + [base + 1] * remainder
    replies = yield [{"messages": answer_messages(problem), "cap": cap, "seed": seed + i}
                     for i, cap in enumerate(caps)]
    width = len(problem["targets"])
    votes = []
    for reply in replies:
        prediction, _ = parse_answer(reply["final_text"], problem)
        reply["status"] = reply_status(reply, prediction)
        reply["vote"] = prediction if reply["status"] == "answered" else None
        votes.append(reply["vote"] or [None] * width)
    # Each target takes the value at least two attempts agree on, as in run_wide.py.
    prediction = []
    for column in zip(*votes):
        counts = Counter(v for v in column if v is not None).most_common(1)
        prediction.append(counts[0][0] if counts and counts[0][1] * 2 > attempts else None)
    any_correct = [a in column for column, a in zip(zip(*votes), problem["answer"])]
    fields = {"any_correct_targets": sum(any_correct) / width}
    if None in prediction:
        fields["status"] = "no_majority"
    return finish(problem, budget, list(replies), prediction, **fields)


def store_total(problem, report):
    """Read a store's total from its worker's oracle report, or None if it is unreadable."""
    if report is None:
        return None
    if problem["question_type"] == "total":
        match = TOTAL.search(report)
        return int(match.group(1)) if match else None
    values = customer_values(report)
    return sum(values.values()) if values else None


def worker_steps(problem, instruction, cap, seed, calls):
    """Give every store's ledger to its own worker, one parallel round per link level.

    A store with a linked row waits for the store it refers to, and its worker is told that
    store's total as read from that worker's report, so upstream slips carry downstream.
    """
    links = {l["store"]: l for l in problem.get("links", [])}
    reports = {}
    for level in range(max((l["level"] for l in links.values()), default=0) + 1):
        stores = [i for i in range(1, len(problem["stores"]) + 1)
                  if links.get(i, {"level": 0})["level"] == level]
        requests = []
        for i in stores:
            content = f"{instruction}\n\n{problem['stores'][i - 1]}"
            if i in links:
                # Stated before the ledger, workers began their running total with this number.
                source = links[i]["source"]
                total = store_total(problem, reports[source])
                given = (f"Store {source}'s total is {total}." if total is not None else
                         f"Store {source}'s total is unknown: its report is missing.")
                content += (f"\n\n{given} It is not a row of this ledger; use it only to work "
                            f"out the amount of the row that refers to Store {source}.")
            messages = [{"role": "system", "content": WORK_PROMPT},
                        {"role": "user", "content": content}]
            requests.append({"messages": messages, "cap": cap, "seed": seed + i})
        replies = yield requests
        for i, request, reply in zip(stores, requests, replies):
            report = extract_report(reply["final_text"]) if reply["stop_reason"] == "stop" else None
            status = reply_status(reply, report)
            calls.append({"stage": "worker", "store": i, "level": level,
                          "messages": request["messages"], **reply, "status": status,
                          "report": report})
            reports[i] = report
    return [reports[i] for i in range(1, len(problem["stores"]) + 1)]


def given_total(source, report):
    """Tell a worker another store's total, read from that store's report."""
    match = STORE_TOTAL.search(report or "")
    given = (f"Store {source}'s total is {match.group(1)}." if match else
             f"Store {source}'s total is unknown: its report is missing.")
    return (f"\n\n{given} It is not a row of this ledger; use it only to work out the amount "
            f"of the row that refers to Store {source}.")


def linked_worker_steps(problem, instruction, budget, seed, calls, needs=None):
    """Run workers in rounds, giving each store the totals of the stores it needs.

    With needs (store to set of stores, from a planner), a worker waits for those stores. Without
    it, every worker starts at once, and a worker that writes NEED: Store j runs again once j has
    reported. Each round splits the budget left after the reconstructor's share over the stores
    still without a report, since rerun calls are not known in advance.
    """
    k = len(problem["stores"])
    discover = needs is None
    needs = {i: set(needs.get(i, ())) if needs else set() for i in range(1, k + 1)}
    system = WORK_PROMPT + DISCOVER_RULE if discover else WORK_PROMPT
    held = budget // RECONSTRUCT_SHARE
    done, runs = {}, Counter()
    for round_no in range(1, 2 * k + 2):
        ready = [i for i in range(1, k + 1) if i not in done and needs[i] <= set(done)]
        if not ready:
            break
        left = budget - held - sum(c["total_tokens"] for c in calls)
        cap = max(left, 0) // (k - len(done))
        requests = []
        for i in ready:
            content = f"{instruction}\n\n{problem['stores'][i - 1]}"
            content += "".join(given_total(j, done[j]) for j in sorted(needs[i]))
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": content}]
            requests.append({"messages": messages, "cap": cap, "seed": seed + i + k * runs[i]})
            runs[i] += 1
        replies = yield requests
        for i, request, reply in zip(ready, requests, replies):
            text = reply["final_text"] if reply["stop_reason"] == "stop" else ""
            asked = {int(j) for j in NEED.findall(text)} if discover else set()
            new = {j for j in asked if 1 <= j <= k and j != i} - needs[i]
            report = None if new else extract_report(text)
            status = "need" if new else reply_status(reply, report)
            calls.append({"stage": "worker", "store": i, "round": round_no,
                          "given": sorted(needs[i]), "need": sorted(new),
                          "messages": request["messages"], **reply, "status": status,
                          "report": report})
            if new:
                needs[i] |= new
            else:
                done[i] = report
    # A cycle of needs, or the round limit, leaves stores without a report.
    return [done.get(i) for i in range(1, k + 1)]


def link_scores(problem, found):
    """Compare (store, source) pairs a system found with the true links."""
    true = {(l["store"], l["source"]) for l in problem.get("links", [])}
    hit = len(true & found)
    return {"link_recall": hit / len(true) if true else None,
            "link_precision": hit / len(found) if found else None}


def acyclic_needs(pairs, k):
    """Keep planned (store, source) pairs in order, dropping any that would close a cycle."""
    needs = {}
    def reaches(a, b):
        stack, seen = [a], set()
        while stack:
            x = stack.pop()
            if x == b:
                return True
            if x not in seen:
                seen.add(x)
                stack.extend(needs.get(x, ()))
        return False
    for a, b in pairs:
        if 1 <= a <= k and 1 <= b <= k and a != b and not reaches(b, a):
            needs.setdefault(a, set()).add(b)
    return needs


def team_messages(system, problem, instruction, items):
    """Show labelled reports, such as ("Store 3", text) or ("Stores 1-4", text), with context."""
    body = "\n\n".join(f"{label} report:\n{r or '(no report)'}" for label, r in items)
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"Question: {problem['question']}\n\n"
                                        f"Instruction given to each worker:\n{instruction}\n\n"
                                        f"{body}"}]


def reconstruct_steps(problem, instruction, items, cap, seed, calls):
    system = RECONSTRUCT_PROMPT.format(form=problem["answer_format"])
    messages = team_messages(system, problem, instruction, items)
    (reply,) = yield [{"messages": messages, "cap": cap, "seed": seed, "job": "merge"}]
    prediction, format_ok = parse_answer(reply["final_text"], problem)
    calls.append({"stage": "reconstructor", "messages": messages, **reply,
                  "status": reply_status(reply, prediction)})
    return prediction, reply_status(reply, prediction)


def merge_steps(problem, instruction, reports, budget, seed, calls, tree):
    """Merge worker reports in one reconstructor call, or in a tree of mergers when tree is set.

    Each tree level runs its mergers in parallel, and the budget left after the workers is
    split evenly over the merge calls still to come.
    """
    def label(lo, hi):
        return f"Store {lo}" if lo == hi else f"Stores {lo}-{hi}"

    items = [((i, i), r) for i, r in enumerate(reports, start=1)]
    sizes, m = [], len(items)
    while tree and m > FAN_IN:
        m = math.ceil(m / FAN_IN)
        sizes.append(m)
    pending = sum(sizes) + 1
    seed += 1 + len(reports)
    for level, size in enumerate(sizes, start=1):
        left = budget - sum(c["total_tokens"] for c in calls)
        groups = [items[g * FAN_IN:(g + 1) * FAN_IN] for g in range(size)]
        requests = []
        for g, group in enumerate(groups):
            messages = team_messages(MERGE_PROMPT, problem, instruction,
                                     [(label(*span), r) for span, r in group])
            requests.append({"messages": messages, "cap": left // pending, "seed": seed + g,
                             "job": "merge"})
        replies = yield requests
        seed += len(groups)
        pending -= len(groups)
        merged = []
        for group, request, reply in zip(groups, requests, replies):
            report = extract_report(reply["final_text"]) if reply["stop_reason"] == "stop" else None
            span = (group[0][0][0], group[-1][0][1])
            calls.append({"stage": "merger", "level": level, "stores": list(span),
                          "messages": request["messages"], **reply,
                          "status": reply_status(reply, report), "report": report})
            merged.append((span, report))
        items = merged
    left = budget - sum(c["total_tokens"] for c in calls)
    return (yield from reconstruct_steps(
        problem, instruction, [(label(*span), r) for span, r in items], left, seed, calls))


def report_fields(calls):
    workers = [c for c in calls if c["stage"] == "worker"]
    return {"worker_reports": sum(c["report"] is not None for c in workers) / len(workers)}


def oracle_steps(problem, budget, seed):
    """Workers follow the correct instruction and code merges their reports."""
    calls = []
    cap = budget // len(problem["stores"])
    reports = yield from worker_steps(problem, problem["oracle_instruction"], cap, seed, calls)
    prediction = merge_reports(problem, reports)
    fields = report_fields(calls)
    if prediction is None:
        fields["status"] = "bad_report"
    return finish(problem, budget, calls, prediction, **fields)


def reconstruct_oracle_steps(problem, budget, seed, tree=False):
    """Workers follow the correct instruction and a model merges their reports."""
    calls = []
    held = budget // RECONSTRUCT_SHARE
    cap = (budget - held) // len(problem["stores"])
    instruction = problem["oracle_instruction"]
    reports = yield from worker_steps(problem, instruction, cap, seed, calls)
    code_merge = merge_reports(problem, reports)
    prediction, status = yield from merge_steps(
        problem, instruction, reports, budget, seed, calls, tree)
    fields = {**report_fields(calls), "code_merge_correct": code_merge == problem["answer"]}
    if status != "answered":
        fields["status"] = status
    return finish(problem, budget, calls, prediction, **fields)


def hierarchy_steps(problem, budget, seed, tree=False):
    """A planner writes the worker instruction, workers report, and a model merges the reports."""
    example = problem["stores"][0]
    messages = [{"role": "system", "content": PLAN_PROMPT},
                {"role": "user", "content": f"There are {problem['width']} stores with "
                                            f"{problem['rows']} rows each. One ledger looks like "
                                            f"this:\n{example}\n\nQuestion: {problem['question']}"}]
    (reply,) = yield [{"messages": messages, "cap": budget // PLAN_SHARE, "seed": seed,
                       "job": "plan"}]
    calls = [{"stage": "planner", "messages": messages, **reply}]
    instruction = reply["final_text"] if reply["stop_reason"] == "stop" else ""
    if not instruction:
        status = "invalid_plan" if reply["stop_reason"] == "stop" else reply["stop_reason"]
        return finish(problem, budget, calls, None, status=status, failed_stage="planner")
    held = budget // RECONSTRUCT_SHARE
    cap = (budget - reply["total_tokens"] - held) // len(problem["stores"])
    reports = yield from worker_steps(problem, instruction, cap, seed, calls)
    prediction, status = yield from merge_steps(
        problem, instruction, reports, budget, seed, calls, tree)
    fields = {**report_fields(calls), "instruction": instruction}
    if status != "answered":
        fields["status"] = status
    return finish(problem, budget, calls, prediction, **fields)


def schedule_steps(problem, budget, seed):
    """A planner reads every ledger, writes the instruction and which stores need which, workers
    run in that order, and a model merges the reports."""
    messages = [{"role": "system", "content": SCHEDULE_PROMPT},
                {"role": "user", "content": problem["prompt"]}]
    # The planner reads all ledgers, so B/8 would not even cover its prompt at 16 stores.
    (reply,) = yield [{"messages": messages, "cap": budget // 4, "seed": seed, "job": "plan"}]
    calls = [{"stage": "planner", "messages": messages, **reply}]
    text = reply["final_text"] if reply["stop_reason"] == "stop" else ""
    pairs = [(int(a), int(b)) for a, b in NEEDS_LINE.findall(text)]
    instruction = NEEDS_LINE.sub("", text).strip()
    if not instruction:
        status = "invalid_plan" if reply["stop_reason"] == "stop" else reply["stop_reason"]
        return finish(problem, budget, calls, None, status=status, failed_stage="planner")
    instruction += "\n" + STORE_TOTAL_LINE
    needs = acyclic_needs(pairs, len(problem["stores"]))
    reports = yield from linked_worker_steps(problem, instruction, budget, seed, calls, needs)
    prediction, status = yield from merge_steps(
        problem, instruction, reports, budget, seed, calls, False)
    found = {(a, b) for a, bs in needs.items() for b in bs}
    fields = {"worker_reports": sum(r is not None for r in reports) / len(reports),
              **link_scores(problem, found), "instruction": instruction,
              "planned_needs": sorted(found)}
    if status != "answered":
        fields["status"] = status
    return finish(problem, budget, calls, prediction, **fields)


def discover_steps(problem, budget, seed):
    """A planner sees one ledger as in hierarchy, and workers ask for the totals they need."""
    example = problem["stores"][0]
    messages = [{"role": "system", "content": PLAN_PROMPT},
                {"role": "user", "content": f"There are {problem['width']} stores with "
                                            f"{problem['rows']} rows each. One ledger looks like "
                                            f"this:\n{example}\n\nQuestion: {problem['question']}"}]
    (reply,) = yield [{"messages": messages, "cap": budget // PLAN_SHARE, "seed": seed,
                       "job": "plan"}]
    calls = [{"stage": "planner", "messages": messages, **reply}]
    instruction = reply["final_text"] if reply["stop_reason"] == "stop" else ""
    if not instruction:
        status = "invalid_plan" if reply["stop_reason"] == "stop" else reply["stop_reason"]
        return finish(problem, budget, calls, None, status=status, failed_stage="planner")
    instruction += "\n" + STORE_TOTAL_LINE
    reports = yield from linked_worker_steps(problem, instruction, budget, seed, calls)
    prediction, status = yield from merge_steps(
        problem, instruction, reports, budget, seed, calls, False)
    workers = [c for c in calls if c["stage"] == "worker"]
    found = {(c["store"], j) for c in workers for j in c["need"]}
    # Rerun calls would count as missing reports, so count the stores that ended with one.
    fields = {"worker_reports": sum(r is not None for r in reports) / len(reports),
              **link_scores(problem, found), "instruction": instruction,
              "worker_calls": len(workers)}
    if status != "answered":
        fields["status"] = status
    return finish(problem, budget, calls, prediction, **fields)


def ceo_steps(problem, budget, seed):
    """A leader plans for managers, each manager plans for and merges one group of FAN_IN stores'
    workers, and the leader merges the group reports.

    The budget left after the leader's plan and merge share is split evenly over the groups.
    Within a group, the manager's plan and merge each get a MANAGER_SHARE share and the workers
    split the rest.
    """
    k = len(problem["stores"])
    groups = [list(range(g + 1, min(g + FAN_IN, k) + 1)) for g in range(0, k, FAN_IN)]
    labels = [f"Store {g[0]}" if len(g) == 1 else f"Stores {g[0]}-{g[-1]}" for g in groups]
    messages = [{"role": "system", "content": CEO_PLAN_PROMPT},
                {"role": "user", "content": f"There are {k} stores with {problem['rows']} rows "
                                            f"each, in {len(groups)} groups of up to {FAN_IN} "
                                            f"stores. One ledger looks like this:\n"
                                            f"{problem['stores'][0]}\n\n"
                                            f"Question: {problem['question']}"}]
    (reply,) = yield [{"messages": messages, "cap": budget // PLAN_SHARE, "seed": seed,
                       "job": "plan"}]
    calls = [{"stage": "ceo_plan", "messages": messages, **reply}]
    lead = reply["final_text"] if reply["stop_reason"] == "stop" else ""
    if not lead:
        status = "invalid_plan" if reply["stop_reason"] == "stop" else reply["stop_reason"]
        return finish(problem, budget, calls, None, status=status, failed_stage="ceo_plan")
    share = (budget - reply["total_tokens"] - budget // CEO_MERGE_SHARE) // len(groups)
    seed += 1

    requests = []
    for g, (stores, label) in enumerate(zip(groups, labels)):
        messages = [{"role": "system", "content": MANAGER_PLAN_PROMPT},
                    {"role": "user", "content": f"Question: {problem['question']}\n\n"
                                                f"Your group is {label}.\n\nYour leader's "
                                                f"instruction for the group report:\n{lead}\n\n"
                                                f"One of your group's ledgers looks like this:\n"
                                                f"{problem['stores'][stores[0] - 1]}"}]
        requests.append({"messages": messages, "cap": share // MANAGER_SHARE, "seed": seed + g,
                         "job": "manager"})
    replies = yield requests
    seed += len(groups)
    plans, spent = [], []
    for g, (request, reply) in enumerate(zip(requests, replies)):
        plan = reply["final_text"] if reply["stop_reason"] == "stop" else ""
        calls.append({"stage": "manager_plan", "group": g + 1, "messages": request["messages"],
                      **reply, "status": "planned" if plan else reply["stop_reason"]})
        plans.append(plan)
        spent.append(reply["total_tokens"])

    # Every group's workers run in the same round; a group whose plan failed has none.
    requests, owners = [], []
    for g, stores in enumerate(groups):
        if not plans[g]:
            continue
        cap = (share - spent[g] - share // MANAGER_SHARE) // len(stores)
        for i in stores:
            messages = [{"role": "system", "content": WORK_PROMPT},
                        {"role": "user", "content": f"{plans[g]}\n\n{problem['stores'][i - 1]}"}]
            requests.append({"messages": messages, "cap": cap, "seed": seed + i})
            owners.append((g, i))
    # A round with no calls would have no slowest call to time, so it is skipped.
    replies = (yield requests) if requests else []
    seed += k
    reports = {}
    for (g, i), request, reply in zip(owners, requests, replies):
        report = extract_report(reply["final_text"]) if reply["stop_reason"] == "stop" else None
        calls.append({"stage": "worker", "group": g + 1, "store": i,
                      "messages": request["messages"], **reply,
                      "status": reply_status(reply, report), "report": report})
        reports[i] = report
        spent[g] += reply["total_tokens"]

    requests, merging = [], []
    for g, stores in enumerate(groups):
        if not plans[g]:
            continue
        body = "\n\n".join(f"Store {i} report:\n{reports[i] or '(no report)'}" for i in stores)
        messages = [{"role": "system", "content": MANAGER_MERGE_PROMPT},
                    {"role": "user", "content": f"Question: {problem['question']}\n\nYour "
                                                f"leader's instruction for the group report:\n"
                                                f"{lead}\n\nYour instruction to the workers:\n"
                                                f"{plans[g]}\n\n{body}"}]
        requests.append({"messages": messages, "cap": share - spent[g], "seed": seed + g,
                         "job": "manager"})
        merging.append(g)
    replies = (yield requests) if requests else []
    seed += len(groups)
    group_reports = [None] * len(groups)
    for g, request, reply in zip(merging, requests, replies):
        report = extract_report(reply["final_text"]) if reply["stop_reason"] == "stop" else None
        calls.append({"stage": "manager_merge", "group": g + 1, "messages": request["messages"],
                      **reply, "status": reply_status(reply, report), "report": report})
        group_reports[g] = report

    body = "\n\n".join(f"{label} group report:\n{r or '(no report)'}"
                        for label, r in zip(labels, group_reports))
    messages = [{"role": "system", "content": CEO_MERGE_PROMPT.format(form=problem["answer_format"])},
                {"role": "user", "content": f"Question: {problem['question']}\n\nInstruction "
                                            f"given to each manager:\n{lead}\n\n{body}"}]
    left = budget - sum(c["total_tokens"] for c in calls)
    (reply,) = yield [{"messages": messages, "cap": left, "seed": seed, "job": "merge"}]
    prediction, format_ok = parse_answer(reply["final_text"], problem)
    status = reply_status(reply, prediction)
    calls.append({"stage": "ceo_merge", "messages": messages, **reply, "status": status})
    fields = {"worker_reports": sum(r is not None for r in reports.values()) / k,
              "group_reports": sum(r is not None for r in group_reports) / len(groups),
              "instruction": lead}
    if status != "answered":
        fields["status"] = status
    return finish(problem, budget, calls, prediction, **fields)


def with_jobs(steps, jobs):
    """Send requests tagged with a job (plan, merge, manager) to that job's model, if one is set."""
    replies = None
    while True:
        try:
            requests = steps.send(replies)
        except StopIteration as done:
            return done.value
        replies = yield [{**r, **jobs.get(r.get("job"), {})} for r in requests]


SOLVERS = {
    "single_agent": single_steps,
    "ensemble": ensemble_steps,
    "hierarchy_oracle": oracle_steps,
    "hierarchy_reconstruct": reconstruct_oracle_steps,
    "hierarchy": hierarchy_steps,
    "hierarchy_tree_reconstruct": functools.partial(reconstruct_oracle_steps, tree=True),
    "hierarchy_tree": functools.partial(hierarchy_steps, tree=True),
    "hierarchy_schedule": schedule_steps,
    "hierarchy_discover": discover_steps,
    "hierarchy_ceo": ceo_steps,
}


def read_problems(path):
    problems = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not all(key in record for key in (
                "group_id", "question_type", "stores", "targets", "answer", "oracle_instruction")):
            raise ValueError(f"{path}:{line_no}: expected a record from generate_ledger.py")
        problems.append(record)
    if not problems:
        raise ValueError(f"{path}: no problems to run")
    return problems


def problem_key(record):
    return record["width"], record["group_id"], record["question_type"], record.get("depth")


def print_summary(results, path):
    cells = {}
    for r in results:
        cells.setdefault((r["width"], r.get("depth", -1), r["question_type"]), []).append(r)
    for (width, depth, kind), rows in sorted(cells.items()):
        n = len(rows)
        where = f"{width} stores" + (f" depth {depth}" if depth >= 0 else "")
        print(f"{where}, {kind}: {sum(r['correct'] for r in rows)}/{n} correct, "
              f"target accuracy {sum(r['target_accuracy'] for r in rows) / n:.2f}, "
              f"{sum(r['total_tokens'] for r in rows) / n:.0f} tokens, "
              f"{sum(r['critical_output_tokens'] for r in rows) / n:.0f} critical output tokens")
    print(f"Saved {len(results)} results to {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problems", type=Path, help="JSONL file from generate_ledger.py")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="logs go to OUT_DIR/<model name>/b<budget>/<system>.jsonl")
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(SYSTEMS))
    parser.add_argument("--budgets", nargs="+", type=int, default=[8000],
                        help="total tokens per problem, one run per budget")
    parser.add_argument("--seed", type=int, default=42, help="starting sampling seed for each problem")
    parser.add_argument("--model", default="google/gemma-3-4b-it", help="OpenRouter model ID")
    parser.add_argument("--provider", default="deepinfra/bf16",
                        help="OpenRouter endpoint slug; requests never fall back to another")
    parser.add_argument("--concurrency", type=int, default=32, help="simultaneous API requests")
    parser.add_argument("--problem-threads", type=int, default=8, help="problems solved at once")
    for job, who in (("plan", "planners, including the CEO"),
                     ("merge", "reconstructors, mergers, and the CEO's merge"),
                     ("manager", "hierarchy_ceo managers")):
        parser.add_argument(f"--{job}-model", help=f"OpenRouter model ID for {who}")
        parser.add_argument(f"--{job}-provider", help=f"endpoint slug for --{job}-model")
    parser.add_argument("--resume", action="store_true",
                        help="continue existing logs, skipping finished problems")
    args = parser.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        parser.error("set OPENROUTER_API_KEY")
    if args.concurrency < 1 or args.problem_threads < 1:
        parser.error("--concurrency and --problem-threads must be positive")
    try:
        problems = read_problems(args.problems)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    # The linked store's total is read from the oracle report format, which a planner may not use.
    planned = {"hierarchy", "hierarchy_tree", "hierarchy_ceo"} & set(args.systems)
    if planned and any(p.get("links") for p in problems):
        parser.error(f"{', '.join(sorted(planned))} cannot run problems with linked rows")

    jobs = {}
    for job in ("plan", "merge", "manager"):
        model, provider = getattr(args, f"{job}_model"), getattr(args, f"{job}_provider")
        if provider and not model:
            parser.error(f"--{job}-provider needs --{job}-model")
        if model:
            jobs[job] = {"model": model, "provider": provider or args.provider}
    # Runs with other models for some jobs get their own directory, named after those models.
    run_name = args.model.split("/")[-1] + "".join(
        f"_{job}-{jobs[job]['model'].split('/')[-1]}" for job in sorted(jobs))

    chat = OpenRouterChat(args.model, args.provider, key, tokenizer_file(), args.concurrency)
    lock = threading.Lock()
    for budget in args.budgets:
        for system in args.systems:
            path = args.out_dir / run_name / f"b{budget}" / f"{system}.jsonl"
            settings = {"model": args.model, "provider": args.provider, "budget": budget,
                        "seed": args.seed, "backend": "openrouter", "sampling": SAMPLING}
            if jobs:
                settings["jobs"] = jobs
            try:
                finished = resume_log(path, settings, args.resume, key=problem_key)
            except ValueError as error:
                parser.error(str(error))
            path.parent.mkdir(parents=True, exist_ok=True)
            remaining = [p for p in problems if problem_key(p) not in finished]
            print(f"=== {system}, budget {budget}, {len(remaining)} problems", flush=True)
            results = []
            with ThreadPoolExecutor(args.problem_threads) as pool:
                futures = [pool.submit(run_problem,
                                       with_jobs(SOLVERS[system](p, budget, args.seed), jobs), chat)
                           for p in remaining]
                for future in as_completed(futures):
                    result = {**future.result(), **settings}
                    with lock:
                        with path.open("a", encoding="utf-8") as log:
                            print(json.dumps(result), file=log)
                    results.append(result)
                    depth = f" depth {result['depth']}" if "depth" in result else ""
                    print(f"{result['width']} stores{depth} group {result['group_id']} "
                          f"{result['question_type']}: {result['status']}, "
                          f"{result['total_tokens']} tokens, {result['critical_seconds']}s",
                          flush=True)
            if results:
                print_summary(results, path)


if __name__ == "__main__":
    main()
