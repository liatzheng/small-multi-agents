"""Run the single agent, ensemble, and hierarchies on ledger aggregation problems through OpenRouter."""

import argparse
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
SYSTEMS = ("single_agent", "ensemble", "hierarchy_oracle", "hierarchy_reconstruct", "hierarchy")
# The reconstructor's share is held back from the workers so they cannot starve it.
RECONSTRUCT_SHARE = 4
PLAN_SHARE = 8

REPORT_LINE = re.compile(r"^[\W_]*report[\W_]*:[*_\s]*", re.I | re.M)
ANSWER_LINE = re.compile(r"answer[*_\s]*:[*_\s]*(.+)$", re.I | re.M)
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
PAIR = re.compile(r"([A-Za-z]+)\**\s*[=:]\s*\**\s*(-?\d+)")


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
    matches = list(REPORT_LINE.finditer(text))
    if not matches:
        return None
    report = text[matches[-1].end():].strip().strip("*`").strip()
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


def worker_steps(problem, instruction, cap, seed, calls):
    """Give every store's ledger to its own worker in one parallel round."""
    requests = []
    for i, store in enumerate(problem["stores"]):
        messages = [{"role": "system", "content": WORK_PROMPT},
                    {"role": "user", "content": f"{instruction}\n\n{store}"}]
        requests.append({"messages": messages, "cap": cap, "seed": seed + 1 + i})
    replies = yield requests
    reports = []
    for i, (request, reply) in enumerate(zip(requests, replies), start=1):
        report = extract_report(reply["final_text"]) if reply["stop_reason"] == "stop" else None
        status = reply_status(reply, report)
        calls.append({"stage": "worker", "store": i, "messages": request["messages"], **reply,
                      "status": status, "report": report})
        reports.append(report)
    return reports


def reconstruct_steps(problem, instruction, reports, cap, seed, calls):
    form = problem["answer_format"]
    body = "\n\n".join(f"Store {i} report:\n{r or '(no report)'}"
                       for i, r in enumerate(reports, start=1))
    messages = [{"role": "system", "content": RECONSTRUCT_PROMPT.format(form=form)},
                {"role": "user", "content": f"Question: {problem['question']}\n\n"
                                            f"Instruction given to each worker:\n{instruction}\n\n"
                                            f"{body}"}]
    (reply,) = yield [{"messages": messages, "cap": cap,
                       "seed": seed + 1 + len(problem["stores"])}]
    prediction, format_ok = parse_answer(reply["final_text"], problem)
    calls.append({"stage": "reconstructor", "messages": messages, **reply,
                  "status": reply_status(reply, prediction)})
    return prediction, reply_status(reply, prediction)


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


def reconstruct_oracle_steps(problem, budget, seed):
    """Workers follow the correct instruction and a model merges their reports."""
    calls = []
    held = budget // RECONSTRUCT_SHARE
    cap = (budget - held) // len(problem["stores"])
    instruction = problem["oracle_instruction"]
    reports = yield from worker_steps(problem, instruction, cap, seed, calls)
    code_merge = merge_reports(problem, reports)
    left = budget - sum(c["total_tokens"] for c in calls)
    prediction, status = yield from reconstruct_steps(problem, instruction, reports, left, seed, calls)
    fields = {**report_fields(calls), "code_merge_correct": code_merge == problem["answer"]}
    if status != "answered":
        fields["status"] = status
    return finish(problem, budget, calls, prediction, **fields)


def hierarchy_steps(problem, budget, seed):
    """A planner writes the worker instruction, workers report, and a model merges the reports."""
    example = problem["stores"][0]
    messages = [{"role": "system", "content": PLAN_PROMPT},
                {"role": "user", "content": f"There are {problem['width']} stores with "
                                            f"{problem['rows']} rows each. One ledger looks like "
                                            f"this:\n{example}\n\nQuestion: {problem['question']}"}]
    (reply,) = yield [{"messages": messages, "cap": budget // PLAN_SHARE, "seed": seed}]
    calls = [{"stage": "planner", "messages": messages, **reply}]
    instruction = reply["final_text"] if reply["stop_reason"] == "stop" else ""
    if not instruction:
        status = "invalid_plan" if reply["stop_reason"] == "stop" else reply["stop_reason"]
        return finish(problem, budget, calls, None, status=status, failed_stage="planner")
    held = budget // RECONSTRUCT_SHARE
    cap = (budget - reply["total_tokens"] - held) // len(problem["stores"])
    reports = yield from worker_steps(problem, instruction, cap, seed, calls)
    left = budget - sum(c["total_tokens"] for c in calls)
    prediction, status = yield from reconstruct_steps(problem, instruction, reports, left, seed, calls)
    fields = {**report_fields(calls), "instruction": instruction}
    if status != "answered":
        fields["status"] = status
    return finish(problem, budget, calls, prediction, **fields)


SOLVERS = {
    "single_agent": single_steps,
    "ensemble": ensemble_steps,
    "hierarchy_oracle": oracle_steps,
    "hierarchy_reconstruct": reconstruct_oracle_steps,
    "hierarchy": hierarchy_steps,
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
    return record["width"], record["group_id"], record["question_type"]


def print_summary(results, path):
    cells = {}
    for r in results:
        cells.setdefault((r["width"], r["question_type"]), []).append(r)
    for (width, kind), rows in sorted(cells.items()):
        n = len(rows)
        print(f"{width} stores, {kind}: {sum(r['correct'] for r in rows)}/{n} correct, "
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

    chat = OpenRouterChat(args.model, args.provider, key, tokenizer_file(), args.concurrency)
    lock = threading.Lock()
    for budget in args.budgets:
        for system in args.systems:
            path = args.out_dir / args.model.split("/")[-1] / f"b{budget}" / f"{system}.jsonl"
            settings = {"model": args.model, "provider": args.provider, "budget": budget,
                        "seed": args.seed, "backend": "openrouter", "sampling": SAMPLING}
            try:
                finished = resume_log(path, settings, args.resume, key=problem_key)
            except ValueError as error:
                parser.error(str(error))
            path.parent.mkdir(parents=True, exist_ok=True)
            remaining = [p for p in problems if problem_key(p) not in finished]
            print(f"=== {system}, budget {budget}, {len(remaining)} problems", flush=True)
            results = []
            with ThreadPoolExecutor(args.problem_threads) as pool:
                futures = [pool.submit(run_problem, SOLVERS[system](p, budget, args.seed), chat)
                           for p in remaining]
                for future in as_completed(futures):
                    result = {**future.result(), **settings}
                    with lock:
                        with path.open("a", encoding="utf-8") as log:
                            print(json.dumps(result), file=log)
                    results.append(result)
                    print(f"{result['width']} stores group {result['group_id']} "
                          f"{result['question_type']}: {result['status']}, "
                          f"{result['total_tokens']} tokens, {result['critical_seconds']}s",
                          flush=True)
            if results:
                print_summary(results, path)


if __name__ == "__main__":
    main()
