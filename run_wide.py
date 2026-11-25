"""Run the single agent, ensemble, and hierarchies on wide chain problems through OpenRouter."""

import argparse
import functools
import http.client
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

API_URL = "https://openrouter.ai/api/v1/chat/completions"
# All Gemma 3 sizes share this tokenizer. The unsloth mirror is ungated, unlike google/gemma-3-*.
TOKENIZER_URL = ("https://huggingface.co/unsloth/gemma-3-4b-it/resolve/"
                 "bf46152c47f5dd20b896357cb51abc4c03b8ee8c/tokenizer.json")
# Gemma 3's generation_config.json sets top_k 64 and top_p 0.95 with the default temperature 1.0.
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
# DeepInfra's standard Gemma endpoints accept at most this many completion tokens.
MAX_OUTPUT = 16384
RETRY_CODES = (408, 429, 500, 502, 503, 504)

ANSWER_PROMPT = (
    "Work through the problem step by step. End your response with the final answer on its "
    "own line: the requested values as one tuple of integers in parentheses, in the requested order."
)
WORK_PROMPT = (
    "You are a worker on a team. Solve your assignment step by step. End your response with "
    "one line of the form <variable> = <value> for the variable you were asked to find."
)
PLAN_PROMPT = (
    "You are the orchestrator of a team of workers. Do not compute any values. For each "
    "requested variable, in the requested order, write an assignment for one worker. A worker "
    "sees only its assignment, not the problem, so the assignment must contain everything it "
    "needs: the given values it starts from and the equations to evaluate, in order. If the "
    "chain for a variable starts from another requested variable, leave that variable's "
    "equations out and set \"after\" to it; its value will be passed on when its worker finishes. "
    "Return only a JSON list with one object per requested variable: "
    '{"target": "<variable>", "after": "<requested variable>" or null, "assignment": "<text>"}.'
)
SYSTEMS = ("single_agent", "ensemble", "hierarchy_oracle", "hierarchy")


class OpenRouterChat:
    """Answer batches of chat requests, each under a cap on its prompt plus output tokens."""

    def __init__(self, model, provider, key, tokenizer_path, concurrency):
        from tokenizers import Tokenizer

        self.model, self.provider, self.key = model, provider, key
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.pool = ThreadPoolExecutor(concurrency)

    def count(self, messages):
        # Gemma's chat template folds the system message into the first user turn.
        system, user = messages[0]["content"], messages[1]["content"]
        text = (f"<bos><start_of_turn>user\n{system}\n\n{user.strip()}<end_of_turn>\n"
                "<start_of_turn>model\n")
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def generate(self, requests):
        return list(self.pool.map(self.call, requests))

    def call(self, request):
        cap, seed = request["cap"], request["seed"]
        local_input = self.count(request["messages"])
        if local_input >= cap:
            return {
                "stop_reason": "prompt_too_long", "required_input_tokens": local_input,
                "final_text": "", "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                "budget": cap, "sampling_seed": seed, "seconds": 0.0,
            }
        max_tokens = min(cap - local_input, MAX_OUTPUT)
        # A request can name its own model, as a hierarchy planner of another size does.
        model = request.get("model", self.model)
        provider = request.get("provider", self.provider)
        data, seconds = self.post({
            "model": model, "messages": request["messages"], "max_tokens": max_tokens,
            "seed": seed, **SAMPLING,
            "provider": {"order": [provider], "allow_fallbacks": False,
                         "require_parameters": True},
        })
        choice, usage = data["choices"][0], data["usage"]
        reason = choice.get("finish_reason")
        return {
            "stop_reason": {"stop": "stop", "length": "length_limit"}.get(reason, reason),
            "final_text": (choice["message"].get("content") or "").strip(),
            # Budgets are charged with the provider's counts; the local count only sets max_tokens.
            "input_tokens": usage["prompt_tokens"], "local_input_tokens": local_input,
            "output_tokens": usage["completion_tokens"],
            "total_tokens": usage["prompt_tokens"] + usage["completion_tokens"],
            "budget": cap, "max_tokens": max_tokens, "sampling_seed": seed,
            "seconds": seconds, "model": model, "provider": data.get("provider"),
        }

    def post(self, body):
        payload = json.dumps(body).encode()
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        for attempt in range(8):
            began = time.perf_counter()
            try:
                request = urllib.request.Request(API_URL, payload, headers)
                with urllib.request.urlopen(request, timeout=600) as response:
                    data = json.load(response)
            except urllib.error.HTTPError as error:
                detail = error.read().decode(errors="replace")[:500]
                if error.code not in RETRY_CODES:
                    raise RuntimeError(f"OpenRouter HTTP {error.code}: {detail}") from error
                problem = f"HTTP {error.code}: {detail}"
            # OSError covers URLError and timeouts, plus socket errors such as "No route to
            # host" raised while reading a response after the network drops. HTTPException
            # covers a response cut off mid-body (IncompleteRead).
            except (OSError, http.client.HTTPException) as error:
                problem = repr(error)
            else:
                choices = data.get("choices") or []
                if choices and choices[0].get("finish_reason") != "error":
                    return data, round(time.perf_counter() - began, 2)
                problem = json.dumps(data.get("error") or choices)[:500]
            wait = min(60, 2 ** attempt)
            print(f"Retrying in {wait}s after {problem}", flush=True)
            time.sleep(wait)
        raise RuntimeError(f"OpenRouter failed 8 times; last error: {problem}")


def parse_tuple(text, width):
    """Read the last tuple of width integers and flag whether the last line is just that tuple."""
    pattern = r"\(\s*" + r"\s*,\s*".join([r"(-?\d+)"] * width) + r"\s*\)"
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None, False
    lines = text.splitlines()
    format_ok = bool(lines) and re.fullmatch(pattern, lines[-1].strip()) is not None
    return [int(n) for n in matches[-1].groups()], format_ok


def parse_report(text):
    """Read the value on a worker's last line, such as 'v12 = 40' or '**v12 = 40**.'"""
    lines = text.splitlines()
    last = lines[-1].strip("*`$. ") if lines else ""
    match = re.search(r"(?:=|\bis)\s*\**\s*(-?\d+)$", last)
    return int(match.group(1)) if match else None


def answer_messages(problem):
    return [
        {"role": "system", "content": ANSWER_PROMPT},
        {"role": "user", "content": problem["prompt"]},
    ]


def finish(problem, budget, calls, prediction, **fields):
    """Score a prediction with None for unanswered targets and total every call's tokens."""
    answer = problem["answer"]
    hits = [p == a for p, a in zip(prediction or [None] * len(answer), answer)]
    if "status" not in fields:
        fields["status"] = "correct" if all(hits) else "wrong_answer"
    input_tokens = sum(c["input_tokens"] for c in calls)
    output_tokens = sum(c["output_tokens"] for c in calls)
    return {
        **problem, **fields, "prediction": prediction, "correct": all(hits),
        "target_accuracy": sum(hits) / len(hits), "calls": calls, "budget": budget,
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def reply_status(reply, prediction):
    if reply["stop_reason"] != "stop":
        return reply["stop_reason"]
    return "format_error" if prediction is None else "answered"


def single_steps(problem, budget, seed):
    (reply,) = yield [{"messages": answer_messages(problem), "cap": budget, "seed": seed}]
    prediction, format_ok = parse_tuple(reply["final_text"], len(problem["targets"]))
    status = reply_status(reply, prediction)
    fields = {"format_ok": format_ok}
    if status != "answered":
        fields["status"] = status
    return finish(problem, budget, [reply], prediction, **fields)


def ensemble_steps(problem, budget, seed, attempts=3):
    base, remainder = divmod(budget, attempts)
    caps = [base] * (attempts - remainder) + [base + 1] * remainder
    replies = yield [
        {"messages": answer_messages(problem), "cap": cap, "seed": seed + i}
        for i, cap in enumerate(caps)
    ]
    width = len(problem["targets"])
    votes = []
    for reply in replies:
        prediction, _ = parse_tuple(reply["final_text"], width)
        reply["status"] = reply_status(reply, prediction)
        reply["vote"] = prediction if reply["status"] == "answered" else None
        votes.append(reply["vote"] or [None] * width)
    # Each target takes the value at least two attempts agree on, so attempts that slip on
    # different targets can still produce a fully correct answer.
    prediction = []
    for column in zip(*votes):
        counts = Counter(v for v in column if v is not None).most_common(1)
        prediction.append(counts[0][0] if counts and counts[0][1] * 2 > attempts else None)
    any_correct = [any(col_value == a for col_value in column)
                   for column, a in zip(zip(*votes), problem["answer"])]
    fields = {"any_correct_targets": sum(any_correct) / width}
    if None in prediction:
        fields["status"] = "no_majority"
    return finish(problem, budget, list(replies), prediction, **fields)


def worker_steps(assignments, cap, seed, modulus, calls):
    """Run workers in rounds, each as soon as the worker it is after has reported."""
    values, workers = {}, {}
    while len(workers) < len(assignments):
        requests, sent = [], []
        for i, a in enumerate(assignments):
            if a["target"] in workers or (a["after"] is not None and a["after"] not in workers):
                continue
            text = a["text"]
            if a["after"] is not None:
                if values[a["after"]] is None:
                    workers[a["target"]] = {"status": "missing_input"}
                    values[a["target"]] = None
                    continue
                text += f"\n\nAnother worker computed {a['after']} = {values[a['after']]}."
            text += f"\nUse remainders from 0 to {modulus - 1}."
            messages = [{"role": "system", "content": WORK_PROMPT},
                        {"role": "user", "content": text}]
            # Seeds follow plan positions, so they do not depend on how workers are scheduled.
            requests.append({"messages": messages, "cap": cap, "seed": seed + 1 + i})
            sent.append((a["target"], messages))
        if not requests:
            continue
        replies = yield requests
        for (target, messages), reply in zip(sent, replies):
            value = parse_report(reply["final_text"]) if reply["stop_reason"] == "stop" else None
            status = reply_status(reply, value)
            calls.append({"stage": "worker", "target": target, "messages": messages, **reply,
                          "status": status, "value": value})
            workers[target] = {"status": status}
            values[target] = value
    return values, workers


def oracle_steps(problem, budget, seed):
    """Workers get assignments built from the generator's plan, so planning cannot fail."""
    assignments = []
    for step in problem["plan"]:
        text = "\n".join(step["equations"]) + f"\nFind {step['target']}."
        if step["start"] is not None:
            text = f"Given {step['start'][0]} = {step['start'][1]}.\n" + text
        assignments.append({"target": step["target"], "after": step["after"], "text": text})
    calls = []
    values, _ = yield from worker_steps(
        assignments, budget // len(assignments), seed, problem["modulus"], calls)
    return finish(problem, budget, calls, [values[t] for t in problem["targets"]])


def parse_plan(text, targets):
    """Return worker assignments from the planner's JSON list, or None if it is unusable."""
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        plan = json.loads(body)
    except ValueError:
        try:
            plan = json.loads(body[body.find("["):body.rfind("]") + 1])
        except ValueError:
            return None
    if not (isinstance(plan, list) and len(plan) == len(targets)):
        return None
    for step, target in zip(plan, targets):
        if not (isinstance(step, dict) and step.get("target") == target
                and isinstance(step.get("assignment"), str) and step["assignment"].strip()):
            return None
    # Planners often put an intermediate variable in "after". Dropping it lets an otherwise
    # complete assignment run, so the score reflects the plan rather than this one field.
    after = {}
    for step in plan:
        prev = step.get("after")
        after[step["target"]] = prev if prev in targets and prev != step["target"] else None
    for target in targets:
        seen = set()
        while target is not None:
            if target in seen:
                return None
            seen.add(target)
            target = after[target]
    return [{"target": s["target"], "after": after[s["target"]], "text": s["assignment"],
             "dropped_after": s.get("after") is not None and after[s["target"]] is None}
            for s in plan]


def hierarchy_steps(problem, budget, seed, planner=None):
    """A planner writes self-contained assignments; workers solve them; code merges the values.

    planner optionally names a "model" and "provider" for the planner call only.
    """
    messages = [{"role": "system", "content": PLAN_PROMPT},
                {"role": "user", "content": problem["prompt"]}]
    (reply,) = yield [{"messages": messages, "cap": budget // 4, "seed": seed, **(planner or {})}]
    calls = [{"stage": "planner", "messages": messages, **reply}]
    targets = problem["targets"]
    assignments = parse_plan(reply["final_text"], targets) if reply["stop_reason"] == "stop" else None
    if assignments is None:
        status = "invalid_plan" if reply["stop_reason"] == "stop" else reply["stop_reason"]
        return finish(problem, budget, calls, None, status=status, failed_stage="planner")
    # The oracle's assignments end by naming the target, and workers otherwise report whichever
    # variable they computed last, so model-planned assignments get the same line.
    for a in assignments:
        a["text"] += f"\nFind {a['target']}."
    oracle_after = {step["target"]: step["after"] for step in problem["plan"]}
    plan_after_right = sum(a["after"] == oracle_after[a["target"]] for a in assignments) / len(targets)
    cap = (budget - reply["total_tokens"]) // len(targets)
    values, _ = yield from worker_steps(assignments, cap, seed, problem["modulus"], calls)
    return finish(problem, budget, calls, [values[t] for t in targets],
                  plan_after_right=plan_after_right,
                  plan_dropped_after=sum(a["dropped_after"] for a in assignments))


SOLVERS = {
    "single_agent": single_steps,
    "ensemble": ensemble_steps,
    "hierarchy_oracle": oracle_steps,
    "hierarchy": hierarchy_steps,
}


def run_problem(steps, chat):
    """Send each round of requests together and time the rounds that must run in sequence."""
    rounds, replies = [], None
    while True:
        try:
            requests = steps.send(replies)
        except StopIteration as done:
            result = done.value
            break
        replies = chat.generate(requests)
        rounds.append(replies)
    result["rounds"] = len(rounds)
    # Calls within a round run in parallel, so only the slowest one adds to the time.
    result["critical_seconds"] = round(sum(max(r["seconds"] for r in rs) for rs in rounds), 2)
    result["critical_output_tokens"] = sum(max(r["output_tokens"] for r in rs) for rs in rounds)
    return result


def read_problems(path):
    problems = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not all(key in record for key in (
                "group_id", "dependent_fraction", "prompt", "targets", "answer", "plan")):
            raise ValueError(f"{path}:{line_no}: expected a record from generate_wide.py")
        problems.append(record)
    if not problems:
        raise ValueError(f"{path}: no problems to run")
    return problems


def problem_key(record):
    return record["width"], record["group_id"], record["dependent_fraction"]


def resume_log(path, settings, resume, key=problem_key):
    """Return the keys already finished in path after trimming an interrupted write."""
    if not path.exists():
        return set()
    if not resume:
        raise ValueError(f"{path} already exists; choose a new --out-dir or pass --resume")
    data = path.read_bytes()
    keep = data.rfind(b"\n") + 1
    finished = set()
    for line_no, line in enumerate(data[:keep].splitlines(), start=1):
        record = json.loads(line)
        if any(record.get(k) != v for k, v in settings.items()):
            raise ValueError(f"{path}:{line_no}: result does not match these run settings")
        finished.add(key(record))
    with path.open("r+b") as stream:
        stream.truncate(keep)
    print(f"Resuming after {len(finished)} finished problems in {path}", flush=True)
    return finished


def tokenizer_file():
    path = Path.home() / ".cache" / "small-multi-agents" / "gemma3-tokenizer.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(TOKENIZER_URL, path)
    return path


def print_summary(results, path):
    cells = {}
    for r in results:
        cells.setdefault((r["width"], r["dependent_fraction"]), []).append(r)
    for (width, fraction), rows in sorted(cells.items()):
        n = len(rows)
        print(f"width {width}, fraction {fraction}: {sum(r['correct'] for r in rows)}/{n} all correct, "
              f"target accuracy {sum(r['target_accuracy'] for r in rows) / n:.2f}, "
              f"{sum(r['total_tokens'] for r in rows) / n:.0f} tokens, "
              f"{sum(r['critical_output_tokens'] for r in rows) / n:.0f} critical output tokens")
    print(f"Saved {len(results)} results to {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problems", type=Path, help="JSONL file from generate_wide.py")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="logs go to OUT_DIR/<model name>/b<budget>/<system>.jsonl")
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(SYSTEMS))
    parser.add_argument("--budgets", nargs="+", type=int, default=[8000],
                        help="total tokens per problem, one run per budget")
    parser.add_argument("--seed", type=int, default=42, help="starting sampling seed for each problem")
    parser.add_argument("--model", default="google/gemma-3-4b-it", help="OpenRouter model ID")
    parser.add_argument("--provider", default="deepinfra/bf16",
                        help="OpenRouter endpoint slug; requests never fall back to another")
    parser.add_argument("--planner-model",
                        help="OpenRouter model ID for the hierarchy planner (default: --model)")
    parser.add_argument("--planner-provider",
                        help="endpoint slug for --planner-model (default: --provider)")
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
    if (args.planner_model or args.planner_provider) and args.systems != ["hierarchy"]:
        parser.error("--planner-model and --planner-provider need --systems hierarchy")
    planner = None
    if args.planner_model or args.planner_provider:
        planner = {"model": args.planner_model or args.model,
                   "provider": args.planner_provider or args.provider}
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
            solve = SOLVERS[system]
            if planner:
                settings.update(planner_model=planner["model"], planner_provider=planner["provider"])
                solve = functools.partial(solve, planner=planner)
            try:
                finished = resume_log(path, settings, args.resume)
            except ValueError as error:
                parser.error(str(error))
            path.parent.mkdir(parents=True, exist_ok=True)
            remaining = [p for p in problems if problem_key(p) not in finished]
            print(f"=== {system}, budget {budget}, {len(remaining)} problems", flush=True)
            results = []
            with ThreadPoolExecutor(args.problem_threads) as pool:
                futures = [pool.submit(run_problem, solve(p, budget, args.seed), chat)
                           for p in remaining]
                for future in as_completed(futures):
                    result = {**future.result(), **settings}
                    with lock:
                        # Reopen per record so a Drive mount or an interrupted run keeps every line.
                        with path.open("a", encoding="utf-8") as log:
                            print(json.dumps(result), file=log)
                    results.append(result)
                    print(f"width {result['width']} group {result['group_id']} "
                          f"fraction {result['dependent_fraction']}: "
                          f"{result['status']}, {result['target_accuracy']:.2f} of targets, "
                          f"{result['total_tokens']} tokens, {result['critical_seconds']}s",
                          flush=True)
            if results:
                print_summary(results, path)


if __name__ == "__main__":
    main()
