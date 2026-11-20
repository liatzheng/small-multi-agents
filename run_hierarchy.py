"""Run a Qwen3-4B orchestrator and two workers under one problem-level token budget."""

import json
import re

from run_single_agent import SYSTEM_PROMPT, parse_answer, run_cli, run_hf

PLAN_PROMPT = (
    "Split the problem into two distinct subtasks, one per worker. Put any prerequisite "
    "subtask first. Write instructions, without solving them. Return only a JSON object "
    'with "first" and "second" (short assignment strings) and "pass_first" (a boolean). '
    "Set pass_first to true only if worker 2 needs worker 1's result."
)
WORK_PROMPT = (
    "Solve your assigned subtask. Your final response is a report to the orchestrator. "
    "Give the requested variable and its value in one short line."
)


class ProtocolFailure(Exception):
    def __init__(self, stage, status):
        super().__init__(f"{stage}: {status}")
        self.stage = stage
        self.status = status


def parse_plan(text):
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text.strip(), flags=re.DOTALL)
    try:
        plan = json.loads(fenced.group(1) if fenced else text)
    except ValueError:
        plan = None
    if not (
        isinstance(plan, dict)
        and all(isinstance(plan.get(k), str) and plan[k].strip() for k in ("first", "second"))
        and isinstance(plan.get("pass_first"), bool)
    ):
        raise ProtocolFailure("planner", "invalid_plan")
    return plan


def solve_hierarchy_steps(problem, budget, seed):
    calls = []
    prompt = problem["prompt"]

    def call(stage, instruction, content):
        index = len(calls)
        # Release one quarter per stage and carry forward unused tokens.
        cap = budget * (index + 1) // 4 - sum(c["total_tokens"] for c in calls)
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": content},
        ]
        (reply,) = yield [{"messages": messages, "cap": cap, "seed": seed + index}]
        calls.append({"stage": stage, "messages": messages, **reply})
        if reply["stop_reason"] != "stop":
            raise ProtocolFailure(stage, reply["stop_reason"])
        if not reply["final_text"].strip():
            raise ProtocolFailure(stage, "empty_response")
        return reply["final_text"]

    answer, prediction, format_ok, failed_stage = "", None, False, None
    try:
        plan = parse_plan((yield from call("planner", PLAN_PROMPT, prompt)))
        first = yield from call("worker_1", WORK_PROMPT, f"Problem:\n{prompt}\n\nAssignment:\n{plan['first']}")
        task = f"Problem:\n{prompt}\n\nAssignment:\n{plan['second']}"
        if plan["pass_first"]:
            task += f"\n\nWorker 1 report:\n{first}"
        second = yield from call("worker_2", WORK_PROMPT, task)
        answer = yield from call(
            "merge", SYSTEM_PROMPT,
            f"Problem:\n{prompt}\n\nWorker 1 assignment: {plan['first']}\nWorker 1 report: {first}"
            f"\n\nWorker 2 assignment: {plan['second']}\nWorker 2 report: {second}"
            "\n\nCombine the two reports into the requested final answer.",
        )
        prediction, format_ok = parse_answer(answer)
        if prediction is None:
            status, failed_stage = "format_error", "merge"
        else:
            status = "correct" if prediction == problem["answer"] else "wrong_answer"
    except ProtocolFailure as failure:
        status, failed_stage = failure.status, failure.stage

    input_tokens = sum(c["input_tokens"] for c in calls)
    output_tokens = sum(c["output_tokens"] for c in calls)
    return {
        **problem, "final_text": answer, "prediction": prediction, "format_ok": format_ok,
        "correct": status == "correct", "status": status, "failed_stage": failed_stage,
        "calls": calls, "budget": budget, "input_tokens": input_tokens,
        "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens,
    }


def solve_hierarchy(problem, model, tokenizer, budget, seed):
    return run_hf(solve_hierarchy_steps(problem, budget, seed), model, tokenizer)


if __name__ == "__main__":
    run_cli(__doc__, solve_hierarchy, "hierarchy_results.jsonl")
