"""Run one fresh Qwen3-4B chat per problem from generate_problems.py."""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-4B"
REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
SYSTEM_PROMPT = "Return your final answer as one ordered pair of integers in parentheses."


def read_problems(path):
    problems = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not (
                isinstance(record, dict)
                and isinstance(record.get("pair_id"), int)
                and record.get("kind") in ("independent", "dependent")
                and isinstance(record.get("prompt"), str)
                and isinstance(record.get("answer"), list)
                and len(record["answer"]) == 2
                and all(type(n) is int for n in record["answer"])
            ):
                raise ValueError(f"{path}:{line_no}: expected a record from generate_problems.py")
            problems.append(record)
    if not problems:
        raise ValueError(f"{path}: no problems to run")
    return problems


def final_answer(tokenizer, output_ids):
    # Qwen can emit a reasoning block before the final response.
    end_think = tokenizer.convert_tokens_to_ids("</think>")
    if end_think in output_ids:
        last_end = len(output_ids) - output_ids[::-1].index(end_think)
        output_ids = output_ids[last_end:]
    elif tokenizer.convert_tokens_to_ids("<think>") in output_ids:
        return ""
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def parse_answer(text):
    """Read the final pair and flag whether the response was just that pair."""
    pair = r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)"
    bare = re.fullmatch(pair, text)
    match = bare
    if not match and text:
        match = re.fullmatch(pair, text.splitlines()[-1].strip())
    if not match and text.count(r"\boxed") == 1:
        # Accept an explicitly boxed final answer with optional math delimiters.
        box = r"\\boxed\s*\{\s*" + pair + r"\s*\}"
        ending = r"\s*(?:[$]{1,2}|\\\)|\\\])?\s*[.!]?\Z"
        match = re.search(box + ending, text)
    prediction = [int(n) for n in match.groups()] if match else None
    return prediction, bare is not None


@torch.inference_mode()
def generate_chat(messages, model, tokenizer, budget, seed):
    """Run one fresh chat under a cap that includes its rendered prompt."""
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=True,
        return_tensors="pt", return_dict=True,
    ).to(model.device)
    input_tokens = inputs["input_ids"].shape[1]
    if input_tokens >= budget:
        # Tokenization happened locally, but no tokens were sent through the model.
        return {
            "stop_reason": "prompt_too_long", "required_input_tokens": input_tokens,
            "final_text": "", "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            "budget": budget, "sampling_seed": seed,
        }

    torch.manual_seed(seed)
    sequence = model.generate(
        **inputs, max_new_tokens=budget - input_tokens,
        do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
    )
    output_ids = sequence[0, input_tokens:].tolist()
    eos = model.generation_config.eos_token_id
    stopped = output_ids[-1] in (eos if isinstance(eos, list) else [eos])
    return {
        "stop_reason": "stop" if stopped else "length_limit",
        "final_text": final_answer(tokenizer, output_ids), "input_tokens": input_tokens,
        "output_tokens": len(output_ids), "total_tokens": input_tokens + len(output_ids),
        "budget": budget, "sampling_seed": seed,
    }


def run_hf(steps, model, tokenizer):
    """Answer a solver's model requests one at a time with Transformers."""
    replies = None
    while True:
        try:
            requests = steps.send(replies)
        except StopIteration as done:
            return {**done.value, "device": model.device.type, "dtype": str(model.dtype)}
        replies = [
            generate_chat(r["messages"], model, tokenizer, r["cap"], r["seed"]) for r in requests
        ]


def score_reply(problem, reply):
    """Score one direct answer to the problem."""
    reply = dict(reply)
    stop_reason = reply.pop("stop_reason")
    if stop_reason == "prompt_too_long":
        needed = reply["required_input_tokens"]
        raise ValueError(f"Prompt requires {needed} tokens; budget {reply['budget']} leaves no output.")
    predicted, format_ok = parse_answer(reply["final_text"])
    if stop_reason == "length_limit":
        status = "length_limit"
    elif predicted is None:
        status = "format_error"
    elif predicted == problem["answer"]:
        status = "correct"
    else:
        status = "wrong_answer"
    return {
        **problem, **reply, "prediction": predicted, "status": status,
        "correct": status == "correct", "format_ok": format_ok,
    }


def direct_messages(problem):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem["prompt"]},
    ]


def solve_steps(problem, budget, seed):
    """Yield lists of model requests and receive their replies; return the scored result."""
    (reply,) = yield [{"messages": direct_messages(problem), "cap": budget, "seed": seed}]
    return score_reply(problem, reply)


def solve(problem, model, tokenizer, budget, seed):
    return run_hf(solve_steps(problem, budget, seed), model, tokenizer)


def load_model(model_name=MODEL, revision=REVISION):
    """Load one model, using native CUDA precision and streaming GPU weights."""
    device = "cpu"
    dtype = "auto"
    options = {}
    if torch.cuda.is_available():
        device = "cuda"
        # T4 uses FP16. A100 supports native BF16.
        native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        dtype = torch.bfloat16 if native_bf16 else torch.float16
        # Accelerate loads weights onto the GPU without a full CPU copy.
        options["device_map"] = {"": "cuda:0"}
    elif torch.backends.mps.is_available():
        device = "mps"
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, torch_dtype=dtype, use_safetensors=True,
        **options,
    ).eval()
    if device != "cuda":
        model = model.to(device)
    return model, tokenizer


def read_finished(path, problems, settings):
    """Return the logged results to keep and the byte length they occupy."""
    data = path.read_bytes()
    # Every record ends with a newline, so text after the last one is an interrupted write.
    keep = data.rfind(b"\n") + 1
    finished = [json.loads(line) for line in data[:keep].splitlines()]
    if len(finished) > len(problems):
        raise ValueError(f"{path} has more results than the problem file")
    for line_no, (result, problem) in enumerate(zip(finished, problems), start=1):
        expected = {**problem, **settings}
        # Logs written before the backend field existed all came from Transformers.
        actual = {"backend": "transformers", **result}
        if any(actual.get(key) != value for key, value in expected.items()):
            raise ValueError(f"{path}:{line_no}: result does not match this problem file and run settings")
    return finished, keep


def resume_log(path, problems, settings, resume):
    """Return the finished results in an existing log after trimming an interrupted write."""
    if not path.exists():
        return []
    if not resume:
        raise ValueError(f"{path} already exists; choose a new output or pass --resume")
    finished, keep = read_finished(path, problems, settings)
    with path.open("r+b") as stream:
        stream.truncate(keep)
    print(f"Resuming after {len(finished)} finished problems in {path}", flush=True)
    return finished


def log_result(path, result):
    # Colab's Drive mount showed no new lines in files held open, so each record closes the file.
    with path.open("a", encoding="utf-8") as log:
        print(json.dumps(result), file=log)
    print(f"pair {result['pair_id']} {result['kind']}: {result['status']}, "
          f"{result['total_tokens']} tokens, {result['seconds']}s", flush=True)


def print_summary(results, path):
    counts = {"independent": [0, 0], "dependent": [0, 0]}
    for result in results:
        count = counts[result["kind"]]
        count[0] += result["correct"]
        count[1] += 1
    for kind, (correct, total) in counts.items():
        if total:
            print(f"{kind}: {correct}/{total} correct")
    print(f"Saved {len(results)} responses to {path}", flush=True)


def run_cli(description, solve_problem, default_output):
    """Load Qwen once and run the selected evaluation on each JSONL problem."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("problems", type=Path, help="JSONL file from the problem generator")
    parser.add_argument("--output", type=Path, default=Path(default_output))
    parser.add_argument("--budget", type=int, default=4000, help="total tokens per problem")
    parser.add_argument("--seed", type=int, default=42, help="starting sampling seed for each problem")
    parser.add_argument("--model", default=MODEL, help="Hugging Face model ID")
    parser.add_argument("--revision", help=f"model commit (required unless --model is {MODEL})")
    parser.add_argument("--resume", action="store_true",
                        help="continue an existing --output file from the first unfinished problem")
    args = parser.parse_args()
    if not 1 <= args.budget <= 32768:
        parser.error("--budget must be between 1 and 32768")
    if args.revision is None:
        if args.model != MODEL:
            parser.error("--revision is required with --model so the weights stay pinned")
        args.revision = REVISION
    try:
        problems = read_problems(args.problems)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    settings = {
        "model": args.model, "revision": args.revision, "budget": args.budget, "seed": args.seed,
        "backend": "transformers",
    }
    try:
        finished = resume_log(args.output, problems, settings, args.resume)
    except ValueError as error:
        parser.error(str(error))

    results = list(finished)
    remaining = problems[len(finished):]
    if remaining:
        model, tokenizer = load_model(args.model, args.revision)

        for problem in remaining:
            start = time.perf_counter()
            result = solve_problem(problem, model, tokenizer, args.budget, args.seed)
            result.update(settings, seconds=round(time.perf_counter() - start, 1))
            log_result(args.output, result)
            results.append(result)
    print_summary(results, args.output)


def main():
    run_cli(__doc__, solve, "single_agent_results.jsonl")


if __name__ == "__main__":
    main()
