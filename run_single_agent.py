"""Run one fresh Qwen3-4B chat per problem from generate_problems.py."""

import argparse
import json
import re
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
def solve(problem, model, tokenizer, budget, seed):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem["prompt"]},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=True,
        return_tensors="pt", return_dict=True,
    ).to(model.device)
    input_tokens = inputs["input_ids"].shape[1]
    if input_tokens >= budget:
        raise ValueError(f"Prompt requires {input_tokens} tokens; budget {budget} leaves no output.")

    torch.manual_seed(seed)
    sequence = model.generate(
        **inputs, max_new_tokens=budget - input_tokens,
        do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
    )
    output_ids = sequence[0, input_tokens:].tolist()
    eos = model.generation_config.eos_token_id
    stopped = output_ids[-1] in (eos if isinstance(eos, list) else [eos])
    text = final_answer(tokenizer, output_ids)
    predicted, format_ok = parse_answer(text)
    if not stopped:
        status = "length_limit"
    elif predicted is None:
        status = "format_error"
    elif predicted == problem["answer"]:
        status = "correct"
    else:
        status = "wrong_answer"

    return {
        **problem, "final_text": text, "prediction": predicted, "status": status,
        "correct": status == "correct", "format_ok": format_ok, "input_tokens": input_tokens,
        "output_tokens": len(output_ids), "total_tokens": input_tokens + len(output_ids),
        "model": MODEL, "revision": REVISION, "budget": budget, "sampling_seed": seed,
        "device": model.device.type, "dtype": str(model.dtype),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problems", type=Path, help="JSONL file from the problem generator")
    parser.add_argument("--output", type=Path, default=Path("single_agent_results.jsonl"))
    parser.add_argument("--budget", type=int, default=4000, help="total tokens per problem")
    parser.add_argument("--seed", type=int, default=42, help="sampling seed for each fresh chat")
    args = parser.parse_args()
    if not 1 <= args.budget <= 32768:
        parser.error("--budget must be between 1 and 32768")
    try:
        problems = read_problems(args.problems)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.output.exists():
        parser.error(f"{args.output} already exists; choose a new --output file")

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, torch_dtype="auto", use_safetensors=True,
    ).to(device).eval()

    counts = {"independent": [0, 0], "dependent": [0, 0]}
    with args.output.open("x", encoding="utf-8") as log:
        for problem in problems:
            result = solve(problem, model, tokenizer, args.budget, args.seed)
            print(json.dumps(result), file=log, flush=True)
            count = counts[problem["kind"]]
            count[0] += result["correct"]
            count[1] += 1
    for kind, (correct, total) in counts.items():
        if total:
            print(f"{kind}: {correct}/{total} correct")
    print(f"Saved {len(problems)} responses to {args.output}")


if __name__ == "__main__":
    main()
