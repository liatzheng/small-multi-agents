"""Run three independent Qwen3-4B attempts per problem and score their majority vote."""

from collections import Counter

from run_single_agent import direct_messages, run_cli, run_hf, score_reply


def majority(votes):
    """Return the ordered pair with at least two votes, or None for no majority."""
    counts = Counter(tuple(vote) for vote in votes if vote is not None)
    for pair, count in counts.items():
        if count >= 2:
            return list(pair)
    return None


def solve_ensemble_steps(problem, budget, seed):
    base, remainder = divmod(budget, 3)
    caps = [base] * (3 - remainder) + [base + 1] * remainder
    # Each attempt is a new chat from only the original problem, so all three go out together.
    replies = yield [
        {"messages": direct_messages(problem), "cap": cap, "seed": seed + i}
        for i, cap in enumerate(caps)
    ]
    attempts = []
    for reply in replies:
        result = score_reply(problem, reply)
        attempt = {key: result[key] for key in (
            "sampling_seed", "budget", "input_tokens", "output_tokens", "total_tokens",
            "final_text", "prediction", "format_ok", "status",
        )}
        attempt["vote"] = None if result["status"] == "length_limit" else result["prediction"]
        attempts.append(attempt)

    prediction = majority(attempt["vote"] for attempt in attempts)
    correct = prediction == problem["answer"]
    status = "correct" if correct else "wrong_answer"
    if prediction is None:
        status = "no_majority"

    input_tokens = sum(a["input_tokens"] for a in attempts)
    output_tokens = sum(a["output_tokens"] for a in attempts)
    return {
        **problem, "prediction": prediction, "correct": correct, "status": status,
        "attempts": attempts, "budget": budget, "input_tokens": input_tokens,
        "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens,
    }


def solve_ensemble(problem, model, tokenizer, budget, seed):
    return run_hf(solve_ensemble_steps(problem, budget, seed), model, tokenizer)


if __name__ == "__main__":
    run_cli(__doc__, solve_ensemble, "ensemble_results.jsonl")
