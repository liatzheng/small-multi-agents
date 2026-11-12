"""Generate the matched arithmetic problems in examples.md as JSON Lines.

Usage: python3 generate_problems.py --pairs 10 --seed 42
Each pair produces two records with a shared pair_id. The prompt is the model
input, and answer is the ordered pair [a, b] for scoring.
"""

import argparse
import json
import random


def generate_problems(num_pairs=1, seed=None):
    """Yield one independent and one dependent problem for each set of inputs."""
    digits = range(1, 10)
    # Keeping p + q < r makes both answers positive and single-digit.
    inputs = [
        (p, q, r, s)
        for p in digits
        for q in digits
        for r in range(p + q + 1, 10)
        for s in digits
    ]
    if not 1 <= num_pairs <= len(inputs):
        raise ValueError(f"--pairs must be between 1 and {len(inputs)}")

    rng = random.Random(seed)
    for pair_id, (p, q, r, s) in enumerate(rng.sample(inputs, num_pairs), start=1):
        a = p + q
        for kind, operand, value in (("independent", "p", p), ("dependent", "a", a)):
            prompt = (
                f"Given p = {p}, q = {q}, r = {r}, and s = {s}:\n"
                "1. Compute a = p + q.\n"
                f"2. Compute b = r - {operand}.\n"
                "Return (a, b)."
            )
            yield {
                "pair_id": pair_id,
                "kind": kind,
                "prompt": prompt,
                "answer": [a, r - value],
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs", type=int, default=1, help="number of distinct pairs (1–756)")
    parser.add_argument("--seed", type=int, help="seed for reproducible sampling")
    args = parser.parse_args()
    try:
        for problem in generate_problems(args.pairs, args.seed):
            print(json.dumps(problem))
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
