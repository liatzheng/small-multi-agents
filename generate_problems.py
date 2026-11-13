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


def generate_chain_problems(num_pairs, seed, length):
    """Generate matched pairs with two target chains and one distractor chain."""
    if num_pairs < 1 or length < 1:
        raise ValueError("--pairs and --chain-length must be positive")
    rng = random.Random(seed)
    modulus = 97
    for pair_id in range(1, num_pairs + 1):
        # Random labels and shuffled equations require tracing inputs to targets.
        labels = [f"v{i:02d}" for i in range(3 * (length + 1))]
        rng.shuffle(labels)
        chains = [labels[i:i + length + 1] for i in range(0, len(labels), length + 1)]
        seeds = {chain[0]: rng.randrange(modulus) for chain in chains}
        rules = {}
        for chain in chains:
            for source, dest in zip(chain, chain[1:]):
                rules[dest] = (source, rng.randint(2, 5), rng.randint(1, 19))
        order = list(rules)
        rng.shuffle(order)
        first, second, _ = chains

        def evaluate(chain, start_value):
            value = start_value
            for dest in chain[1:]:
                _, factor, offset = rules[dest]
                value = (factor * value + offset) % modulus
            return value

        a = evaluate(first, seeds[first[0]])
        # Distinct second-chain inputs stay distinct under these invertible updates.
        if seeds[second[0]] == a:
            seeds[second[0]] = (a + rng.randint(1, modulus - 1)) % modulus
        givens = ", ".join(f"{label} = {value}" for label, value in sorted(seeds.items()))
        for kind in ("independent", "dependent"):
            bridge = first[-1] if kind == "dependent" else second[0]
            equations = []
            for dest in order:
                source, factor, offset = rules[dest]
                if dest == second[1]:
                    source = bridge
                equations.append(f"{dest} = ({factor} * {source} + {offset}) mod {modulus}")
            b = evaluate(second, a if kind == "dependent" else seeds[second[0]])
            prompt = (
                f"Given {givens}. Each equation defines one variable. The equations "
                f"are listed in shuffled order. Use remainders from 0 to {modulus - 1}.\n"
                + "\n".join(equations)
                + f"\nReturn ({first[-1]}, {second[-1]}) in that order."
            )
            yield {"pair_id": pair_id, "kind": kind, "chain_length": length,
                   "prompt": prompt, "answer": [a, b]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs", type=int, default=1, help="number of pairs (1–756 for the original examples)")
    parser.add_argument("--seed", type=int, help="seed for reproducible sampling")
    parser.add_argument("--chain-length", type=int, metavar="D",
                        help="generate shuffled modular chains with D updates per chain")
    args = parser.parse_args()
    try:
        if args.chain_length is None:
            problems = generate_problems(args.pairs, args.seed)
        else:
            problems = generate_chain_problems(args.pairs, args.seed, args.chain_length)
        for problem in problems:
            print(json.dumps(problem))
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
