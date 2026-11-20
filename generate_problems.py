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


def generate_chain_problems(num_pairs, seed, length, distractor_length=None,
                            modulus=97, max_factor=5, max_offset=19):
    """Generate matched pairs with two target chains and one distractor chain."""
    if distractor_length is None:
        distractor_length = length
    if num_pairs < 1 or length < 1 or distractor_length < 1:
        raise ValueError("--pairs, --chain-length, and --distractor-length must be positive")
    # A prime modulus makes every update invertible, which keeps the two variants' answers distinct.
    if modulus < 3 or any(modulus % k == 0 for k in range(2, int(modulus ** 0.5) + 1)):
        raise ValueError("--modulus must be a prime of at least 3")
    if not 2 <= max_factor < modulus or not 1 <= max_offset < modulus:
        raise ValueError("--max-factor must be 2 to modulus-1 and --max-offset 1 to modulus-1")
    rng = random.Random(seed)
    for pair_id in range(1, num_pairs + 1):
        # Random labels and shuffled equations require tracing inputs to targets.
        labels = [f"v{i:02d}" for i in range(2 * (length + 1) + distractor_length + 1)]
        rng.shuffle(labels)
        split = length + 1
        chains = [labels[:split], labels[split:2 * split], labels[2 * split:]]
        seeds = {chain[0]: rng.randrange(modulus) for chain in chains}
        rules = {}
        for chain in chains:
            for source, dest in zip(chain, chain[1:]):
                rules[dest] = (source, rng.randint(2, max_factor), rng.randint(1, max_offset))
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
                   "distractor_length": distractor_length, "modulus": modulus,
                   "max_factor": max_factor, "max_offset": max_offset,
                   "prompt": prompt, "answer": [a, b]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs", type=int, default=1, help="number of pairs (1–756 for the original examples)")
    parser.add_argument("--seed", type=int, help="seed for reproducible sampling")
    parser.add_argument("--chain-length", type=int, metavar="D",
                        help="generate shuffled modular chains with D updates per chain")
    parser.add_argument("--distractor-length", type=int, metavar="L",
                        help="updates in the distractor chain (default: D)")
    parser.add_argument("--modulus", type=int, default=97, help="prime modulus for chain updates")
    parser.add_argument("--max-factor", type=int, default=5, help="largest update factor (smallest is 2)")
    parser.add_argument("--max-offset", type=int, default=19, help="largest update offset (smallest is 1)")
    args = parser.parse_args()
    chain_options = (args.distractor_length, args.modulus != 97, args.max_factor != 5, args.max_offset != 19)
    if args.chain_length is None and any(chain_options):
        parser.error("the chain options require --chain-length")
    try:
        if args.chain_length is None:
            problems = generate_problems(args.pairs, args.seed)
        else:
            problems = generate_chain_problems(
                args.pairs, args.seed, args.chain_length, args.distractor_length,
                args.modulus, args.max_factor, args.max_offset)
        for problem in problems:
            print(json.dumps(problem))
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
