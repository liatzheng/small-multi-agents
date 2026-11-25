"""Generate wide chain problems: many target chains among distractors, as JSON Lines.

Usage: python3 generate_wide.py --groups 20 --seed 4001 --width 8 --depth 3 > wide.jsonl
Each group shares labels, coefficients, givens, and equation order across the
dependent fractions. Only the chain starts that link targets together change.
"""

import argparse
import json
import random


def is_prime(n):
    return n >= 2 and all(n % k for k in range(2, int(n ** 0.5) + 1))


def generate_wide_problems(groups, seed, width, depth, distractors, fractions,
                           modulus=97, max_factor=5, max_offset=19):
    """Yield one problem per group and dependent fraction."""
    if groups < 1 or width < 2 or depth < 1 or distractors < 0:
        raise ValueError("--groups and --depth must be positive, --width at least 2, "
                         "--distractors at least 0")
    if not all(0 <= f <= 1 for f in fractions):
        raise ValueError("--fractions must be between 0 and 1")
    if not is_prime(modulus) or modulus < 3:
        raise ValueError("--modulus must be a prime of at least 3")
    if not 2 <= max_factor < modulus or not 1 <= max_offset < modulus:
        raise ValueError("--max-factor must be 2 to modulus-1 and --max-offset 1 to modulus-1")
    rng = random.Random(seed)
    num_chains = width + distractors
    digits = max(2, len(str(num_chains * (depth + 1) - 1)))
    for group_id in range(1, groups + 1):
        # Random labels and shuffled equations require tracing inputs to targets.
        labels = [f"v{i:0{digits}d}" for i in range(num_chains * (depth + 1))]
        rng.shuffle(labels)
        chains = [labels[i * (depth + 1):(i + 1) * (depth + 1)] for i in range(num_chains)]
        seeds = {chain[0]: rng.randrange(modulus) for chain in chains}
        rules = {}
        for chain in chains:
            for source, dest in zip(chain, chain[1:]):
                rules[dest] = (source, rng.randint(2, max_factor), rng.randint(1, max_offset))
        order = list(rules)
        rng.shuffle(order)
        targets = chains[:width]
        # Dependencies link consecutive targets in a hidden order, separate from the requested
        # order, so the request does not reveal which chain feeds which.
        dep_order = list(range(width))
        rng.shuffle(dep_order)
        request = list(range(width))
        rng.shuffle(request)
        # Each fraction activates a prefix of one link order, so higher fractions add links.
        link_order = list(range(width - 1))
        rng.shuffle(link_order)
        givens = ", ".join(f"{label} = {value}" for label, value in sorted(seeds.items()))

        for fraction in fractions:
            links = round(fraction * (width - 1))
            active = set(link_order[:links])
            # after[t] is the target whose final value starts chain t.
            after = {dep_order[0]: None}
            for i in range(width - 1):
                after[dep_order[i + 1]] = dep_order[i] if i in active else None
            finals, plan = {}, {}
            for t in dep_order:
                chain = targets[t]
                prev = after[t]
                value = seeds[chain[0]] if prev is None else finals[prev]
                steps = []
                for dest in chain[1:]:
                    source, factor, offset = rules[dest]
                    if dest == chain[1] and prev is not None:
                        source = targets[prev][-1]
                    steps.append(f"{dest} = ({factor} * {source} + {offset}) mod {modulus}")
                    value = (factor * value + offset) % modulus
                finals[t] = value
                plan[t] = {
                    "target": chain[-1],
                    "after": None if prev is None else targets[prev][-1],
                    "start": [chain[0], seeds[chain[0]]] if prev is None else None,
                    "equations": steps,
                }
            # The first update of a linked chain reads the previous target instead of its start.
            sources = {targets[t][1]: targets[after[t]][-1] for t in after if after[t] is not None}
            equations = []
            for dest in order:
                source, factor, offset = rules[dest]
                source = sources.get(dest, source)
                equations.append(f"{dest} = ({factor} * {source} + {offset}) mod {modulus}")
            run = critical = 1
            for i in range(width - 1):
                run = run + 1 if i in active else 1
                critical = max(critical, run)
            requested = [targets[t][-1] for t in request]
            prompt = (
                f"Given {givens}. Each equation defines one variable. The equations "
                f"are listed in shuffled order. Use remainders from 0 to {modulus - 1}.\n"
                + "\n".join(equations)
                + f"\nReturn ({', '.join(requested)}) in that order."
            )
            yield {
                "group_id": group_id, "dependent_fraction": fraction, "width": width,
                "depth": depth, "distractors": distractors, "modulus": modulus,
                "max_factor": max_factor, "max_offset": max_offset,
                "dependent_links": links, "critical_path": critical,
                "prompt": prompt, "targets": requested,
                "answer": [finals[t] for t in request], "plan": [plan[t] for t in request],
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--groups", type=int, default=1, help="problem groups to generate")
    parser.add_argument("--seed", type=int, help="seed for reproducible sampling")
    parser.add_argument("--width", type=int, required=True, help="number of requested target chains")
    parser.add_argument("--depth", type=int, default=3, help="updates per chain")
    parser.add_argument("--distractors", type=int, help="unrequested chains (default: width)")
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.0, 0.5, 1.0],
                        help="shares of the width-1 possible links between targets")
    parser.add_argument("--modulus", type=int, default=97, help="prime modulus for chain updates")
    parser.add_argument("--max-factor", type=int, default=5, help="largest update factor (smallest is 2)")
    parser.add_argument("--max-offset", type=int, default=19, help="largest update offset (smallest is 1)")
    args = parser.parse_args()
    distractors = args.width if args.distractors is None else args.distractors
    try:
        for problem in generate_wide_problems(
                args.groups, args.seed, args.width, args.depth, distractors, args.fractions,
                args.modulus, args.max_factor, args.max_offset):
            print(json.dumps(problem))
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
