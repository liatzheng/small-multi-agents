"""Generate ledger aggregation problems: many stores' sales ledgers and one question, as JSON Lines.

Usage: python3 generate_ledger.py --groups 10 --seed 7001 --stores 8 > ledger.jsonl
Each group shares one set of ledgers across the question types, which differ in how
per-store results must be merged: a total adds per-store totals, an average needs
per-store totals and counts, and customer totals need every customer's per-store total.
"""

import argparse
import json
import random

CUSTOMERS = ["Ana", "Ben", "Cora", "Dev", "Eli", "Fay", "Gus", "Hana", "Ivo", "Jun", "Kai", "Lea"]
CATEGORIES = ["books", "food", "tools", "toys"]
QUESTION_TYPES = ("total", "average", "customer_totals")


def question(kind, category, names):
    if kind == "total":
        return "What is the total of all amounts across all stores?", "Answer: <number>"
    if kind == "average":
        return (f"What is the average amount of a {category} sale across all stores, "
                "rounded down to a whole number?", "Answer: <number>")
    return (f"What is each customer's total across all stores? Give the totals for "
            f"{', '.join(names)}.", "Answer: " + ", ".join(f"{n} = <total>" for n in names))


def oracle_instruction(kind, category):
    """The worker instruction a correct plan gives, and the report line(s) it asks for."""
    if kind == "total":
        return ("Add up the amounts of all rows, keeping a running total.\n"
                "Report one line: total = <number>")
    if kind == "average":
        return (f"Go through the rows and keep only the {category} rows. Add up their amounts "
                "with a running total, and count them.\n"
                "Report one line: total = <number>, count = <number>")
    return ("Keep a separate running total for each customer: go through the rows in order and "
            "add each amount to that row's customer.\n"
            "Report one line per customer who appears: <customer> = <total>")


def generate_ledger_problems(groups, seed, stores, rows, customers, max_amount, kinds):
    """Yield one problem per group and question type."""
    if groups < 1 or stores < 2 or rows < 1 or max_amount < 2:
        raise ValueError("--groups and --rows must be positive, --stores at least 2, "
                         "--max-amount at least 2")
    if not 2 <= customers <= len(CUSTOMERS):
        raise ValueError(f"--customers must be 2 to {len(CUSTOMERS)}")
    rng = random.Random(seed)
    group_id = 0
    while group_id < groups:
        names = rng.sample(CUSTOMERS, customers)
        ledgers = [[(rng.choice(names), rng.choice(CATEGORIES), rng.randint(1, max_amount))
                    for _ in range(rows)] for _ in range(stores)]
        spent = {}
        for ledger in ledgers:
            for name, _, amount in ledger:
                spent[name] = spent.get(name, 0) + amount
        category = rng.choice(CATEGORIES)
        picked = [amount for ledger in ledgers for _, cat, amount in ledger if cat == category]
        # An absent category has no average; draw again.
        if not picked:
            continue
        group_id += 1
        present = sorted(spent)
        texts = [f"Store {i}:\n" + "\n".join(f"{n}, {c}, {a}" for n, c, a in ledger)
                 for i, ledger in enumerate(ledgers, start=1)]
        answers = {
            "total": (["total"], [sum(spent.values())]),
            "average": (["average"], [sum(picked) // len(picked)]),
            "customer_totals": (present, [spent[n] for n in present]),
        }
        for kind in kinds:
            text, form = question(kind, category, present)
            prompt = (f"Below are the sales ledgers of {stores} stores. Each row is "
                      "customer, category, amount.\n\n" + "\n\n".join(texts)
                      + f"\n\n{text}")
            targets, answer = answers[kind]
            yield {
                "group_id": group_id, "question_type": kind, "width": stores, "rows": rows,
                "customers": customers, "max_amount": max_amount, "category": category,
                "question": text, "answer_format": form, "prompt": prompt, "stores": texts,
                "targets": targets, "answer": answer,
                "oracle_instruction": oracle_instruction(kind, category),
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--groups", type=int, default=1, help="problem groups to generate")
    parser.add_argument("--seed", type=int, help="seed for reproducible sampling")
    parser.add_argument("--stores", type=int, required=True, help="number of ledgers")
    parser.add_argument("--rows", type=int, default=12, help="rows per ledger")
    parser.add_argument("--customers", type=int, default=8, help="customers shared across stores")
    parser.add_argument("--max-amount", type=int, default=9, help="largest sale amount (smallest is 1)")
    parser.add_argument("--types", nargs="+", choices=QUESTION_TYPES, default=list(QUESTION_TYPES))
    args = parser.parse_args()
    try:
        for problem in generate_ledger_problems(
                args.groups, args.seed, args.stores, args.rows, args.customers,
                args.max_amount, args.types):
            print(json.dumps(problem))
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
