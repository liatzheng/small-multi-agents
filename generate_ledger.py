"""Generate ledger aggregation problems: many stores' sales ledgers and one question, as JSON Lines.

Usage: python3 generate_ledger.py --groups 10 --seed 7001 --stores 8 > ledger.jsonl
Each group shares one set of ledgers across the question types, which differ in how
per-store results must be merged: a total adds per-store totals, an average needs
per-store totals and counts, and customer totals need every customer's per-store total.

With --linked and --depths, some stores get one row worth one tenth of another store's total,
so their workers must wait for that store. Each depth reuses the group's ledgers and links in
chains that many stores deep; depth 0 writes the same amounts as plain numbers.
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


def link_ledgers(ledgers, names, linked, depth, rng):
    """Insert a row into `linked` stores worth one tenth of another store's total, rounded down.

    The linked stores form chains `depth` deep: a store at level L refers to one at level L-1,
    and level 0 stores refer to none. Returns rows as (name, category, amount, source), where
    source is the referred store's index or None, and the links with each store's level.
    """
    order = rng.sample(range(len(ledgers)), len(ledgers))
    chained = order[len(ledgers) - linked:]
    level = {s: 0 for s in order}
    # The first `depth` chained stores fill every level, so the deepest chain is exactly `depth`.
    for k, s in enumerate(chained):
        level[s] = k + 1 if k < depth else rng.randint(1, depth)
    source = {s: rng.choice([t for t in order if level[t] == level[s] - 1]) for s in chained}
    full = [[(n, c, a, None) for n, c, a in ledger] for ledger in ledgers]
    totals = [sum(a for _, _, a in ledger) for ledger in ledgers]
    for s in sorted(chained, key=level.get):
        amount = totals[source[s]] // 10
        full[s].insert(rng.randint(0, len(full[s])),
                       (rng.choice(names), rng.choice(CATEGORIES), amount, source[s]))
        totals[s] += amount
    links = [{"store": s + 1, "source": source[s] + 1, "level": level[s]} for s in sorted(chained)]
    return full, links


def row_text(name, category, amount, source):
    if source is None:
        return f"{name}, {category}, {amount}"
    return f"{name}, {category}, one tenth of Store {source + 1}'s total, rounded down"


def generate_ledger_problems(groups, seed, stores, rows, customers, max_amount, kinds,
                             linked=0, depths=(None,)):
    """Yield one problem per group, depth, and question type; depth None means no linked rows."""
    if groups < 1 or stores < 2 or rows < 1 or max_amount < 2:
        raise ValueError("--groups and --rows must be positive, --stores at least 2, "
                         "--max-amount at least 2")
    if not 2 <= customers <= len(CUSTOMERS):
        raise ValueError(f"--customers must be 2 to {len(CUSTOMERS)}")
    if depths != (None,):
        if not 1 <= linked < stores or not all(0 <= d <= linked for d in depths):
            raise ValueError("--linked must be 1 to --stores minus 1, and --depths 0 to --linked")
        if "average" in kinds:
            raise ValueError("linked rows support only the total and customer_totals questions")
    rng = random.Random(seed)
    group_id = 0
    while group_id < groups:
        names = rng.sample(CUSTOMERS, customers)
        ledgers = [[(rng.choice(names), rng.choice(CATEGORIES), rng.randint(1, max_amount))
                    for _ in range(rows)] for _ in range(stores)]
        category = rng.choice(CATEGORIES)
        # An absent category has no average; draw again.
        if not any(cat == category for ledger in ledgers for _, cat, _ in ledger):
            continue
        group_id += 1
        for depth in depths:
            if depth is None:
                full, links = [[(n, c, a, None) for n, c, a in ledger] for ledger in ledgers], []
            else:
                # A string seed keeps the main draws, and so the ledgers, the same for any depths.
                # Depth 0 takes depth 1's links, so the two differ only in how amounts are written.
                link_rng = random.Random(f"{seed}:{group_id}:{max(depth, 1)}")
                full, links = link_ledgers(ledgers, names, linked, max(depth, 1), link_rng)
                if depth == 0:
                    full = [[(n, c, a, None) for n, c, a, _ in ledger] for ledger in full]
                    links = []
            spent = {}
            for ledger in full:
                for name, _, amount, _ in ledger:
                    spent[name] = spent.get(name, 0) + amount
            picked = [a for ledger in full for _, cat, a, _ in ledger if cat == category]
            present = sorted(spent)
            texts = [f"Store {i}:\n" + "\n".join(row_text(*row) for row in ledger)
                     for i, ledger in enumerate(full, start=1)]
            answers = {
                "total": (["total"], [sum(spent.values())]),
                "average": (["average"], [sum(picked) // len(picked)]),
                "customer_totals": (present, [spent[n] for n in present]),
            }
            intro = "Each row is customer, category, amount."
            if links:
                intro += " A store's total is the sum of all its amounts."
            for kind in kinds:
                text, form = question(kind, category, present)
                prompt = (f"Below are the sales ledgers of {stores} stores. {intro}\n\n"
                          + "\n\n".join(texts) + f"\n\n{text}")
                targets, answer = answers[kind]
                record = {
                    "group_id": group_id, "question_type": kind, "width": stores, "rows": rows,
                    "customers": customers, "max_amount": max_amount, "category": category,
                    "question": text, "answer_format": form, "prompt": prompt, "stores": texts,
                    "targets": targets, "answer": answer,
                    "oracle_instruction": oracle_instruction(kind, category),
                }
                if depth is not None:
                    record.update({"linked": linked, "depth": depth, "links": links,
                                   "store_totals": [sum(a for _, _, a, _ in l) for l in full]})
                yield record


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--groups", type=int, default=1, help="problem groups to generate")
    parser.add_argument("--seed", type=int, help="seed for reproducible sampling")
    parser.add_argument("--stores", type=int, required=True, help="number of ledgers")
    parser.add_argument("--rows", type=int, default=12, help="rows per ledger")
    parser.add_argument("--customers", type=int, default=8, help="customers shared across stores")
    parser.add_argument("--max-amount", type=int, default=9, help="largest sale amount (smallest is 1)")
    parser.add_argument("--types", nargs="+", choices=QUESTION_TYPES, default=list(QUESTION_TYPES))
    parser.add_argument("--linked", type=int, default=0,
                        help="stores with a row worth one tenth of another store's total")
    parser.add_argument("--depths", nargs="+", type=int,
                        help="deepest chain of linked stores, one problem set per depth")
    args = parser.parse_args()
    try:
        for problem in generate_ledger_problems(
                args.groups, args.seed, args.stores, args.rows, args.customers,
                args.max_amount, args.types, args.linked, tuple(args.depths or [None])):
            print(json.dumps(problem))
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
