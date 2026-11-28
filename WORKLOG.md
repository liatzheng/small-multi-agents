# Worklog

## 2026-11-24: change of direction to wide tasks

### Why the old direction stalled

The Qwen3-4B runs on the two-chain task (dev_systems, dev_planner, thinking off, 16k) gave single 0.48, ensemble 0.40, hierarchy 0.25, thinking planner 0.28, and trace planner 0.08. Reading the logs showed four problems with the setup rather than with coordination itself:

- The budget never bound. The single agent used about 3.1k of 16k tokens, so the hypothesis (coordination helps when the work exceeds one agent's budget) predicted no gain, and none appeared.
- Failures were mod-reduction slips. Splitting the work cannot remove arithmetic errors. Repeating it can: the ensemble had a correct attempt on 80-90% of d2 problems, but exact-pair voting lost it.
- The worker prompt in `run_hierarchy.py` asked for "the requested variable and its value in one short line". With thinking off, workers answered without working: worker 2 averaged 75 output tokens (17 with the trace planner). The hierarchy scores mostly measured this.
- The dependent version changes one variable reference in a two-chain problem, which only makes the split pure overhead.

### New design

Make tasks bigger, not computationally harder. Each problem asks for w target chains of depth d among w distractor chains, with easy per-step arithmetic (mod 97, factors up to 5, offsets up to 19). A dependent fraction f links a share of the target chains so that one starts from another's result, which forces workers to wait. Every group keeps the same labels, coefficients and equation order across f, so total work is fixed and only parallelism changes.

Predictions:

- The single agent degrades as w grows.
- The ensemble (B/3 per attempt) starves first, since each attempt must still solve all w chains.
- A hierarchy with one worker per chain needs only one chain's worth of tokens per worker and runs workers in parallel, so it wins on accuracy and time.
- Dependence removes the parallel speedup. This reverses the proposal's direction (the proposal predicted gains grow with interdependence), so Lily should hear about it.

Model: Gemma-3 4B/12B/27B through OpenRouter, using the $200 credit. Qwen3 1.7B/4B/8B are not on OpenRouter (checked 2026-11-24). All calls are pinned to DeepInfra with no fallback: 4B and 12B in bf16, 27B in fp8 (no bf16 endpoint). Sampling follows Gemma's generation config: temperature 1.0, top_k 64, top_p 0.95. Gemma has no thinking mode.

### Code

- `generate_wide.py` generates the problems and stores each target's chain (the oracle plan) in the record. Checked: every answer and oracle plan reproduces from the prompt.
- `run_wide.py` is a standalone OpenRouter runner. The old runners and Lia's notebook are unchanged.
  - Systems:
    - `single_agent`: one call with budget B.
    - `ensemble`: 3 attempts at B/3 each, with a majority vote per target.
    - `hierarchy_oracle`: one worker per chain at B/w, with assignments built by code from the oracle plan. Planning costs no tokens, so this is an upper bound.
    - `hierarchy`: a model planner gets B/4 and writes a JSON list of self-contained assignments. Workers split the rest evenly.
  - Workers see only their assignment, never the full problem. Otherwise input cost grows with w squared.
  - Dependent workers wait for the worker they depend on and receive its reported value. Code merges the values; no model does the merge.
  - Budgets are charged with the provider's token counts. A local Gemma tokenizer sets `max_tokens`, and its count matched DeepInfra's exactly on every call checked.
  - Logged per problem: all-correct, per-target accuracy, total tokens, critical-path output tokens and seconds (the slowest call per parallel round), and rounds.
  - `--planner-model` and `--planner-provider` run the planner on another model. If a planner names a non-requested variable in "after", the link is dropped and counted in `plan_dropped_after`. Before this change such plans failed as `invalid_plan`, which was rejecting plans that were otherwise correct.
- Runs go locally under `caffeinate`. The first calibration run stalled for about 1.5 hours when the Mac slept.

### Calibration: Gemma-3 4B, `results/wide_calibrate/`

Settings: 10 groups at each of w = 2, 4, 8, 16, depth 3, f in {0, 0.5, 1}, so 120 problems per system and budget. Budgets were 8k and 16k. Spend was $0.40.

Share of target variables correct at 8k:

| system | w2 | w4 | w8 | w16 |
|---|---|---|---|---|
| single agent | 0.77 | 0.54 | 0.20 | 0.01 |
| ensemble | 0.87 | 0.26 | 0.00 | 0.00 |
| hierarchy, oracle plan | 1.00 | 0.82 | 0.97 | 0.12 |
| hierarchy, 4B planner | 0.10 | 0.05 | 0.00 | 0.00 |

Share of target variables correct at 16k:

| system | w2 | w4 | w8 | w16 |
|---|---|---|---|---|
| single agent | 0.72 | 0.61 | 0.21 | 0.01 |
| ensemble | 0.88 | 0.50 | 0.12 | 0.00 |
| hierarchy, oracle plan | 0.98 | 0.83 | 0.95 | 0.77 |
| hierarchy, 4B planner | 0.07 | 0.06 | 0.00 | 0.00 |

Tokens per problem at 16k (single / ensemble / oracle):

| | w2 | w4 | w8 | w16 |
|---|---|---|---|---|
| total tokens | 1573 / 4550 / 1024 | 2727 / 8508 / 2009 | 5179 / 13166 / 4167 | 11256 / 15651 / 8724 |
| critical-path output tokens, single vs oracle | 1221 vs 520 | 2100 vs 955 | 4004 vs 1728 | 8745 vs 3051 |

Findings:

- **Single agent:** degrades with width and barely changes between 8k and 16k, so the failures come from task size, not budget. At w16 with 8k, 20 of 30 runs hit the cap.
- **Ensemble:** beats the single agent only at w2. From w4 its attempts run out of tokens: 48 of 90 attempts at 8k, and at w8 89 of 90.
- **Oracle hierarchy:** close to perfect through w8 and uses fewer total tokens than the single agent.
  - At w16 with 8k it starves at 500 tokens per worker (195 of 259 worker calls hit the cap). At 16k it recovers to 0.77.
  - With all 16 chains linked (f = 1) it drops to 0.54, because one slip carries through the rest of the chain.
- **Dependence costs time as predicted.** Oracle at w8, 8k: 6.8s, 26s and 56s for f = 0, 0.5 and 1, with 1, 3.7 and 8 rounds, which equals the critical path.
- **Dependence also lowers accuracy for every system.** The single agent at w4, 8k: 0.70 at f = 0 vs 0.42 at f = 1.
- **The 4B planner cannot decompose the task.**
  - It writes invalid plans, or assignments that hold only the target's last equation, so workers invent missing inputs.
  - At w8 with 8k it runs out of its B/4 share.
  - At w16 with 8k the prompt alone is longer than B/4.

Caveats:

- Each width uses different random problems and there are only 10 groups, so differences around 0.1 are noise. The oracle's w4 score (0.82) being below its w8 score (0.97) comes from slips clustered in groups 3, 6 and 9.
- Wall-clock seconds moved with provider load between runs (single agent at w4: 25s at 8k vs 93s at 16k), so critical-path output tokens is the time metric to report.
- One sample per problem.

### Planner size, first run (discarded): `results/wide_planner/`

Setup: 4B, 12B and 27B planners with 4B workers, on the same 120 problems at 8k and 16k. This tests the proposal's claim that the orchestrator matters more than the workers, with the planner's tokens counted in the budget. Spend was $0.19.

Share of target variables correct at 16k:

| planner | w2 | w4 | w8 | w16 |
|---|---|---|---|---|
| 4B | 0.05 | 0.07 | 0.00 | 0.00 |
| 12B | 0.17 | 0.17 | 0.11 | 0.01 |
| 27B | 0.20 | 0.17 | 0.15 | 0.01 |

These numbers understate the planners because of a harness asymmetry. Oracle assignments end with "Find <target>.", but model-planned assignments did not name the target, so workers often computed the target correctly and then reported an intermediate variable on their last line. Fixed in `hierarchy_steps`: code now appends "Find <target>." to every model-planned assignment.

Other observations from this run:

- At w16, every planner hits its B/4 cap or the prompt alone exceeds it (30 of 30 at 8k).
- Workers with complete 27B-planned assignments (every update on the chain plus its start present) were right 61/123 times on independent chains, before the fix above.
- Incomplete assignments were essentially never right.

### Planner size with the fix: `results/wide_planner_v2/`

Same setup as the first run, with "Find <target>." appended to every assignment. Total OpenRouter spend after this run: $0.82.

Share of target variables correct, 4B workers (single agent and oracle repeated from calibration):

| system | budget | w2 | w4 | w8 | w16 |
|---|---|---|---|---|---|
| single agent 4B | 16k | 0.72 | 0.61 | 0.21 | 0.01 |
| hierarchy, oracle plan | 16k | 0.98 | 0.83 | 0.95 | 0.77 |
| hierarchy, 4B planner | 16k | 0.12 | 0.06 | 0.00 | 0.00 |
| hierarchy, 12B planner | 16k | 0.13 | 0.16 | 0.29 | 0.01 |
| hierarchy, 27B planner | 16k | 0.50 | 0.19 | 0.21 | 0.03 |
| hierarchy, 27B planner | 8k | 0.40 | 0.20 | 0.01 | 0.00 |

Fraction of planner dependency links ("after") that match the true structure:

| planner | w2 | w4 | w8 |
|---|---|---|---|
| 4B | 0.59 | 0.55 | 0.27 |
| 12B | 0.46 | 0.43 | 0.26 |
| 27B | 0.78 | 0.52 | 0.46 |

Worker accuracy by assignment quality (16k, w <= 8, plans that ran). An assignment is complete if every update on the chain and its start appear in it.

| planner | complete, independent | complete, dependent | incomplete |
|---|---|---|---|
| 4B | 14/14 | 0/5 | 1/229 |
| 12B | 84/112 | 11/38 | 1/240 |
| 27B | 87/110 | 13/70 | 3/233 |

Findings:

- **Planning is the bottleneck, not the workers.** Given a complete assignment for an independent chain, 4B workers are right about 75-80% of the time. Given an incomplete one, they are almost never right. Even the 27B planner writes a complete assignment for fewer than half of independent chains.
- **Dependencies are where planners fail most.**
  - Planners match only about half the true links. They often name intermediate variables, which are dropped, or invent links.
  - Dependent chains rarely come out right even when the assignment is complete (13/70 for 27B), because a wrong or missing link means the worker never receives its starting value.
- **A bigger planner helps at small widths.** At w2 the 27B planner scores 0.50 at 16k against 0.12 for the 4B planner.
- **The gain over the single agent appears only for wide independent tasks.** At w8 with f = 0 and 16k: 12B planner 0.55 and 27B planner 0.46, against the single agent's 0.40. With dependence the hierarchy loses. This rests on 10 groups, so it is weak evidence.
- **w16 does not work with any planner.** At 8k the prompt alone exceeds the B/4 planner share. At 16k about half the plans still hit the cap, because rewriting 16 chains of equations is long.

### Single-agent baselines with 12B and 27B: `results/wide_calibrate/gemma-3-12b-it|gemma-3-27b-it/`

Same 120 problems at 8k and 16k. 27B runs in fp8, the only DeepInfra option. Total OpenRouter spend after this run: $1.09.

Share of target variables correct at 16k (4B workers in all hierarchies):

| system | w2 | w4 | w8 | w16 |
|---|---|---|---|---|
| single agent 4B | 0.72 | 0.61 | 0.21 | 0.01 |
| single agent 12B | 0.90 | 0.63 | 0.37 | 0.09 |
| single agent 27B | 0.75 | 0.61 | 0.31 | 0.21 |
| hierarchy, 27B planner | 0.50 | 0.19 | 0.21 | 0.03 |
| hierarchy, oracle plan | 0.98 | 0.83 | 0.95 | 0.77 |

At 8k the bigger single agents are similar: 12B 0.85/0.64/0.42/0.04 and 27B 0.70/0.51/0.25/0.11.

Error analysis: each "vX = ... = N" step the single agent writes, judged against its own previous values. This is a heuristic parse of w4-w16 at 16k.

| model | correct given its input | arithmetic slip | source not yet computed |
|---|---|---|---|
| 4B | 57% | 15% | 28% |
| 12B | 71% | 15% | 13% |
| 27B | 73% | 18% | 10% |

Slip rate by position in the answer (steps 0-9 vs 40+):

| model | steps 0-9 | steps 40+ |
|---|---|---|
| 4B | 0.13 | 0.38 |
| 12B | 0.15 | 0.27 |
| 27B | 0.18 | 0.51 |

Findings:

- **Scale barely helps on wide tasks.** 27B is no better than 12B and only somewhat better than 4B, except at w16.
- **Bigger models trace better but still slip.** "Source not yet computed" falls from 28% to 10% with size, but the mod-reduction slip rate stays around 15-18% for every size and rises the later a step comes in a long answer. The 27B also computes the distractor chains, doubling its work.
- **Decomposition does what scale does not.** 4B workers on oracle assignments make far fewer slips. In a short context they write each reduction out, such as "158 = 1 * 97 + 61". With a perfect split, 4B workers beat the 27B single agent by a wide margin (0.95 vs 0.31 at w8, 0.77 vs 0.21 at w16) while using fewer total tokens.
- **A 27B planner with 4B workers does not reach the 27B single agent.** The gain from splitting depends entirely on plan quality, and no Gemma size plans well enough here.
- **Caveat:** 27B in fp8 may understate it. 12B performing about the same as 27B is worth checking against another provider if this comparison goes in the paper.

## 2026-11-24: second task, ledger aggregation

### Why

On the chain task the split is hidden: assigning equations to workers requires tracing the chains, which is most of the task. Planners therefore fail before workers start. Natalia asked for a task whose split is natural without knowing the answer in advance, so the planner and reconstructor have a fair job.

### Design

- `generate_ledger.py`: k store ledgers, each with n rows of `customer, category, amount`. Amounts are 1-9 and 8 customers are shared across stores. Each group of ledgers gets three questions that differ in how per-store results merge:
  - `total`: add the per-store totals (distributive).
  - `average` of one category, rounded down: needs a total and a count per store (algebraic). Averaging the per-store averages is wrong.
  - `customer_totals`: every customer's total across all stores, 8 targets (group-by across chunks). This scores per target like the chain task, so one slip costs one customer rather than the whole problem.
- `run_ledger.py` reuses the OpenRouter client, budget accounting and timing from `run_wide.py`. It has five systems:
  - `single_agent`: the full budget.
  - `ensemble`: 3 attempts at B/3 each, with a per-target majority vote.
  - `hierarchy_oracle`: code gives each store to its own worker with a correct, fixed instruction, and code merges the reports.
  - `hierarchy_reconstruct`: the same workers, but a model reconstructor merges the reports. It gets B/4 held back, and isolates how hard the merge is.
  - `hierarchy`: a model planner (B/8) sees the question and one example ledger and writes the worker instruction. Workers run in parallel, and the reconstructor answers from the reports alone.
- Every role is told to add a list one number at a time with running totals. Without that, 4B summed 12 numbers in one expression and slipped.

### Smoke tests and fixes (scratchpad, not kept)

- **First version:** 4B, amounts 1-60, a top-customer question. Everything was wrong. Workers summed 12 numbers in one line, and the oracle average instruction was ambiguous.
- **Second version:** a general "running totals" hint. Per-store totals rose to about 78%. However:
  - The answer needs all 16 reports right, so exact match compounded to about 0.78^16.
  - The hint made workers keep one grand total even when told to filter or group.
- **Changes:**
  - Amounts became 1-9.
  - The top-customer question became customer totals.
  - The hint now applies only when adding a list.
  - Single-number answers record `abs_error`.
- **Harness bug, now fixed:** 12B writes one `REPORT:` line per customer, and `extract_report` kept only the last one. It now keeps everything after the first marker. Customer-total reports in the smoke tests before the fix were truncated to one customer, for both code and model merges.

### Worker probe: `results/ledger_worker_probe/`

Each cell is a single-store worker with the correct instruction, 20 workers per cell, re-scored after the report fix:

| model | rows | total | average | customer totals (all right) | per customer |
|---|---|---|---|---|---|
| 4B | 4 | 19/20 | 3/20 | 5/20 | 0.49 |
| 4B | 8 | 17/20 | 0/20 | 1/20 | 0.18 |
| 4B | 12 | 14/20 | 0/20 | 0/20 | 0.16 |
| 12B | 4 | 18/20 | 15/20 | 20/20 | 1.00 |
| 12B | 8 | 20/20 | 17/20 | 17/20 | 0.97 |
| 12B | 12 | 20/20 | 15/20 | 13/20 | 0.93 |

4B cannot filter or group even a 4-row ledger: it keeps a total over every row, or loses track. 12B can, and its remaining average failures are genuine filtering slips. Decision: 12B for every role and 8 rows per store.

### Calibration: Gemma-3 12B, `results/ledger_calibrate/`

10 groups × 4, 8, 16 and 32 stores × 3 questions, at budgets of 8k and 16k, all five systems. 12B handles every role. The run produced 1,200 records with no errors; 15 rate-limit retries all recovered.

Target accuracy at 16k, with 4/8/16/32 stores:

| system | total | average | customer totals |
|---|---|---|---|
| single agent | 0.60/0.30/0.00/0.00 | 0.70/0.50/0.20/0.00 | 0.89/0.61/0.12/0.00 |
| ensemble (B/3) | 0.40/0.30/0.00/0.00 | 0.80/0.50/0.10/0.00 | 0.92/0.61/0.06/0.00 |
| oracle split, code merge | 1.00/0.80/0.70/0.80 | 0.40/0.50/0.60/0.40 | 0.85/0.71/0.66/0.36 |
| oracle split, model merge | 1.00/0.90/0.90/0.30 | 0.30/0.60/0.50/0.50 | 0.87/0.66/0.35/0.07 |
| model planner + model merge | 0.90/0.70/0.70/0.10 | 0.30/0.10/0.20/0.40 | 0.80/0.62/0.30/0.00 |

Critical-path output tokens at 16k (the time proxy), with 4 and 32 stores:

| system | total | average | customer totals |
|---|---|---|---|
| single agent | 383 / 1211 | 310 / 763 | 621 / 2648 |
| oracle split, code merge | 74 / 83 | 197 / 276 | 109 / 143 |
| model planner + model merge | 277 / 677 | 536 / 699 | 958 / 625 |

Findings:

- **On totals, the full model hierarchy beats the single agent at every width.** At 16 stores it scores 0.70 against 0.00, with the planner and the reconstructor both written by the model. Every planner instruction for totals was usable (40/40). The single agent's error grows with width: its median miss is 0 at 4 stores, 12 at 8, 26 at 16 and 67 at 32.
- **Averages show the planning trap.** 17/40 planner instructions asked for the category total without the count, and every one of those 17 scored 0. Plans that asked for both scored 0.43. At small widths the single agent beats the oracle split: workers miss the filter about 15% of the time, and 4 workers compound that.
- **For customer totals, the merge is the bottleneck.** With the oracle split, code merge reaches 0.66 at 16 stores, but model merge only 0.35. The reconstructor has to add 16 reports × 8 customers, which is the same wide aggregation the single agent fails at. My keyword check on planner instructions for this question does not separate good plans from bad ones, so there is no plan-quality number yet.
- **Time:** the oracle split has a nearly constant critical path, 74-83 output tokens for totals at every width. Model hierarchies add the planner and reconstructor in sequence but stay below the single agent at large widths.
- **B/N starvation at 8k with 32 stores:** every hierarchy and the ensemble hit the length limit. Each worker gets 250 tokens, and the prompt uses most of them. At 16k only the average question at 32 stores starves, in `hierarchy_reconstruct` and `hierarchy`. The single agent never starves on this task.
- **Caveat:** 10 groups per cell. Differences around 0.1-0.2 are noise.

### Tree merge: `hierarchy_tree_reconstruct` and `hierarchy_tree`

Mergers each combine up to 4 reports into one report in the same format, one tree level per round, and a final reconstructor answers. The merge budget left after the workers is split evenly over the remaining merge calls. With 4 stores the tree is the same as the flat merge. The flat merge's messages are unchanged, checked against all 120 logged calls.

Target accuracy at 16k, with 4/8/16/32 stores:

| system | total | average | customer totals |
|---|---|---|---|
| oracle split, flat model merge | 1.00/0.90/0.90/0.30 | 0.30/0.60/0.50/0.50 | 0.87/0.66/0.35/0.07 |
| oracle split, tree merge | 1.00/0.70/0.70/0.40 | 0.40/0.60/0.60/0.50 | 0.86/0.72/0.60/0.00 |
| model planner, flat merge | 0.90/0.70/0.70/0.10 | 0.30/0.10/0.20/0.40 | 0.80/0.62/0.30/0.00 |
| model planner, tree merge | 0.80/0.70/0.50/0.10 | 0.30/0.20/0.20/0.00 | 0.75/0.65/0.60/0.00 |

- **The tree fixes the customer-totals merge at 16 stores.** It reaches 0.35 → 0.60 with the oracle split and 0.30 → 0.60 with the model planner, close to the code merge's 0.66.
- **At 32 stores the tree starves.** At 16k, all 80 level-1 mergers in the customer-totals problems hit the length limit. Each gets 610 tokens, its input is about 346, and the remaining 264 are too few to add 4 × 8 numbers stepwise. Splitting the merge budget evenly does not scale with tree size; this is the B/N problem moved up one level.
- **Totals and averages:** no clear change, within the noise of 10 groups.
- An `OSError` ("No route to host") stopped the first attempt after 85 problems. `run_wide.py` now retries on any `OSError`, and the run resumed.

## 2026-11-25 to 11-27: main run, `results/ledger_main/`

The main run uses 30 groups × 4, 8, 16 and 32 stores × 3 questions (seed 7301, 8 rows, 12B for every role), which keeps it separate from the calibration problems. It runs from a queue script (scratchpad `night.sh`) with automatic resume. The queue is 12B at 16k, then 12B at 8k, then the 27B (fp8) single agent and ensemble at both budgets.

Run history:

- An `IncompleteRead` (a response cut off mid-body) failed one step. The queue resumed it, and `run_wide.py` now also retries on `http.client.HTTPException`.
- On 11-25, Claude Code stopped the queue because the Mac ran low on memory, after 16k `single_agent`, `ensemble` and `hierarchy_oracle` and 54/360 of `hierarchy_reconstruct`. Natalia committed the partial results.
- On 11-27 the queue was restarted with `--resume` and skipped finished problems. DeepInfra rate-limited heavily early on (about 3,900 retries), but every call went through.
- The queue finished on 11-27: all 7 systems at both budgets for 12B, and the 27B single agent and ensemble at both budgets.

### 16k, complete

Target accuracy with 4/8/16/32 stores:

| system | total | average | customer totals |
|---|---|---|---|
| single agent | 0.40/0.27/0.00/0.00 | 0.73/0.43/0.03/0.10 | 0.88/0.64/0.12/0.03 |
| ensemble (B/3) | 0.43/0.07/0.00/0.00 | 0.87/0.43/0.00/0.03 | 0.94/0.64/0.15/0.00 |
| oracle split, code merge | 1.00/0.83/0.87/0.70 | 0.53/0.63/0.47/0.33 | 0.88/0.82/0.71/0.41 |
| oracle split, flat model merge | 1.00/0.93/0.87/0.50 | 0.57/0.60/0.43/0.27 | 0.86/0.78/0.34/0.01 |
| oracle split, tree merge | 1.00/0.87/0.77/0.27 | 0.53/0.63/0.50/0.33 | 0.86/0.76/0.62/0.00 |
| model planner, flat merge | 0.73/0.83/0.67/0.23 | 0.37/0.17/0.17/0.10 | 0.90/0.72/0.15/0.00 |
| model planner, tree merge | 0.83/0.80/0.70/0.20 | 0.23/0.27/0.20/0.07 | 0.90/0.70/0.30/0.00 |

Critical-path output tokens with 4/8/16/32 stores:

| system | total | average | customer totals |
|---|---|---|---|
| single agent | 414/532/632/1209 | 366/449/449/772 | 631/902/1631/2835 |
| ensemble | 527/710/672/1234 | 402/551/595/1057 | 754/980/1851/3028 |
| oracle split, code merge | 79/83/87/84 | 175/206/229/260 | 114/122/133/148 |
| oracle split, flat model merge | 196/258/374/511 | 375/550/647/664 | 644/1014/1693/2437 |
| oracle split, tree merge | 203/284/324/406 | 361/598/768/902 | 605/1121/1576/799 |
| model planner, flat merge | 269/366/430/585 | 462/502/606/693 | 1103/1574/1716/349 |
| model planner, tree merge | 269/417/429/542 | 456/576/786/771 | 1071/1658/1799/936 |

The 32-store customer-totals critical tokens for the tree and the model planner are low because merge calls starve and stop early, not because they are fast.

### 8k, complete

Target accuracy with 4/8/16/32 stores:

| system | total | average | customer totals |
|---|---|---|---|
| single agent | 0.30/0.27/0.00/0.00 | 0.73/0.40/0.00/0.03 | 0.91/0.65/0.14/0.00 |
| ensemble (B/3) | 0.30/0.13/0.00/0.00 | 0.80/0.60/0.00/0.03 | 0.91/0.65/0.00/0.00 |
| oracle split, code merge | 1.00/0.90/0.87/0.00 | 0.53/0.60/0.43/0.00 | 0.88/0.82/0.67/0.00 |
| oracle split, flat model merge | 1.00/0.93/0.87/0.00 | 0.57/0.53/0.47/0.00 | 0.86/0.77/0.30/0.00 |
| model planner, flat merge | 0.77/0.77/0.67/0.00 | 0.33/0.17/0.10/0.00 | 0.89/0.55/0.00/0.00 |
| oracle split, tree merge | 1.00/0.87/0.77/0.00 | 0.53/0.63/0.43/0.00 | 0.87/0.75/0.00/0.00 |
| model planner, tree merge | 0.80/0.87/0.53/0.00 | 0.27/0.23/0.13/0.03 | 0.81/0.14/0.00/0.00 |

At 32 stores every split system and the ensemble starve on every problem: each worker gets 250 tokens or fewer, and the prompt uses most of them. The single agent never starves. Below 32 stores, the 8k results match 16k for the flat systems, so their gains do not depend on a large budget until B/N gets too small.

The tree merge starves earlier at 8k. With the oracle split, every 16-store customer-totals problem and 70% of 16-store averages starve. With the model planner, 73% of 8-store customer-totals problems starve (0.14, against 0.70 at 16k). The even split of the merge budget is the bottleneck.

### 27B single agent and ensemble

Gemma 3 27B (deepinfra/fp8) on the same 360 problems, target accuracy with 4/8/16/32 stores, next to the 12B oracle split with code merge:

| system | budget | total | average | customer totals |
|---|---|---|---|---|
| 27B single agent | 16k | 1.00/0.90/0.80/0.17 | 1.00/0.83/0.53/0.57 | 0.91/0.83/0.30/0.02 |
| 27B ensemble (B/3) | 16k | 1.00/0.97/0.93/0.07 | 0.97/0.83/0.60/0.73 | 0.93/0.85/0.30/0.00 |
| 12B oracle split, code merge | 16k | 1.00/0.83/0.87/0.70 | 0.53/0.63/0.47/0.33 | 0.88/0.82/0.71/0.41 |
| 27B single agent | 8k | 1.00/0.93/0.83/0.13 | 0.87/0.70/0.40/0.60 | 0.92/0.82/0.32/0.02 |
| 27B ensemble (B/3) | 8k | 1.00/0.97/0.87/0.00 | 0.97/0.80/0.63/0.00 | 0.92/0.82/0.07/0.00 |
| 12B oracle split, code merge | 8k | 1.00/0.90/0.87/0.00 | 0.53/0.60/0.43/0.00 | 0.88/0.82/0.67/0.00 |

27B critical-path output tokens at 16k:

| system | total | average | customer totals |
|---|---|---|---|
| 27B single agent | 402/454/785/1348 | 312/428/603/991 | 587/936/1734/2669 |
| 27B ensemble | 464/464/837/1883 | 344/496/739/1089 | 629/1053/1987/2890 |

The 27B single agent never starves at either budget. At 8k the 27B ensemble starves on every 32-store problem and on 83% of 16-store customer-totals problems, the same pattern as the 12B ensemble: starvation comes from the split, not the model size.

### Findings from the main run

- **The calibration pattern holds on held-out problems with 30 groups per cell.**
- **Totals:** every split system beats the single agent from 8 stores up. The fully model-run hierarchy (12B planner, workers and reconstructor) scores 0.67 at 16 stores at both budgets, against 0.00 for the single agent.
- **Averages:** the model planner's average scores (0.10-0.37) sit below the oracle split's (0.33-0.63). This matches the calibration finding that planners often leave out the count. At 4 stores the single agent beats every split system, because worker filter slips compound.
- **Customer totals:** the flat model merge is the bottleneck (0.34 against 0.71 with code merging at 16 stores, 16k). The tree merge recovers most of it (0.62), and with the model planner it doubles 0.15 to 0.30.
- **32 stores:** this is the B/N limit at 8k for every split system, and at 16k for the tree merge's level-1 mergers. Only the oracle split with code merge stays strong at 32 stores and 16k (0.70 on totals).
- **Time:** the oracle split's critical path is nearly flat with width (79-87 tokens for totals), while the single agent's grows about 3x from 4 to 32 stores. Model hierarchies add planner and merge rounds but stay below the single agent at 16 and 32 stores.
- **27B against 12B structure:** the 27B single agent closes most of the gap up to 16 stores on totals and beats every 12B hierarchy on averages at every width. The 12B oracle split still wins on the largest tasks at 16k: totals at 32 stores (0.70 against 0.17), customer totals at 16 stores (0.71 against 0.30) and 32 stores (0.41 against 0.02). A bigger model moves the point where splitting pays off from about 8 stores to 16-32; it does not remove it. The critical-path gain holds at every width, and the 27B failures at 32 stores are errors over a long computation, not starvation. The 12B model planner is below the 27B single agent almost everywhere, so planning remains the weak link.

## 2026-11-27: dependence in the ledger task, `results/ledger_depth/`

### Why and design

The ledger task had no interdependence, the proposal's second axis. Natalia chose to add it for preliminary poster results.

- `generate_ledger.py --linked N --depths ...` gives N stores one extra row like `Eli, tools, one tenth of Store 7's total, rounded down`, inserted at a random position. A store's total includes its own linked row, so the links form chains, and the deepest chain is exactly the given depth.
- Every depth reuses the group's ledgers. Depth 0 is the control: the same amounts as depth 1, written as plain numbers, so depth 0 against depth 1 is the cost of a reference row and depth 1 against 8 is the cost of chain depth with the number of linked stores fixed.
- The default generator output is unchanged (checked byte for byte against `results/ledger_main/problems.jsonl`). An independent resolver (scratchpad `check_depth.py`) verified all 540 answers, depths, and store totals.
- `run_ledger.py` runs workers in one round per link level. A linked worker is given its source store's total as read from that store's worker report, so upstream slips carry downstream. The model-planned hierarchies are blocked on linked problems, because the total is read from the oracle report format.
- Problems: seed 7401, 30 groups, 8 stores (4 linked, depths 0/1/2/4) and 16 stores (8 linked, depths 0/1/2/4/8), total and customer-totals questions: 540 problems. Run with 12B at 16k: single agent, ensemble, oracle split with code merge, oracle split with flat model merge.

Smoke-test fix: with the source total stated before the ledger, 12B workers started their running total with it. It now comes after the ledger, with "It is not a row of this ledger; use it only to work out the amount of the row that refers to Store j."

The run took 1.5 hours. One step failed on a malformed OpenRouter response (`JSONDecodeError`) and the queue resumed it.

### Results

Target accuracy by depth (8 stores: 0/1/2/4; 16 stores: 0/1/2/4/8):

| system | 8 stores, total | 8 stores, customer totals | 16 stores, total | 16 stores, customer totals |
|---|---|---|---|---|
| single agent | 0.27/0.07/0.00/0.07 | 0.60/0.26/0.16/0.21 | 0.03/0.00/0.00/0.00/0.00 | 0.13/0.00/0.01/0.02/0.00 |
| ensemble (B/3) | 0.20/0.00/0.03/0.03 | 0.57/0.20/0.18/0.20 | 0.00/0.00/0.00/0.00/0.00 | 0.09/0.00/0.00/0.00/0.00 |
| oracle split, code merge | 0.87/0.20/0.37/0.43 | 0.73/0.45/0.42/0.53 | 0.73/0.10/0.20/0.27/0.27 | 0.58/0.22/0.28/0.24/0.26 |
| oracle split, flat model merge | 0.83/0.33/0.33/0.37 | 0.60/0.47/0.41/0.51 | 0.77/0.17/0.13/0.30/0.27 | 0.28/0.15/0.15/0.15/0.13 |

Critical-path output tokens by depth:

| system | 8 stores, total | 8 stores, customer totals | 16 stores, total | 16 stores, customer totals |
|---|---|---|---|---|
| single agent | 512/892/985/1012 | 932/1595/1577/1767 | 642/1144/1111/1281/1467 | 1848/1940/1936/2033/1926 |
| ensemble | 691/1024/1086/1248 | 1137/1923/1929/2111 | 650/1400/1382/1658/1991 | 2109/2218/2494/2367/2482 |
| oracle split, code merge | 92/233/332/560 | 137/344/453/743 | 93/290/428/620/1048 | 132/346/538/798/1400 |
| oracle split, flat model merge | 281/416/516/750 | 1115/1251/1398/1747 | 356/572/681/857/1289 | 1554/1946/2033/2206/2811 |

Rounds are depth + 1 for the oracle split and depth + 2 with the model merge. Almost nothing starved: up to 10% of ensemble problems and 3% of model-merge problems on 16-store customer totals.

Linked-worker diagnosis (oracle split, totals, 1,320 linked workers): given the total they were passed, 81% report the right store total, against 98% of unlinked workers. The rate does not depend on depth (0.72-0.85 per cell). It does depend on where the row sits: 0.78 in the middle of the ledger, 0.92-0.93 as the first or last row. The errors are small: a skipped row (45), rounding up instead of down (28), and off-by-a-few slips; only one worker added the full total.

### Findings

- **A reference row costs much more accuracy than chain depth.** Every system drops sharply from depth 0 to depth 1, with the same amounts and the same number of rounds for the baselines. From depth 1 to 8, accuracy is flat or rises slightly within noise. "One tenth, rounded down" mostly absorbs upstream slips, so errors do not compound along the chain.
- **The hierarchy keeps its accuracy lead under dependence.** At every depth the oracle split beats the single agent, for example 16-store totals 0.27 against 0.00 at depth 8, and 8-store customer totals 0.53 against 0.21 at depth 4.
- **The time gain shrinks linearly with depth, as predicted.** The oracle split's critical path grows from 93 to 1048 tokens on 16-store totals as depth goes 0 to 8, while the single agent's grows from 642 to 1467, mostly at the step from depth 0 to 1. At depth 8 the oracle split's advantage is about 1.4x on totals and on customer totals; with a model merge it nearly disappears on totals (1289 against 1467) and reverses on customer totals (2811 against 1926).
- So dependence costs the hierarchy time, not accuracy, in this task. The accuracy cost of dependence comes from the harder rows, and it hits every system.

### Next tests, in priority order

1. A merge budget proportional to each call's work (values to add), to test whether the 32-store tree collapse is a budgeting artifact.
2. A 27B planner and reconstructor with 12B workers, to see whether a bigger lead fixes the average plan and the wide merge, compared with the 27B single agent at similar cost.
3. Dependence follow-ups: a dependence where upstream errors do compound (the full source total instead of a tenth), and a model planner that must find the dependency order itself.
4. A question where merging needs everything (median sale, distinct customers), where the prediction is that the hierarchy loses. This bounds the claim.
5. Another model family at similar sizes, once the design is settled.
6. Bootstrap confidence intervals on the 30-group cells.

### Not yet run on the chain task

- More groups per cell.
- A plan format where the planner names each chain's variables and code copies the equations, which would cut planner output and copying errors.

### Open items

- Tell Lily about the switch to Gemma, the wide-task design, and the reversed interdependence prediction.
- Check with Lia whether her runs depend on anything here. Nothing shared was changed.
- The OpenRouter key was pasted into the chat. Rotate it after the project.
- Code and results through 11-27, including the dependence run, are committed. `.venv-or/` in the repo is a local environment and should not be committed.
- `__pycache__/` is tracked since commit b70e92e. It should be untracked and added to `.gitignore`.
