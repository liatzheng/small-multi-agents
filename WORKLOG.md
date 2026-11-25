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

### Not yet run

- More groups per cell.
- A plan format where the planner names each chain's variables and code copies the equations, which would cut planner output and copying errors.

### Open items

- Tell Lily about the switch to Gemma, the wide-task design, and the reversed interdependence prediction.
- Check with Lia whether her runs depend on anything here. Nothing shared was changed.
- The OpenRouter key was pasted into the chat. Rotate it after the project.
- The new files and results are not committed yet. `.venv-or/` in the repo is a local environment and should not be committed.
