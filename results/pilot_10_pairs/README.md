# Ten-pair arithmetic pilot

Each system ran once on the same 20 problems. Each pair uses the same numbers, distractor, and answer format. Only the operand in the second operation changes. The independent version uses a given input and the dependent version uses the result of the first operation.

## Accuracy and token usage

Token columns show means per problem across all 10 examples of each kind. Input includes every copy of the chat template, problem, and messages read. Output includes thinking and stop tokens. The denominator includes every recorded outcome.

| System | Problem | Correct | Accuracy | Mean input | Mean output | Mean total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Single agent | Independent | 10/10 | 100% | 72.0 | 393.0 | 465.0 |
| Single agent | Dependent | 10/10 | 100% | 72.0 | 446.0 | 518.0 |
| Ensemble | Independent | 10/10 | 100% | 216.0 | 1,140.0 | 1,356.0 |
| Ensemble | Dependent | 10/10 | 100% | 216.0 | 1,334.9 | 1,550.9 |
| Hierarchy | Independent | 10/10 | 100% | 456.5 | 1,263.9 | 1,720.4 |
| Hierarchy | Dependent | 10/10 | 100% | 468.4 | 1,056.8 | 1,525.2 |

## Within-pair comparison

The differences below are dependent minus independent. They average over the same 10 pairs.

| System | Accuracy difference (percentage points) | Mean total token difference |
| --- | ---: | ---: |
| Single agent | +0 | +53.0 |
| Ensemble | +0 | +194.9 |
| Hierarchy | +0 | -195.2 |

All three systems reached full accuracy on both kinds, so the pilot shows no accuracy gain from coordination on these examples. The single agent used the fewest tokens in both groups. The hierarchy passed worker 1’s report to worker 2 on all ten dependent problems and on none of the independent problems. For the hierarchy, dependent problems averaged 11.9 more input tokens and 207.1 fewer output tokens, a net reduction of 195.2 tokens.

## Score audit

All 60 runs stayed within the 4,000-token allowance. The largest observed total was 1,898 tokens.

All 60 predictions matched the answer key.

## Inputs, settings, and logs

- [Shared input and answer keys](problems.jsonl). Generator seed 2026. Both smoke-test prompts were excluded by this draw.
- [Single-agent log](single_agent.jsonl), [ensemble log with individual votes](ensemble.jsonl), and [hierarchy log with messages and stage costs](hierarchy.jsonl).
- Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`, BF16, Apple M2 Max (MPS), PyTorch 2.8.0, Transformers 4.51.3. Each system used its existing prompts, scoring, allocation, and seeded sampling procedure. The starting sampling seed was 42 for every problem.
- The inference code was at commit `64b267982211da695b9790d27d01ac43a70890bf`. The equal 4,000-token cap was applied separately to each system on each problem.

To reproduce from the repository root in the project Python environment, save fresh logs to new paths.

```sh
python generate_problems.py --pairs 10 --seed 2026 > pilot-problems.jsonl
python run_single_agent.py pilot-problems.jsonl --budget 4000 --seed 42 --output single-agent-rerun.jsonl
python run_ensemble.py pilot-problems.jsonl --budget 4000 --seed 42 --output ensemble-rerun.jsonl
python run_hierarchy.py pilot-problems.jsonl --budget 4000 --seed 42 --output hierarchy-rerun.jsonl
```

This pilot covers one model, one starting seed for each system, one budget, and single-digit problems with two arithmetic steps. Accuracy changes by 10 percentage points for each additional correct problem within a kind.
