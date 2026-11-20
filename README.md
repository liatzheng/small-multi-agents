# Small multi-agent experiments

## Run on Google Colab

[Open the notebook in Colab](https://colab.research.google.com/github/liatzheng/small-multi-agents/blob/main/colab.ipynb)

Select **Runtime → Change runtime type → T4 GPU or A100 GPU**, then run the notebook cells in order and approve Google Drive access. It loads Qwen3-4B once and runs all three systems on the same problems. T4 uses FP16 and A100 uses BF16, with full model weights. CUDA loading streams weights onto the GPU to reduce host RAM use. Colab's installed CUDA PyTorch is retained, and its version is recorded alongside the model revision, GPU, precision, seeds, and repository commit.

The defaults reproduce the existing length-2 and length-4 calibration pairs with a 4,000-token cap per problem per system. Change `PAIRS_PER_LENGTH` to `10` and set a new `RUN_NAME` for a larger pilot. The budget includes every input prompt and generated token across the system's calls.

The notebook saves each completed problem to `MyDrive/small-multi-agents/runs/<RUN_NAME>/`. Reconnect with the same GPU type and rerun the cells to resume saved progress. Use a new run name when changing the GPU, settings, or code version. It exports the three JSONL logs, accuracy and token-usage tables, status counts, problem inputs, and run configuration in that folder. Copy these outputs into `results/` when adding a finished run to GitHub. Model outputs can vary with GPU precision and software versions, so run all three systems in the same environment for each comparison.

## Run the single-agent baseline

From the repository root, install the dependencies and generate one pair of problems. The first run downloads about 8 GB of model weights. The runner uses CUDA, Apple Silicon's MPS, or CPU, in that order.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python generate_problems.py --pairs 1 --seed 42 > problems.jsonl
python run_single_agent.py problems.jsonl --budget 4000 --seed 42 --output results.jsonl
```

Increase `--pairs` to generate a larger dataset. Each input line gets one fresh chat with the same system instruction. Only the problem text is sent to the model. The first seed selects problems and the second sets the sampling seed for each chat. The runner uses the original Qwen3-4B weights with thinking enabled and samples with temperature 0.6, top-p 0.95, and top-k 20. [Qwen model card](https://huggingface.co/Qwen/Qwen3-4B)

Each output line includes the problem, answer key, final response, parsed prediction, correctness, and token counts. The budget covers the tokenized chat template and all generated tokens, including reasoning and the end-of-sequence token. The scorer reads an ordered pair such as `(7, 5)` on the final line or a single explicitly boxed pair at the end of the final response. It compares the two integers in order. The separate `format_ok` flag records whether the entire final response was just the requested pair. Unparseable and truncated responses count as incorrect and have separate status labels. The output file is created fresh and each result is flushed as it completes. Each record also stores the model, revision, starting seed, and wall-clock seconds for the problem.

All three runners accept the same extra flags. `--resume` continues an existing output file from its first unfinished problem. It refuses a file whose records do not match the problem file, model, revision, budget, and seed, and it drops a final line cut off by an interrupted write. `--model` selects another Hugging Face model and requires `--revision` with a commit hash, so the weights stay pinned.

The [two-problem sample log](results/single_agent_smoke.jsonl) was produced on Apple Silicon with the settings above.

## Run the independent ensemble

Use the same generated input file and environment as the single-agent run.

```sh
python run_ensemble.py problems.jsonl --budget 4000 --seed 42 --output ensemble_results.jsonl
```

Each problem gets three fresh chats with sampling seeds 42, 43, and 44 when using the command above. Each chat sees the same problem and system instruction. The three caps are 1,333, 1,333, and 1,334 tokens, including each copy of the prompt. An attempt that finishes early leaves its remaining tokens unused.

Each completed, parseable answer casts one vote for its ordered pair. Truncated and unparseable responses abstain. A pair needs two of the three votes to win. When no pair gets two votes, the prediction is null and the problem counts as incorrect. Voting uses no model tokens.

The [ensemble sample log](results/ensemble_smoke.jsonl) records the three attempts in order, with their seeds, caps, token counts, final responses, and votes. The outer record contains the majority prediction, correctness, and total token usage for the problem.

## Run the hierarchy

Use the same input JSONL and environment as the baselines.

```sh
python run_hierarchy.py problems.jsonl --budget 4000 --seed 42 --output hierarchy_results.jsonl
```

1. The Qwen3-4B orchestrator reads the problem and writes one assignment for each worker. Its plan includes a boolean indicating whether worker 2 needs worker 1's result.
2. Worker 1 reads the problem and its assignment and sends a short report.
3. Worker 2 reads the problem and its own assignment. It also gets the first report when the plan requests it.
4. The orchestrator reads the original problem, both assignments, and both reports to produce the final pair. The same parser and answer key used by the baselines score its answer.

All roles use fresh chats with the same model revision, thinking mode, and sampling settings as the baselines. The four stages use consecutive seeds starting at `--seed`. The runner releases one quarter of the total budget at each stage and carries unused tokens forward. For a 4,000-token cap, cumulative spending after each stage can reach at most 1,000, 2,000, 3,000, and 4,000 tokens respectively. Each call pays for its full prompt, including repeated problem text and forwarded messages, and all generated tokens.

The [hierarchy sample log](results/hierarchy_smoke.jsonl) contains each stage's messages, final text, cap, stop reason, and token counts. A malformed plan, empty report, or truncated stage ends the problem with an incorrect result and records the failed stage. If a prompt cannot fit its allowance, that stage records its required length and consumes zero model tokens. The runner makes no retries.

## Generate harder dependency problems

```sh
python generate_problems.py --pairs 10 --seed 2028 --chain-length 2 > chains.jsonl
python run_single_agent.py chains.jsonl --budget 4000 --seed 42 --output chain-results.jsonl
```

Each target chain has `--chain-length` updates of the form `next = (factor * previous + offset) mod 97`. The prompt shuffles the equations and interleaves a third chain of the same length as a distractor. The solver must find the two requested variables and return their values in order.

Both members of a pair have the same givens, target labels, coefficients, and equation order. One variable reference changes. In the independent version the second chain starts from its own given value. In the dependent version it starts from the first chain’s final result. At length 2, both prompts contain six equations and four relevant updates. Their longest relevant paths have two and four updates, respectively. [Single-agent calibration](results/chain_calibration/README.md)

## Run on Colab

[budget_check.ipynb](budget_check.ipynb) runs the length-2 calibration pair through all three systems at 4k, 8k, and 16k tokens on an L4 GPU and summarizes accuracy, token use, cap hits, and time. It clones this repository, so push changes before running it. Results are saved in Google Drive under `MyDrive/small-multi-agents/results/budget_check/`, with a `b<budget>/` folder of per-system logs and two summary CSV files. Rerunning the cells after a disconnect resumes each run.
