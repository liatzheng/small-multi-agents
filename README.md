# Small multi-agent experiments

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

Each output line includes the problem, answer key, final response, parsed prediction, correctness, and token counts. The budget covers the tokenized chat template and all generated tokens, including reasoning and the end-of-sequence token. The scorer reads an ordered pair such as `(7, 5)` on the final line or a single explicitly boxed pair at the end of the final response. It compares the two integers in order. The separate `format_ok` flag records whether the entire final response was just the requested pair. Unparseable and truncated responses count as incorrect and have separate status labels. The output file is created fresh and each result is flushed as it completes.

The [two-problem sample log](results/single_agent_smoke.jsonl) was produced on Apple Silicon with the settings above.
