"""Run the single agent, ensemble, and hierarchy with batched vLLM generation."""

import argparse
import time
from pathlib import Path

from run_ensemble import solve_ensemble_steps
from run_hierarchy import solve_hierarchy_steps
from run_single_agent import (
    MODEL, REVISION, final_answer, log_result, print_summary, read_problems, resume_log, solve_steps,
)

SYSTEMS = {
    "single_agent": solve_steps,
    "ensemble": solve_ensemble_steps,
    "hierarchy": solve_hierarchy_steps,
}


class VllmChat:
    """Answer batches of chat requests under the same budget rules as generate_chat."""

    def __init__(self, model_name, revision, max_model_len):
        import torch
        import vllm
        from transformers import AutoTokenizer, GenerationConfig

        # T4 has no native BF16, so it runs FP16 as load_model does.
        native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        self.dtype = "bfloat16" if native_bf16 else "float16"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        eos = GenerationConfig.from_pretrained(model_name, revision=revision).eos_token_id
        self.eos = eos if isinstance(eos, list) else [eos]
        self.llm = vllm.LLM(
            model=model_name, revision=revision, tokenizer_revision=revision,
            dtype=self.dtype, max_model_len=max_model_len,
        )
        self.sampling_params = vllm.SamplingParams
        print(f"vllm {vllm.__version__}, torch {torch.__version__}, "
              f"{torch.cuda.get_device_name()}, {self.dtype}", flush=True)

    def prompt_ids(self, messages):
        # Rendering to text first gives the same IDs as tokenize=True across Transformers versions.
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def generate(self, requests):
        replies = [None] * len(requests)
        prompts, params, sent = [], [], []
        for i, request in enumerate(requests):
            ids = self.prompt_ids(request["messages"])
            cap, seed = request["cap"], request["seed"]
            if len(ids) >= cap:
                replies[i] = {
                    "stop_reason": "prompt_too_long", "required_input_tokens": len(ids),
                    "final_text": "", "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                    "budget": cap, "sampling_seed": seed,
                }
                continue
            prompts.append({"prompt_token_ids": ids})
            params.append(self.sampling_params(
                temperature=0.6, top_p=0.95, top_k=20, max_tokens=cap - len(ids), seed=seed,
            ))
            sent.append((i, len(ids), cap, seed))
        outputs = self.llm.generate(prompts, params, use_tqdm=False) if prompts else []
        for (i, input_tokens, cap, seed), output in zip(sent, outputs):
            completion = output.outputs[0]
            output_ids = list(completion.token_ids)
            stopped = completion.finish_reason == "stop"
            output_tokens = len(output_ids)
            # generate_chat counts the end-of-sequence token, so count it here if vLLM dropped it.
            if stopped and (not output_ids or output_ids[-1] not in self.eos):
                output_tokens += 1
            replies[i] = {
                "stop_reason": "stop" if stopped else "length_limit",
                "final_text": final_answer(self.tokenizer, output_ids),
                "input_tokens": input_tokens, "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens, "budget": cap, "sampling_seed": seed,
            }
        return replies


def run_batch(all_steps, chat):
    """Advance every solver together, sending each round of requests as one batch."""
    results = [None] * len(all_steps)
    replies = {i: None for i in range(len(all_steps))}
    while replies:
        requests = {}
        for i, reply in replies.items():
            try:
                requests[i] = all_steps[i].send(reply)
            except StopIteration as done:
                results[i] = done.value
        answers = iter(chat.generate([r for i in requests for r in requests[i]]))
        replies = {i: [next(answers) for _ in requests[i]] for i in requests}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problems", type=Path, help="JSONL file from the problem generator")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="logs go to OUT_DIR/b<budget>/<system>.jsonl")
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(SYSTEMS))
    parser.add_argument("--budgets", nargs="+", type=int, default=[4000],
                        help="total tokens per problem, one run per budget")
    parser.add_argument("--seed", type=int, default=42, help="starting sampling seed for each problem")
    parser.add_argument("--model", default=MODEL, help="Hugging Face model ID")
    parser.add_argument("--revision", help=f"model commit (required unless --model is {MODEL})")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="problems per batch; results are saved after each batch")
    parser.add_argument("--resume", action="store_true",
                        help="continue existing logs from their first unfinished problem")
    args = parser.parse_args()
    if not all(1 <= budget <= 32768 for budget in args.budgets):
        parser.error("--budgets must be between 1 and 32768")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.revision is None:
        if args.model != MODEL:
            parser.error("--revision is required with --model so the weights stay pinned")
        args.revision = REVISION
    try:
        problems = read_problems(args.problems)
    except (ValueError, OSError) as error:
        parser.error(str(error))

    jobs = []
    for budget in args.budgets:
        for system in args.systems:
            path = args.out_dir / f"b{budget}" / f"{system}.jsonl"
            settings = {
                "model": args.model, "revision": args.revision, "budget": budget,
                "seed": args.seed, "backend": "vllm",
            }
            try:
                finished = resume_log(path, problems, settings, args.resume)
            except ValueError as error:
                parser.error(str(error))
            jobs.append((system, budget, path, settings, finished))
    if all(len(finished) == len(problems) for *_, finished in jobs):
        print("All runs are already finished.")
        return

    # A prompt plus its output never exceeds the problem budget.
    chat = VllmChat(args.model, args.revision, max_model_len=max(args.budgets))
    for system, budget, path, settings, finished in jobs:
        print(f"=== {system}, budget {budget}", flush=True)
        results = list(finished)
        path.parent.mkdir(parents=True, exist_ok=True)
        for start in range(len(finished), len(problems), args.batch_size):
            batch = problems[start:start + args.batch_size]
            began = time.perf_counter()
            batch_results = run_batch(
                [SYSTEMS[system](problem, budget, args.seed) for problem in batch], chat,
            )
            # Batched problems share wall time, so each gets an equal share.
            seconds = round((time.perf_counter() - began) / len(batch), 2)
            for result in batch_results:
                result.update(settings, device="cuda", dtype=chat.dtype,
                              seconds=seconds, batch_size=len(batch))
                log_result(path, result)
                results.append(result)
        print_summary(results, path)


if __name__ == "__main__":
    main()
