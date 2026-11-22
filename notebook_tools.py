"""Helpers shared by the Colab notebooks."""

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def use_hf_token():
    """Pass the HF_TOKEN Colab secret to the runners, which inherit this environment."""
    from google.colab import userdata

    try:
        os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    except (userdata.SecretNotFoundError, userdata.NotebookAccessError):
        print("No HF_TOKEN secret with notebook access; Hugging Face downloads stay unauthenticated.")
    else:
        print("Using the HF_TOKEN Colab secret.")


def load_runs(folder):
    """Return one row per logged problem in folder/b<budget>/<system>.jsonl."""
    rows = []
    for path in sorted(Path(folder).glob("b*/*.jsonl")):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            ran_out = r["status"] == "length_limit" or any(
                a["status"] == "length_limit" for a in r.get("attempts", []))
            rows.append({
                "system": path.stem, "model": r.get("model"), "thinking": r.get("thinking", True),
                "budget": r["budget"], "d": r.get("chain_length"),
                "kind": r["kind"], "pair_id": r["pair_id"],
                # Problems generated before the modulus field existed all used 97.
                "modulus": r.get("modulus", 97), "status": r["status"], "correct": r["correct"],
                "ran_out": ran_out, "failed_stage": r.get("failed_stage"),
                "output": r["output_tokens"], "total": r["total_tokens"],
            })
    return rows


def run_logged(args, log_path):
    """Run a command, show its output, and append every line to log_path.

    The log is reopened for each line because Colab's Drive mount showed no new
    output in a file that tee held open for a whole run.
    """
    def append(text):
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(text)

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    header = f"=== {' '.join(args)}\n=== started {stamp}\n"
    print(header, end="")
    append(header)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
    with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, env=env) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            append(line)
    if process.returncode:
        append(f"=== exited with {process.returncode}\n")
        raise RuntimeError(f"{args[0]} exited with {process.returncode}; see {log_path}")
