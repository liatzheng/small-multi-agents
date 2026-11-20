"""Helpers shared by the Colab notebooks."""

import os
import subprocess
from datetime import datetime, timezone


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
