#!/usr/bin/env python3
"""Run every unfinished, currently-idle cfg_main_exp job in parallel.

The launcher is intended to live inside tmux.  It never starts a configuration
that is already complete or whose config path is present in a running
``run_experiment.py`` command.  Failed children are retried from their newest
checkpoint.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "configs" / "cfg_main_exp"
LOG_DIR = ROOT / "launch_logs_main_new_all_parallel"
CHECKPOINT_RE = re.compile(r"state_round_(\d+)\.pth$")
GPU_ASSIGNMENT = ["0"] * 7 + ["1"] * 6 + ["2"] * 6 + ["3"] * 6


@dataclass(frozen=True)
class Job:
    config: Path
    name: str
    save_path: Path
    rounds: int


def stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def latest_checkpoint(save_path: Path) -> tuple[Path | None, int]:
    latest_path = None
    latest_round = -1
    for path in save_path.glob("state_round_*.pth"):
        match = CHECKPOINT_RE.match(path.name)
        if match and int(match.group(1)) > latest_round:
            latest_path = path
            latest_round = int(match.group(1))
    return latest_path, latest_round


def running_command_lines() -> str:
    result = subprocess.run(
        ["ps", "-eo", "args="], capture_output=True, text=True, check=True
    )
    return result.stdout


def pending_idle_jobs() -> tuple[list[Job], int, int]:
    process_text = running_command_lines()
    pending: list[Job] = []
    complete = 0
    running = 0

    for config in sorted(CONFIG_DIR.glob("*.yaml")):
        with config.open(encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        save_path = Path(cfg["experiment"]["save_path"])
        if not save_path.is_absolute():
            save_path = (ROOT / save_path).resolve()
        job = Job(
            config=config.resolve(),
            name=cfg["experiment"]["name"],
            save_path=save_path,
            rounds=int(cfg["training"]["rounds"]),
        )
        _, saved_round = latest_checkpoint(save_path)
        if saved_round >= job.rounds - 1:
            complete += 1
        elif str(job.config) in process_text or str(config) in process_text:
            running += 1
        else:
            pending.append(job)
    return pending, complete, running


def run_job(job: Job, gpu: str, max_retries: int = 2) -> bool:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{job.name}.out"

    for attempt in range(max_retries + 1):
        checkpoint, saved_round = latest_checkpoint(job.save_path)
        if saved_round >= job.rounds - 1:
            print(f"[{stamp()}] [GPU {gpu}] SKIP complete {job.name}", flush=True)
            return True

        command = [sys.executable, "run_experiment.py", "--config", str(job.config)]
        if checkpoint is not None:
            command.extend(["--resume", str(checkpoint)])

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["PYTHONUNBUFFERED"] = "1"
        print(
            f"[{stamp()}] [GPU {gpu}] START {job.name} "
            f"checkpoint={saved_round} attempt={attempt + 1}/{max_retries + 1}",
            flush=True,
        )
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            log.write(
                f"\n\n===== {stamp()} GPU={gpu} attempt={attempt + 1} "
                f"resume={checkpoint or 'none'} =====\n"
            )
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            log.write(f"===== {stamp()} exit_code={result.returncode} =====\n")

        if result.returncode == 0:
            print(f"[{stamp()}] [GPU {gpu}] DONE {job.name}", flush=True)
            return True
        print(
            f"[{stamp()}] [GPU {gpu}] EXIT {job.name} code={result.returncode}",
            flush=True,
        )
        if attempt < max_retries:
            time.sleep(30)
    return False


def main() -> int:
    jobs, complete, running = pending_idle_jobs()
    if len(jobs) > len(GPU_ASSIGNMENT):
        raise RuntimeError(
            f"Found {len(jobs)} pending jobs but only "
            f"{len(GPU_ASSIGNMENT)} GPU assignments"
        )
    print(
        f"[{stamp()}] all-parallel snapshot: pending={len(jobs)} "
        f"running={running} complete={complete}",
        flush=True,
    )
    if not jobs:
        return 0

    failures: list[str] = []
    lock = threading.Lock()

    def target(job: Job, gpu: str) -> None:
        if not run_job(job, gpu):
            with lock:
                failures.append(job.name)

    threads = [
        threading.Thread(target=target, args=(job, gpu), name=job.name)
        for job, gpu in zip(jobs, GPU_ASSIGNMENT)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    print(
        f"[{stamp()}] all-parallel finished failures={len(failures)} "
        f"names={','.join(failures) if failures else 'none'}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
