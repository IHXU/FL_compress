#!/usr/bin/env python3
"""Resume all unfinished cfg_main_exp experiments with one worker per GPU."""

from __future__ import annotations

import argparse
import os
import queue
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
LOG_DIR = ROOT / "launch_logs_main_new_resume"
CHECKPOINT_RE = re.compile(r"state_round_(\d+)\.pth$")


@dataclass(frozen=True)
class Job:
    config: Path
    name: str
    save_path: Path
    rounds: int


def latest_checkpoint(save_path: Path) -> tuple[Path | None, int]:
    latest_path = None
    latest_round = -1
    for path in save_path.glob("state_round_*.pth"):
        match = CHECKPOINT_RE.match(path.name)
        if match and int(match.group(1)) > latest_round:
            latest_path = path
            latest_round = int(match.group(1))
    return latest_path, latest_round


def load_jobs() -> tuple[list[Job], list[Job]]:
    pending = []
    completed = []
    for config in sorted(CONFIG_DIR.glob("*.yaml")):
        with config.open(encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        save_path = Path(cfg["experiment"]["save_path"])
        if not save_path.is_absolute():
            save_path = (ROOT / save_path).resolve()
        job = Job(
            config=config,
            name=cfg["experiment"]["name"],
            save_path=save_path,
            rounds=int(cfg["training"]["rounds"]),
        )
        _, saved_round = latest_checkpoint(save_path)
        (completed if saved_round >= job.rounds - 1 else pending).append(job)
    return pending, completed


def detect_gpus(requested: str | None) -> list[str]:
    if requested:
        return [item.strip() for item in requested.split(",") if item.strip()]
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_job(job: Job, gpu: str, max_retries: int) -> bool:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{job.name}.out"

    for attempt in range(max_retries + 1):
        checkpoint, saved_round = latest_checkpoint(job.save_path)
        if saved_round >= job.rounds - 1:
            print(f"[{stamp()}] [GPU {gpu}] SKIP completed {job.name}", flush=True)
            return True

        command = [sys.executable, "run_experiment.py", "--config", str(job.config)]
        if checkpoint is not None:
            command.extend(["--resume", str(checkpoint)])

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["PYTHONUNBUFFERED"] = "1"
        start_round = saved_round + 2 if checkpoint is not None else 1
        print(
            f"[{stamp()}] [GPU {gpu}] START {job.name} at round {start_round}/{job.rounds} "
            f"attempt {attempt + 1}/{max_retries + 1} log={log_path}",
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


def worker(gpu: str, jobs: queue.Queue[Job], max_retries: int, failures: list[str]) -> None:
    while True:
        try:
            job = jobs.get_nowait()
        except queue.Empty:
            return
        try:
            if not run_job(job, gpu, max_retries):
                failures.append(job.name)
        finally:
            jobs.task_done()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", help="Comma-separated physical GPU IDs (default: all)")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retries after a non-zero exit; each retry resumes the newest checkpoint",
    )
    args = parser.parse_args()

    pending, completed = load_jobs()
    gpus = detect_gpus(args.gpus)
    if not gpus:
        raise RuntimeError("No GPUs detected")

    print(
        f"[{stamp()}] main_new resume queue: pending={len(pending)} "
        f"completed={len(completed)} GPUs={','.join(gpus)}",
        flush=True,
    )
    for job in completed:
        print(f"[{stamp()}] already complete: {job.name}", flush=True)
    if not pending:
        return 0

    job_queue: queue.Queue[Job] = queue.Queue()
    for job in pending:
        job_queue.put(job)

    failures: list[str] = []
    threads = [
        threading.Thread(
            target=worker,
            args=(gpu, job_queue, args.max_retries, failures),
            name=f"gpu-{gpu}",
        )
        for gpu in gpus
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    print(
        f"[{stamp()}] queue finished: failed={len(failures)} "
        f"names={','.join(failures) if failures else 'none'}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
