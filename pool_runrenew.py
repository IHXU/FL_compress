# pool_run.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import itertools
import subprocess
import shlex
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import os
import os.path as osp
import sys
from pathlib import Path

# --- 在这里修改你要运行的命令列表 ---
# 这是一个示例列表，包含了一些会耗费不同时间且可能成功或失败的命令
COMMANDS_TO_RUN = []

# 锚定到脚本自身所在目录，运行新生成的 IID / Non-IID 全知 Krum-space 实验。
# 原来的 cfg_main_exp_noniid 路径已移除。
script_dir = Path(__file__).resolve().parent
run_experiment_path = script_dir / 'run_experiment.py'
config_dirs = [
    script_dir / 'configs' / 'cfg_main_exp_omniscient_krum',
    script_dir / 'configs' / 'cfg_main_exp_noniid_omniscient_krum',
]

all_configs = sorted(
    config
    for config_dir in config_dirs
    for config in config_dir.glob('*.yaml')
)
for config in all_configs:
    cmd = (
        f'{shlex.quote(sys.executable)} '
        f'{shlex.quote(str(run_experiment_path))} '
        f'--config {shlex.quote(str(config))}'
    )
    COMMANDS_TO_RUN.append(cmd)


# --- 并发与显卡分配设置 ---
# 显卡不再由配置文件里的 gpus 控制，pool_run.py 会自动检测可用显卡。
# 并发只由 MAX_WORKERS 控制：一次同时运行多少个任务；
# 每个任务启动时按轮询（round-robin）方式轮流取一张显卡，实现各卡均分。
MAX_WORKERS = 32

# 轮询分配显卡用的全局计数器（任务启动时按顺序轮流取下一张卡）
_GPU_ROUND_ROBIN = itertools.count()
_GPU_LOCK = threading.Lock()

def format_duration(seconds: float) -> str:
    """把秒数格式化成更容易读的时长。"""
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"

def detect_gpus() -> list:
    """自动检测当前机器可用的显卡编号列表（遵循 CUDA_VISIBLE_DEVICES 的限制）。

    返回:
        list[int]: 可用显卡编号列表，例如 [0, 1, 2, 3]。
    """
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10, check=False
        )
        gpus = [int(line.strip()) for line in out.stdout.splitlines() if line.strip()]
        if gpus:
            return gpus
    except Exception:
        pass
    # nvidia-smi 不可用时，退回到 CUDA_VISIBLE_DEVICES，最后默认单卡
    cvd = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    ids = [int(d) for d in cvd.split(',') if d.strip()]
    return ids if ids else [0]

def get_next_gpu(gpus: list) -> int:
    """轮询取卡：任务启动时按顺序轮流取下一张可用显卡，实现各卡动态均分。"""
    with _GPU_LOCK:
        return gpus[next(_GPU_ROUND_ROBIN) % len(gpus)]

def run_command(command: str, command_id: int, gpus: list):
    """
    执行单个 shell 命令并返回结果。

    参数:
    - command (str): 要执行的命令字符串。
    - command_id (int): 命令的唯一标识符，用于跟踪。
    - gpus (list[int]): 可用显卡编号列表，任务启动时轮询取卡。

    返回:
    - dict: 包含命令执行结果的字典。
    """
    gpu_id = get_next_gpu(gpus)  # 轮询取一张显卡
    print(f"[线程池] 任务 {command_id} 已开始执行 (GPU {gpu_id}): {command}")

    start_time = time.time()

    try:
        # 通过环境变量把任务固定到分配好的显卡上，忽略配置文件里的 gpus 设置
        task_env = dict(os.environ)
        task_env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        # 使用 subprocess.run 来执行命令
        # - shell=True: 允许我们像在终端一样执行复杂的命令字符串。
        #   注意：如果命令来自不可信的输入，这会带来安全风险。
        # - capture_output=True: 捕获 stdout 和 stderr。
        # - text=True: 将 stdout 和 stderr 解码为字符串。
        # - check=False: 如果命令返回非零退出码，不抛出异常。我们手动检查。
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=False,
            env=task_env,
            cwd=str(script_dir),
        )

        end_time = time.time()

        return {
            "id": command_id,
            "command": command,
            "gpu": gpu_id,
            "return_code": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "duration": end_time - start_time,
            "status": "成功" if result.returncode == 0 else "失败"
        }
    except Exception as e:
        end_time = time.time()
        return {
            "id": command_id,
            "command": command,
            "gpu": gpu_id,
            "return_code": -1,
            "stdout": "",
            "stderr": f"执行时发生 Python 异常: {e}",
            "duration": end_time - start_time,
            "status": "异常"
        }

def main():
    """
    主函数，创建线程池并执行所有命令。
    自动检测可用显卡，并按任务序号轮转（round-robin）平均分配到每一张卡。
    """
    total_commands = len(COMMANDS_TO_RUN)
    gpus = detect_gpus()
    num_gpus = len(gpus)

    print(f"检测到 {num_gpus} 张可用显卡: {gpus}")
    print(f"准备执行 {total_commands} 个命令，同时最多跑 {MAX_WORKERS} 个任务，"
          f"任务启动时按轮询方式均分到各张显卡。")
    print("-" * 40)

    if total_commands == 0:
        print("没有找到需要执行的命令。")
        return

    start_total_time = time.time()

    results = []
    failed_results = []
    completed_count = 0
    success_count = 0
    failure_count = 0

    # 使用 with 语句创建并管理线程池
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 使用 aenumerate 和 submit 来提交任务，这样我们可以跟踪每个任务
        future_to_command = {
            executor.submit(run_command, cmd, i, gpus): (i, cmd)
            for i, cmd in enumerate(COMMANDS_TO_RUN, 1)
        }

        # 使用 as_completed 来获取已完成任务的结果，哪个先完成就先处理哪个
        for future in as_completed(future_to_command):
            command_id, command_str = future_to_command[future]
            try:
                result_data = future.result()
                results.append(result_data)
                completed_count += 1
                if result_data["return_code"] == 0:
                    success_count += 1
                else:
                    failure_count += 1
                    failed_results.append(result_data)

                remaining_count = total_commands - completed_count
                elapsed = time.time() - start_total_time
                print(
                    f"[进度] 总数: {total_commands} | "
                    f"已跑完: {completed_count} | "
                    f"没跑完: {remaining_count} | "
                    f"成功: {success_count} | "
                    f"失败: {failure_count} | "
                    f"已用时: {format_duration(elapsed)}"
                )
                print(
                    f"[线程池] 任务 {command_id} (GPU {result_data['gpu']}) 已完成。"
                    f"状态: {result_data['status']}，耗时: {format_duration(result_data['duration'])}"
                )
            except Exception as exc:
                completed_count += 1
                failure_count += 1
                failed_result = {
                    "id": command_id,
                    "command": command_str,
                    "return_code": -1,
                    "stdout": "",
                    "stderr": f"future.result() 异常: {exc}",
                    "duration": 0,
                    "status": "异常",
                }
                results.append(failed_result)
                failed_results.append(failed_result)
                remaining_count = total_commands - completed_count
                elapsed = time.time() - start_total_time
                print(
                    f"[进度] 总数: {total_commands} | "
                    f"已跑完: {completed_count} | "
                    f"没跑完: {remaining_count} | "
                    f"成功: {success_count} | "
                    f"失败: {failure_count} | "
                    f"已用时: {format_duration(elapsed)}"
                )
                print(f"任务 {command_id} ({command_str[:30]}...) 执行时产生异常: {exc}")

    end_total_time = time.time()
    
    print("-" * 40)
    print("所有任务执行完毕。结果摘要：")
    
    # 对结果按 ID 排序，以便清晰地查看
    results.sort(key=lambda x: x['id'])
    
    print(f"总数: {total_commands}")
    print(f"成功: {success_count}")
    print(f"失败: {failure_count}")
    print(f"总耗时: {format_duration(end_total_time - start_total_time)}")

    print("-" * 40)
    if failed_results:
        failed_results.sort(key=lambda x: x["id"])
        print("失败实验列表：")
        for res in failed_results:
            print(
                f"\n--- 失败 [ID: {res['id']}] ---\n"
                f"命令: {res['command']}\n"
                f"状态: {res['status']} (返回码: {res['return_code']})\n"
                f"耗时: {format_duration(res['duration'])}\n"
                f"stderr:\n{res['stderr'] or '无'}"
            )
    else:
        print("失败实验列表：无")

if __name__ == "__main__":
    main()
