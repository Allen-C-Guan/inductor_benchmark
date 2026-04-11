"""Benchmark orchestrator — CLI entry point.

Master process only: parses args, spawns workers serially with a watchdog,
and assembles the final wide-table result.
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Queue, get_context
from pathlib import Path
from typing import Any

from .compare import compare_outputs
from .worker import worker_main


def project_root() -> Path:
    """Return repository root, matching the pattern in download_hf_model.py."""
    return Path(__file__).resolve().parents[2]


def resolve_model_dir(alias: str) -> Path:
    """Return the absolute path to model/<alias>/."""
    return project_root() / "model" / alias


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference benchmark on a local HF model snapshot.",
    )
    parser.add_argument("--model", required=True, help="Model alias (directory name under model/)")
    parser.add_argument("--backend", required=True, choices=["cpu", "npu"], help="Device backend")
    parser.add_argument("--dtype", default="float32", help="Data type (default: float32)")
    parser.add_argument(
        "--warmup-iters", type=int, default=3, help="Warmup iterations (default: 3)"
    )
    parser.add_argument(
        "--test-iters", type=int, default=10, help="Benchmark iterations (default: 10)"
    )
    parser.add_argument(
        "--timeout-seconds", type=int, default=600, help="Per-mode watchdog timeout in seconds"
    )
    parser.add_argument(
        "--task",
        default="auto",
        choices=["auto", "causal-lm", "seq2seq-lm", "masked-lm"],
        help="Task type hint (default: auto)",
    )
    return parser.parse_args(argv)


def build_task_payload(args: argparse.Namespace, is_compile: bool) -> dict[str, Any]:
    """Construct a primitive-only task dict for the worker."""
    return {
        "model_dir": str(resolve_model_dir(args.model)),
        "is_compile": is_compile,
        "backend": args.backend,
        "dtype": args.dtype,
        "warmup_iters": args.warmup_iters,
        "test_iters": args.test_iters,
        "task": args.task,
    }


def run_mode_with_timeout(task: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
    """Spawn a single worker subprocess and wait with a watchdog.

    Returns the worker result dict, or a TIMEOUT dict on deadlock.
    """
    ctx = get_context("spawn")
    result_queue: Queue = ctx.Queue()

    proc = ctx.Process(target=worker_main, args=(task, result_queue), daemon=True)
    proc.start()

    try:
        result = result_queue.get(timeout=timeout_seconds)   # 这里上放置业务卡死
    except Exception:
        # Timeout or queue error — kill the worker 这里是防止垃圾回收卡死
        proc.terminate()  # proc.terminate() 主进程向子进程发送 SIGTERM，让他自行清理内存
        proc.join(timeout=300) # 如果十秒了，子进程还没有自行了断完成
        if proc.is_alive():
            proc.kill()     # 那就直接杀掉子进程
            proc.join(timeout=180)
        return {
            "status": "TIMEOUT",
            "eager_latency_ms": 0.0,
            "compile_latency_ms": 0.0,
            "speedup": 0.0,
            "compile_p99_ms": 0.0,
            "precision_match": False,
            "error_message": f"Worker timed out after {timeout_seconds}s",
            "output": None,
        }

    proc.join(timeout=10)
    return result


def assemble_final_result(
    eager_result: dict[str, Any],
    compile_result: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Combine eager + compile results into the canonical wide-table schema."""
    # Determine overall status
    for key in ("eager_result", "compile_result"):
        res = eager_result if key == "eager_result" else compile_result
        if res["status"] not in ("SUCCESS",):
            # Propagate the first non-success status
            mode_label = "eager" if key == "eager_result" else "compile"
            error_msg = res.get("error_message", "")
            return {
                "model_id": args.model,
                "dtype": args.dtype,
                "status": res["status"],
                "eager_latency_ms": eager_result.get("eager_latency_ms", 0.0),
                "compile_latency_ms": compile_result.get("compile_latency_ms", 0.0),
                "speedup": 0.0,
                "compile_p99_ms": compile_result.get("compile_p99_ms", 0.0),
                "precision_match": False,
                "error_message": f"[{mode_label}] {error_msg}",
            }

    # Both SUCCESS — run precision comparison
    eager_output = eager_result.get("output")
    compile_output = compile_result.get("output")

    if eager_output is None or compile_output is None:
        return {
            "model_id": args.model,
            "dtype": args.dtype,
            "status": "PRECISION_FAIL",
            "eager_latency_ms": eager_result.get("eager_latency_ms", 0.0),
            "compile_latency_ms": compile_result.get("compile_latency_ms", 0.0),
            "speedup": 0.0,
            "compile_p99_ms": compile_result.get("compile_p99_ms", 0.0),
            "precision_match": False,
            "error_message": "One or both outputs are None, cannot compare precision",
        }

    compare = compare_outputs(eager_output, compile_output, args.dtype)

    eager_lat = eager_result["eager_latency_ms"]
    compile_lat = compile_result["compile_latency_ms"]
    speedup = eager_lat / compile_lat if compile_lat > 0 else 0.0

    status = "SUCCESS" if compare.precision_match else "PRECISION_FAIL"

    return {
        "model_id": args.model,
        "dtype": args.dtype,
        "status": status,
        "eager_latency_ms": round(eager_lat, 4),
        "compile_latency_ms": round(compile_lat, 4),
        "speedup": round(speedup, 4),
        "compile_p99_ms": round(compile_result.get("compile_p99_ms", 0.0), 4),
        "precision_match": compare.precision_match,
        "error_message": compare.error_message if not compare.precision_match else "",
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    model_path = resolve_model_dir(args.model)
    if not model_path.exists():
        print(f"Error: model directory {model_path} does not exist.", file=sys.stderr)
        return 1

    print(f"Benchmarking model={args.model} backend={args.backend} dtype={args.dtype}")

    # 1. Eager mode
    print("  Running eager mode...")
    eager_task = build_task_payload(args, is_compile=False)
    eager_result = run_mode_with_timeout(eager_task, args.timeout_seconds)
    print(f"  Eager status: {eager_result['status']}")

    # 2. Compile mode
    print("  Running compile mode...")
    compile_task = build_task_payload(args, is_compile=True)
    compile_result = run_mode_with_timeout(compile_task, args.timeout_seconds)
    print(f"  Compile status: {compile_result['status']}")

    # 3. Assemble & print result
    final = assemble_final_result(eager_result, compile_result, args)
    print("\n=== Benchmark Result ===")
    print(json.dumps(final, indent=2, ensure_ascii=False))

    return 0 if final["status"] == "SUCCESS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
