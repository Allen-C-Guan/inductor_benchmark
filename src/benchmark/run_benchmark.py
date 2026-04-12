"""Benchmark orchestrator — CLI entry point.

Master process only: parses args, spawns workers serially with a watchdog,
and assembles the final wide-table result.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
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


def discover_models() -> list[str]:
    """Discover all model directories under model/."""
    model_root = project_root() / "model"
    if not model_root.exists():
        return []
    models = []
    for entry in model_root.iterdir():
        if entry.is_dir() and not entry.name.startswith("."):
            models.append(entry.name)
    return sorted(models)


def resolve_output_dir() -> Path:
    """Return the absolute path to output/ directory."""
    return project_root() / "output"


def save_results(results: list[dict[str, Any]], output_dir: Path) -> Path:
    """Save benchmark results to CSV file in output directory.

    Returns the path to the saved file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"benchmark_{timestamp}.csv"

    if not results:
        # Write empty file with headers
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "model_id",
                    "dtype",
                    "status",
                    "eager_latency_ms",
                    "compile_latency_ms",
                    "speedup",
                    "compile_p99_ms",
                    "precision_match",
                    "error_message",
                ]
            )
        return csv_path

    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    return csv_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference benchmark on a local HF model snapshot.",
    )
    parser.add_argument(
        "--model", help="Model alias (directory name under model/). Runs all if not specified."
    )
    parser.add_argument("--all", action="store_true", help="Benchmark all models under model/")
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


def build_task_payload(
    model_alias: str, args: argparse.Namespace, is_compile: bool
) -> dict[str, Any]:
    """Construct a primitive-only task dict for the worker."""
    return {
        "model_dir": str(resolve_model_dir(model_alias)),
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
        result = result_queue.get(timeout=timeout_seconds)  # 这里上放置业务卡死
    except Exception:
        # Timeout or queue error — kill the worker 这里是防止垃圾回收卡死
        proc.terminate()  # proc.terminate() 主进程向子进程发送 SIGTERM，让他自行清理内存
        proc.join(timeout=300)  # 如果十秒了，子进程还没有自行了断完成
        if proc.is_alive():
            proc.kill()  # 那就直接杀掉子进程
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
    model_alias: str,
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
                "model_id": model_alias,
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
            "model_id": model_alias,
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
        "model_id": model_alias,
        "dtype": args.dtype,
        "status": status,
        "eager_latency_ms": round(eager_lat, 4),
        "compile_latency_ms": round(compile_lat, 4),
        "speedup": round(speedup, 4),
        "compile_p99_ms": round(compile_result.get("compile_p99_ms", 0.0), 4),
        "precision_match": compare.precision_match,
        "error_message": compare.error_message if not compare.precision_match else "",
    }


def run_single_model(model_alias: str, args: argparse.Namespace) -> dict[str, Any]:
    """Run benchmark for a single model and return the result."""
    model_path = resolve_model_dir(model_alias)
    if not model_path.exists():
        return {
            "model_id": model_alias,
            "dtype": args.dtype,
            "status": "CRASH",
            "eager_latency_ms": 0.0,
            "compile_latency_ms": 0.0,
            "speedup": 0.0,
            "compile_p99_ms": 0.0,
            "precision_match": False,
            "error_message": f"Model directory {model_path} does not exist",
        }

    print(f"Benchmarking model={model_alias} backend={args.backend} dtype={args.dtype}")

    # 1. Eager mode
    print("  Running eager mode...")
    eager_task = build_task_payload(model_alias, args, is_compile=False)
    eager_result = run_mode_with_timeout(eager_task, args.timeout_seconds)
    print(f"  Eager status: {eager_result['status']}")

    # 2. Compile mode
    print("  Running compile mode...")
    compile_task = build_task_payload(model_alias, args, is_compile=True)
    compile_result = run_mode_with_timeout(compile_task, args.timeout_seconds)
    print(f"  Compile status: {compile_result['status']}")

    # 3. Assemble result
    return assemble_final_result(model_alias, eager_result, compile_result, args)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Determine which models to benchmark
    if args.model:
        models = [args.model]
    else:
        models = discover_models()

    if not models:
        print("Error: No models found to benchmark.", file=sys.stderr)
        return 1

    print(f"Found {len(models)} model(s) to benchmark: {models}")

    results: list[dict[str, Any]] = []
    for model_alias in models:
        result = run_single_model(model_alias, args)
        results.append(result)
        print(f"\n=== {model_alias} Result ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    # Save results to output directory
    output_dir = resolve_output_dir()
    csv_path = save_results(results, output_dir)
    print(f"\n=== Results saved to {csv_path} ===")

    # Return non-zero if any model failed
    success_count = sum(1 for r in results if r["status"] == "SUCCESS")
    return 0 if success_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
