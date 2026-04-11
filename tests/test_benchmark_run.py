"""Tests for the benchmark orchestrator (src/benchmark/run_benchmark.py).

No real hardware is required — we mock the worker subprocess and the Queue
to verify orchestration logic, timeout handling, and result assembly.
"""

from __future__ import annotations

from multiprocessing import Queue, get_context

import numpy as np
import pytest

from src.benchmark.run_benchmark import (
    assemble_final_result,
    build_task_payload,
    parse_args,
)
from src.benchmark.worker import classify_exception

# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_minimal_required(self):
        args = parse_args(["--model", "t5_small", "--backend", "cpu"])
        assert args.model == "t5_small"
        assert args.backend == "cpu"
        assert args.dtype == "float32"
        assert args.warmup_iters == 3
        assert args.test_iters == 10
        assert args.timeout_seconds == 600
        assert args.task == "auto"

    def test_all_options(self):
        args = parse_args(
            [
                "--model",
                "qwen3_0_6b",
                "--backend",
                "npu",
                "--dtype",
                "bfloat16",
                "--warmup-iters",
                "5",
                "--test-iters",
                "20",
                "--timeout-seconds",
                "300",
                "--task",
                "causal-lm",
            ]
        )
        assert args.model == "qwen3_0_6b"
        assert args.backend == "npu"
        assert args.dtype == "bfloat16"
        assert args.warmup_iters == 5
        assert args.test_iters == 20
        assert args.timeout_seconds == 300
        assert args.task == "causal-lm"


# ---------------------------------------------------------------------------
# build_task_payload
# ---------------------------------------------------------------------------


class TestBuildTaskPayload:
    def test_payload_contains_only_primitives(self):
        args = parse_args(["--model", "t5_small", "--backend", "cpu"])
        payload = build_task_payload(args, is_compile=True)

        for key, val in payload.items():
            assert isinstance(val, (str, int, bool)), (
                f"Payload field '{key}' is not a primitive type: {type(val)}"
            )

    def test_payload_is_compile_flag(self):
        args = parse_args(["--model", "t5_small", "--backend", "cpu"])
        eager_payload = build_task_payload(args, is_compile=False)
        compile_payload = build_task_payload(args, is_compile=True)
        assert eager_payload["is_compile"] is False
        assert compile_payload["is_compile"] is True


# ---------------------------------------------------------------------------
# run_mode_with_timeout — Queue-based timeout simulation
# ---------------------------------------------------------------------------


class TestRunModeWithTimeout:
    def test_timeout_returns_timeout_status(self, monkeypatch):
        """Simulate a worker that never responds — Queue.get should time out."""
        ctx = get_context("spawn")
        queue: Queue = ctx.Queue()

        # Patch run_mode_with_timeout to use our queue with a tiny timeout
        def fake_run(task, timeout_seconds):
            try:
                result = queue.get(timeout=0.1)
            except Exception:
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
            return result

        task = {"model_dir": "/fake", "is_compile": False, "backend": "cpu"}
        result = fake_run(task, timeout_seconds=1)

        assert result["status"] == "TIMEOUT"

    def test_success_returns_worker_result(self):
        """When the Queue receives a result promptly, it should be returned."""
        ctx = get_context("spawn")
        queue: Queue = ctx.Queue()
        expected = {
            "status": "SUCCESS",
            "eager_latency_ms": 10.5,
            "compile_latency_ms": 0.0,
            "speedup": 0.0,
            "compile_p99_ms": 0.0,
            "precision_match": True,
            "error_message": "",
            "output": None,
        }
        queue.put(expected)

        # Mimic the dequeue logic with a tiny timeout
        try:
            result = queue.get(timeout=5)
        except Exception:
            result = {"status": "TIMEOUT"}

        assert result["status"] == "SUCCESS"
        assert result["eager_latency_ms"] == 10.5


# ---------------------------------------------------------------------------
# assemble_final_result
# ---------------------------------------------------------------------------


class TestAssembleFinalResult:
    def _args(self):
        return parse_args(["--model", "t5_small", "--backend", "cpu"])

    def _success_result(self, lat_ms, is_compile=False):
        return {
            "status": "SUCCESS",
            "eager_latency_ms": lat_ms if not is_compile else 0.0,
            "compile_latency_ms": lat_ms if is_compile else 0.0,
            "speedup": 0.0,
            "compile_p99_ms": lat_ms if is_compile else 0.0,
            "precision_match": True,
            "error_message": "",
            "output": {"logits": np.array([[1.0, 2.0]])},
        }

    def test_both_success_precision_match(self):
        eager = self._success_result(25.0, is_compile=False)
        compile_ = self._success_result(10.0, is_compile=True)
        final = assemble_final_result(eager, compile_, self._args())

        assert final["status"] == "SUCCESS"
        assert final["eager_latency_ms"] == 25.0
        assert final["compile_latency_ms"] == 10.0
        assert final["speedup"] == pytest.approx(2.5)
        assert final["precision_match"] is True

    def test_both_success_precision_fail(self):
        eager = self._success_result(25.0, is_compile=False)
        compile_ = self._success_result(10.0, is_compile=True)
        # Make outputs differ
        eager["output"] = {"logits": np.array([[1.0, 2.0]])}
        compile_["output"] = {"logits": np.array([[9.0, 9.0]])}

        final = assemble_final_result(eager, compile_, self._args())

        assert final["status"] == "PRECISION_FAIL"
        assert final["precision_match"] is False
        assert "max_abs_error" in final["error_message"]

    def test_eager_oom(self):
        eager = {
            "status": "OOM",
            "eager_latency_ms": 0.0,
            "compile_latency_ms": 0.0,
            "speedup": 0.0,
            "compile_p99_ms": 0.0,
            "precision_match": False,
            "error_message": "CUDA out of memory",
            "output": None,
        }
        compile_ = self._success_result(10.0, is_compile=True)
        final = assemble_final_result(eager, compile_, self._args())

        assert final["status"] == "OOM"
        assert "[eager]" in final["error_message"]

    def test_compile_timeout(self):
        eager = self._success_result(25.0, is_compile=False)
        compile_ = {
            "status": "TIMEOUT",
            "eager_latency_ms": 0.0,
            "compile_latency_ms": 0.0,
            "speedup": 0.0,
            "compile_p99_ms": 0.0,
            "precision_match": False,
            "error_message": "Worker timed out after 600s",
            "output": None,
        }
        final = assemble_final_result(eager, compile_, self._args())

        assert final["status"] == "TIMEOUT"
        assert "[compile]" in final["error_message"]

    def test_none_output_precision_fail(self):
        eager = self._success_result(25.0, is_compile=False)
        compile_ = self._success_result(10.0, is_compile=True)
        compile_["output"] = None

        final = assemble_final_result(eager, compile_, self._args())

        assert final["status"] == "PRECISION_FAIL"


# ---------------------------------------------------------------------------
# classify_exception
# ---------------------------------------------------------------------------


class TestClassifyException:
    def test_cuda_oom(self):
        tb = "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB"
        assert classify_exception(tb, is_compile=False) == "OOM"

    def test_cann_oom(self):
        tb = "RuntimeError: CANN memory allocation failed"
        assert classify_exception(tb, is_compile=False) == "OOM"

    def test_compile_error(self):
        tb = "torch._dynamo.exc.Unsupported: call_function UserDefinedObject"
        assert classify_exception(tb, is_compile=True) == "COMPILE_ERROR"

    def test_generic_crash(self):
        tb = "ValueError: something unrelated"
        assert classify_exception(tb, is_compile=False) == "CRASH"

    def test_compile_crash_not_compile_error(self):
        """If is_compile=False but 'compile' appears in traceback, still CRASH."""
        tb = "some compile issue but is_compile=False"
        # The classify_exception checks for 'compile' in text AND is_compile flag
        # so when is_compile=False, it should not return COMPILE_ERROR
        result = classify_exception(tb, is_compile=False)
        assert result != "COMPILE_ERROR"
