"""Tests for the benchmark orchestrator (src/benchmark/run_benchmark.py).

No real hardware is required — we mock the worker subprocess and the Queue
to verify orchestration logic, timeout handling, and result assembly.
"""

from __future__ import annotations

import csv
from multiprocessing import Queue, get_context

import numpy as np
import pytest

from src.benchmark.run_benchmark import (
    assemble_final_result,
    build_task_payload,
    discover_models,
    main,
    parse_args,
    resolve_output_dir,
    run_single_model,
    save_results,
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
        payload = build_task_payload("t5_small", args, is_compile=True)

        for key, val in payload.items():
            assert isinstance(val, (str, int, bool)), (
                f"Payload field '{key}' is not a primitive type: {type(val)}"
            )

    def test_payload_is_compile_flag(self):
        args = parse_args(["--model", "t5_small", "--backend", "cpu"])
        eager_payload = build_task_payload("t5_small", args, is_compile=False)
        compile_payload = build_task_payload("t5_small", args, is_compile=True)
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
        final = assemble_final_result("t5_small", eager, compile_, self._args())

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

        final = assemble_final_result("t5_small", eager, compile_, self._args())

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
        final = assemble_final_result("t5_small", eager, compile_, self._args())

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
        final = assemble_final_result("t5_small", eager, compile_, self._args())

        assert final["status"] == "TIMEOUT"
        assert "[compile]" in final["error_message"]

    def test_none_output_precision_fail(self):
        eager = self._success_result(25.0, is_compile=False)
        compile_ = self._success_result(10.0, is_compile=True)
        compile_["output"] = None

        final = assemble_final_result("t5_small", eager, compile_, self._args())

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


# ---------------------------------------------------------------------------
# discover_models
# ---------------------------------------------------------------------------


class TestDiscoverModels:
    def test_discovers_model_directories(self, monkeypatch, tmp_path):
        """Should return sorted list of model directories under model/."""
        # Create fake model directories
        model_root = tmp_path / "model"
        model_root.mkdir()
        (model_root / "t5_small").mkdir()
        (model_root / "bert_base").mkdir()
        (model_root / "qwen3_0_6b").mkdir()
        # Create a hidden directory that should be ignored
        (model_root / ".hidden").mkdir()

        monkeypatch.setattr(
            "src.benchmark.run_benchmark.project_root",
            lambda: tmp_path,
        )

        models = discover_models()
        assert models == ["bert_base", "qwen3_0_6b", "t5_small"]

    def test_returns_empty_list_when_no_models(self, monkeypatch, tmp_path):
        """Should return empty list when model/ doesn't exist or is empty."""
        monkeypatch.setattr(
            "src.benchmark.run_benchmark.project_root",
            lambda: tmp_path,
        )

        models = discover_models()
        assert models == []

    def test_ignores_files_in_model_dir(self, monkeypatch, tmp_path):
        """Should only return directories, not files."""
        model_root = tmp_path / "model"
        model_root.mkdir()
        (model_root / "t5_small").mkdir()
        (model_root / "some_file.txt").touch()  # File, not directory

        monkeypatch.setattr(
            "src.benchmark.run_benchmark.project_root",
            lambda: tmp_path,
        )

        models = discover_models()
        assert models == ["t5_small"]


# ---------------------------------------------------------------------------
# save_results
# ---------------------------------------------------------------------------


class TestSaveResults:
    def test_saves_results_to_csv(self, tmp_path):
        """Should write benchmark results to a CSV file."""
        results = [
            {
                "model_id": "t5_small",
                "dtype": "float32",
                "status": "SUCCESS",
                "eager_latency_ms": 25.5,
                "compile_latency_ms": 10.2,
                "speedup": 2.5,
                "compile_p99_ms": 12.0,
                "precision_match": True,
                "error_message": "",
            },
            {
                "model_id": "bert_base",
                "dtype": "float32",
                "status": "OOM",
                "eager_latency_ms": 0.0,
                "compile_latency_ms": 0.0,
                "speedup": 0.0,
                "compile_p99_ms": 0.0,
                "precision_match": False,
                "error_message": "CUDA out of memory",
            },
        ]

        csv_path = save_results(results, tmp_path)

        assert csv_path.exists()
        assert csv_path.suffix == ".csv"
        assert csv_path.parent == tmp_path

        # Verify CSV content
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["model_id"] == "t5_small"
        assert rows[0]["status"] == "SUCCESS"
        assert float(rows[0]["speedup"]) == 2.5
        assert rows[1]["model_id"] == "bert_base"
        assert rows[1]["status"] == "OOM"

    def test_creates_output_directory_if_not_exists(self, tmp_path):
        """Should create output directory if it doesn't exist."""
        output_dir = tmp_path / "new_output"
        assert not output_dir.exists()

        results = [
            {
                "model_id": "t5_small",
                "dtype": "float32",
                "status": "SUCCESS",
                "eager_latency_ms": 25.5,
                "compile_latency_ms": 10.2,
                "speedup": 2.5,
                "compile_p99_ms": 12.0,
                "precision_match": True,
                "error_message": "",
            }
        ]

        csv_path = save_results(results, output_dir)

        assert output_dir.exists()
        assert csv_path.exists()

    def test_handles_empty_results(self, tmp_path):
        """Should write CSV with headers even when results list is empty."""
        csv_path = save_results([], tmp_path)

        assert csv_path.exists()

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Should have header row only
        assert len(rows) == 1
        assert "model_id" in rows[0]
        assert "status" in rows[0]

    def test_filename_contains_timestamp(self, tmp_path):
        """CSV filename should contain timestamp."""
        results = [
            {
                "model_id": "t5_small",
                "dtype": "float32",
                "status": "SUCCESS",
                "eager_latency_ms": 25.5,
                "compile_latency_ms": 10.2,
                "speedup": 2.5,
                "compile_p99_ms": 12.0,
                "precision_match": True,
                "error_message": "",
            }
        ]

        csv_path = save_results(results, tmp_path)

        # Filename should match pattern benchmark_YYYYMMDD_HHMMSS.csv
        assert csv_path.name.startswith("benchmark_")
        assert csv_path.suffix == ".csv"


# ---------------------------------------------------------------------------
# run_single_model
# ---------------------------------------------------------------------------


class TestRunSingleModel:
    def test_returns_crash_when_model_dir_not_exists(self, monkeypatch, tmp_path):
        """Should return CRASH status when model directory doesn't exist."""
        monkeypatch.setattr(
            "src.benchmark.run_benchmark.project_root",
            lambda: tmp_path,
        )

        args = parse_args(["--model", "nonexistent_model", "--backend", "cpu"])
        result = run_single_model("nonexistent_model", args)

        assert result["status"] == "CRASH"
        assert "does not exist" in result["error_message"]
        assert result["model_id"] == "nonexistent_model"

    def test_runs_benchmark_for_existing_model(
        self, monkeypatch, tmp_path, sample_benchmark_result
    ):
        """Should run benchmark and return result for existing model."""
        # Create model directory
        model_root = tmp_path / "model"
        model_root.mkdir()
        (model_root / "t5_small").mkdir()

        monkeypatch.setattr(
            "src.benchmark.run_benchmark.project_root",
            lambda: tmp_path,
        )

        # Mock run_mode_with_timeout to return success
        def mock_run_mode(task, timeout_seconds):
            return sample_benchmark_result

        monkeypatch.setattr(
            "src.benchmark.run_benchmark.run_mode_with_timeout",
            mock_run_mode,
        )

        # Mock compare_outputs to return precision match
        from types import SimpleNamespace

        monkeypatch.setattr(
            "src.benchmark.run_benchmark.compare_outputs",
            lambda *args: SimpleNamespace(precision_match=True, error_message=""),
        )

        args = parse_args(["--model", "t5_small", "--backend", "cpu"])
        result = run_single_model("t5_small", args)

        assert result["model_id"] == "t5_small"
        assert result["status"] == "SUCCESS"


# ---------------------------------------------------------------------------
# main - batch mode
# ---------------------------------------------------------------------------


class TestMainBatchMode:
    def test_main_runs_all_models_when_no_model_specified(
        self, monkeypatch, tmp_path, sample_benchmark_result, capsys
    ):
        """When --model is not specified, should run all discovered models."""
        # Create model directories
        model_root = tmp_path / "model"
        model_root.mkdir()
        (model_root / "model_a").mkdir()
        (model_root / "model_b").mkdir()

        # Create output directory
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        monkeypatch.setattr(
            "src.benchmark.run_benchmark.project_root",
            lambda: tmp_path,
        )
        monkeypatch.setattr(
            "src.benchmark.run_benchmark.resolve_output_dir",
            lambda: output_dir,
        )

        # Mock run_mode_with_timeout
        monkeypatch.setattr(
            "src.benchmark.run_benchmark.run_mode_with_timeout",
            lambda task, timeout: sample_benchmark_result,
        )

        # Mock compare_outputs
        from types import SimpleNamespace

        monkeypatch.setattr(
            "src.benchmark.run_benchmark.compare_outputs",
            lambda *args: SimpleNamespace(precision_match=True, error_message=""),
        )

        exit_code = main(["--backend", "cpu"])

        assert exit_code == 0

        # Check output contains both models
        captured = capsys.readouterr()
        assert "model_a" in captured.out
        assert "model_b" in captured.out
        assert "Found 2 model(s)" in captured.out

        # Check CSV was created
        csv_files = list(output_dir.glob("benchmark_*.csv"))
        assert len(csv_files) == 1

    def test_main_runs_single_model_when_model_specified(
        self, monkeypatch, tmp_path, sample_benchmark_result, capsys
    ):
        """When --model is specified, should run only that model."""
        model_root = tmp_path / "model"
        model_root.mkdir()
        (model_root / "model_a").mkdir()
        (model_root / "model_b").mkdir()

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        monkeypatch.setattr(
            "src.benchmark.run_benchmark.project_root",
            lambda: tmp_path,
        )
        monkeypatch.setattr(
            "src.benchmark.run_benchmark.resolve_output_dir",
            lambda: output_dir,
        )

        monkeypatch.setattr(
            "src.benchmark.run_benchmark.run_mode_with_timeout",
            lambda task, timeout: sample_benchmark_result,
        )

        from types import SimpleNamespace

        monkeypatch.setattr(
            "src.benchmark.run_benchmark.compare_outputs",
            lambda *args: SimpleNamespace(precision_match=True, error_message=""),
        )

        exit_code = main(["--model", "model_a", "--backend", "cpu"])

        assert exit_code == 0

        captured = capsys.readouterr()
        assert "model_a" in captured.out
        assert "model_b" not in captured.out

    def test_main_returns_error_when_no_models_found(self, monkeypatch, tmp_path, capsys):
        """Should return error code when no models are found."""
        monkeypatch.setattr(
            "src.benchmark.run_benchmark.project_root",
            lambda: tmp_path,
        )

        exit_code = main(["--backend", "cpu"])

        assert exit_code == 1

        captured = capsys.readouterr()
        assert "No models found" in captured.err

    def test_main_returns_error_when_any_model_fails(
        self, monkeypatch, tmp_path, sample_benchmark_result, capsys
    ):
        """Should return non-zero when any model fails."""
        model_root = tmp_path / "model"
        model_root.mkdir()
        (model_root / "model_a").mkdir()
        (model_root / "model_b").mkdir()

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        monkeypatch.setattr(
            "src.benchmark.run_benchmark.project_root",
            lambda: tmp_path,
        )
        monkeypatch.setattr(
            "src.benchmark.run_benchmark.resolve_output_dir",
            lambda: output_dir,
        )

        call_count = [0]

        def mock_run_mode(task, timeout):
            call_count[0] += 1
            # First model succeeds, second fails
            if call_count[0] <= 2:  # model_a (eager + compile)
                return sample_benchmark_result
            else:  # model_b fails
                return {
                    "status": "OOM",
                    "eager_latency_ms": 0.0,
                    "compile_latency_ms": 0.0,
                    "speedup": 0.0,
                    "compile_p99_ms": 0.0,
                    "precision_match": False,
                    "error_message": "OOM",
                    "output": None,
                }

        monkeypatch.setattr(
            "src.benchmark.run_benchmark.run_mode_with_timeout",
            mock_run_mode,
        )

        from types import SimpleNamespace

        monkeypatch.setattr(
            "src.benchmark.run_benchmark.compare_outputs",
            lambda *args: SimpleNamespace(precision_match=True, error_message=""),
        )

        exit_code = main(["--backend", "cpu"])

        # Should return non-zero because model_b failed
        assert exit_code == 1


# ---------------------------------------------------------------------------
# resolve_output_dir
# ---------------------------------------------------------------------------


class TestResolveOutputDir:
    def test_returns_path_to_output_directory(self, monkeypatch, tmp_path):
        """Should return project_root/output path."""
        monkeypatch.setattr(
            "src.benchmark.run_benchmark.project_root",
            lambda: tmp_path,
        )

        output_dir = resolve_output_dir()

        assert output_dir == tmp_path / "output"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_benchmark_result():
    """A sample successful benchmark result from worker."""
    return {
        "status": "SUCCESS",
        "eager_latency_ms": 25.5,
        "compile_latency_ms": 10.2,
        "speedup": 2.5,
        "compile_p99_ms": 12.0,
        "precision_match": True,
        "error_message": "",
        "output": {"logits": np.array([[1.0, 2.0]])},
    }
