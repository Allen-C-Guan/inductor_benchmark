# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

This repository currently centers on downloading and storing Hugging Face model snapshots for local benchmarking.

## Common commands

- Install the required downloader dependency:
  - `python -m pip install huggingface_hub`
- Download all configured models:
  - `python src/model_download/download_hf_model.py`

## Test/lint/build status

- There is currently no project-level test suite, linter config, or build system checked into this repository.
- There is no supported “single test” command yet because no repo test files/config are defined.

## High-level architecture

- `src/model_download/download_hf_model.py` is the orchestration entrypoint.
  - `MODEL_REGISTRY` maps local model aliases to Hugging Face repo IDs.
  - `download_all_models()` iterates that registry and snapshots each model into `model/<alias>/`.
  - Downloads use `huggingface_hub.snapshot_download(...)` with `ignore_patterns` to skip non-benchmark artifacts (for example `*.h5`, `*.gguf`, `onnx/`).
- `model/` is the local artifact store for downloaded model contents (e.g. `model/t5_small/`).
- `.gitignore` intentionally excludes virtualenvs, logs, and large model artifacts (`*.pt`, `*.pth`, `*.bin`, `*.safetensors`) from version control.

## Operational notes

- The downloader expects repository-root-relative layout and resolves root via `Path(__file__).resolve().parents[2]`.
- If `huggingface_hub` is missing, the script exits with an explicit install hint.

## Benchmark script skill (from `benchmark notes.md`)

When writing benchmark scripts in this repo, follow this protocol strictly:

1. **Master-worker stateless design**
   - Main process is orchestrator only (task matrix, watchdog timeout, result aggregation).
   - Never initialize CUDA/CANN runtime or deep-learning global state in main process.
   - Worker subprocess must use `spawn` start method.

2. **Crash containment and scheduling**
   - Each benchmark task runs in a short-lived dedicated subprocess.
   - Main process must enforce watchdog via `Queue.get(timeout=...)`.
   - On timeout/deadlock, classify as `TIMEOUT_CRASH` and reclaim worker.
   - Run eager and compile benchmarks in strict serial order (no bus/bandwidth contention).

3. **Worker lifecycle (7 required steps)**
   - Import torch inside worker first.
   - Load model from disk/HF, call `tie_weights()`, materialize any remaining meta tensors.
   - Move to device and `eval()`.
   - Generate fresh inputs in worker (no tensor transfer from parent, no input reuse).
   - Warmup (>=3 forwards) + synchronized timing using device sync around `perf_counter`.
   - Move outputs to CPU and serialize as NumPy payload before IPC.
   - Push dict result to Queue and exit process immediately.

4. **Precision verifier requirements**
   - Recursively compare nested `ModelOutput` / dict / tuple / list structures.
   - Int/bool tensors require exact match.
   - Float32 uses `rtol=1e-4, atol=1e-4`.
   - BFloat16 uses `rtol=1e-2, atol=1e-2`.
   - Treat aligned NaN/Inf as equal (`equal_nan=True`).
   - On mismatch, report max absolute error, max relative error, and failing tensor path.

5. **IPC and schemas**
   - Parent→worker payload must contain only primitive fields (str/int/bool).
   - Worker→parent result must be deterministic wide-table fields, including:
     - `status` in `{SUCCESS, OOM, TIMEOUT, PRECISION_FAIL, CRASH}`
     - `eager_latency_ms`, `compile_latency_ms`, `speedup=eager/compile`, `compile_p99_ms`, `precision_match`, `error_message`.

6. **Failure translation and cleanup**
   - Parse traceback text to classify OOM (`CUDA out of memory` / `CANN memory allocation failed`) as `OOM`.
   - Distinguish compiler failures as `COMPILE_ERROR` (for compiler-team escalation).
   - In `finally`, call cross-device cache cleanup (`empty_cache`) before process teardown.
