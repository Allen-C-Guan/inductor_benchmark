"""Isolated sandbox worker for benchmark inference.

All hardware interaction (device init, model load, forward pass) happens
inside this module, which is run in a dedicated ``spawn`` subprocess.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from queue import Queue
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Task type inference
# ---------------------------------------------------------------------------


def infer_task_type(model_dir: str, task_hint: str = "auto") -> str:
    """Return one of 'causal-lm', 'seq2seq-lm', 'masked-lm'."""
    if task_hint != "auto":
        return task_hint

    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {model_dir}")

    import json

    config = json.loads(config_path.read_text())
    architectures = config.get("architectures", [])
    model_type = config.get("model_type", "").lower()

    arch_str = " ".join(architectures).lower()

    causal_types = ("gpt2", "llama", "qwen2", "qwen3", "mistral", "gemma", "phi")
    if "causallm" in arch_str or model_type in causal_types:
        return "causal-lm"
    if "seq2seq" in arch_str or "conditionalgeneration" in arch_str:
        return "seq2seq-lm"
    if model_type in ("t5", "t5withlm"):
        return "seq2seq-lm"
    if "maskedlm" in arch_str or "fillmask" in arch_str:
        return "masked-lm"
    masked_types = ("bert", "roberta", "xlm-roberta")
    if model_type in masked_types:
        return "masked-lm"

    raise ValueError(
        f"Cannot infer task type from config.json.\n"
        f"  architectures={architectures}\n"
        f"  model_type={model_type}\n"
        f"Use --task to specify manually."
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_from_disk(
    model_dir: str,
    task_type: str,
    dtype_str: str,
):
    """Load model from local HF directory and return (model, config)."""
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForMaskedLM,
        AutoModelForSeq2SeqLM,
    )

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)

    dtype_map = {
        "float32": "float32",
        "float16": "float16",
        "bfloat16": "bfloat16",
        "fp32": "float32",
        "fp16": "float16",
        "bf16": "bfloat16",
    }
    torch_dtype = getattr(__import__("torch"), dtype_map.get(dtype_str, dtype_str))

    model_cls = {
        "causal-lm": AutoModelForCausalLM,
        "seq2seq-lm": AutoModelForSeq2SeqLM,
        "masked-lm": AutoModelForMaskedLM,
    }[task_type]

    model = model_cls.from_pretrained(
        model_dir,
        config=config,
        torch_dtype=torch_dtype,
        local_files_only=True,
    )

    # Tie weights if available
    if hasattr(model, "tie_weights"):
        model.tie_weights()

    return model, config


# ---------------------------------------------------------------------------
# Input generation (synthetic, no tokenizer)
# ---------------------------------------------------------------------------


def make_inputs(config: Any, task_type: str, device: str, dtype: str):
    """Generate synthetic input tensors on *device*.

    Returns a dict of kwargs suitable for ``model(**inputs)``.
    """
    import torch

    # Heuristic: use a small sequence length to keep memory low
    seq_len = 128
    vocab_size = getattr(config, "vocab_size", 32000)
    batch_size = 1

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)

    inputs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    if task_type == "seq2seq-lm":
        decoder_seq_len = 32
        decoder_input_ids = torch.randint(
            0,
            vocab_size,
            (batch_size, decoder_seq_len),
            device=device,
        )
        inputs["decoder_input_ids"] = decoder_input_ids

    return inputs


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


def get_device_str(backend: str) -> str:
    """Return a torch device string like 'cpu', 'cuda', 'npu'."""
    if backend == "cpu":
        return "cpu"
    if backend == "npu":
        return "npu:0"
    # Unknown backend — let torch fail naturally
    return f"{backend}:0"


def synchronize_if_torch_device(device: str) -> None:
    """Call the appropriate synchronize API for the device."""
    import torch

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device.startswith("npu"):
        torch.npu.synchronize()


def cleanup_device(backend: str) -> None:
    """Release cached device memory."""
    import torch

    if backend == "cpu":
        return
    if backend == "npu":
        torch.npu.empty_cache()
    else:
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def serialize_output(value: Any) -> Any:
    """Recursively convert a model output to pure Python / NumPy types.

    Tensors are moved to CPU and converted to numpy arrays.
    """
    import torch
    from transformers.utils import ModelOutput

    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().float().numpy()
        return arr
    if isinstance(value, ModelOutput):
        return {k: serialize_output(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {k: serialize_output(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_output(v) for v in value]
    # Scalars
    if isinstance(value, (int, float, bool, str, type(None))):
        return value
    # Fallback: try numpy
    try:
        return np.array(value)
    except Exception:
        return str(value)


# ---------------------------------------------------------------------------
# Exception classification
# ---------------------------------------------------------------------------


def classify_exception(tb_text: str, is_compile: bool) -> str:
    """Classify an exception string into a canonical status label."""
    lower = tb_text.lower()
    if "cuda out of memory" in lower or "cann memory allocation failed" in lower:
        return "OOM"
    if is_compile and ("compile" in lower or "dynamo" in lower or "triton" in lower):
        return "COMPILE_ERROR"
    return "CRASH"


# ---------------------------------------------------------------------------
# Worker main
# ---------------------------------------------------------------------------


def worker_main(task: dict[str, Any], result_queue: Queue) -> None:
    """Entry point for the spawned subprocess.

    Lifecycle:
      1. import torch
      2. load model
      3. model.to(device).eval()
      4. generate inputs (fresh, inside worker)
      5. warmup + timed inference
      6. serialize outputs to CPU numpy
      7. push result dict to queue and exit
    """
    import torch  # must be first import inside worker

    model_dir: str = task["model_dir"]
    is_compile: bool = task["is_compile"]
    backend: str = task["backend"]
    dtype_str: str = task.get("dtype", "float32")
    warmup_iters: int = task.get("warmup_iters", 3)
    test_iters: int = task.get("test_iters", 10)
    task_type_hint: str = task.get("task", "auto")
    mode_label = "compile" if is_compile else "eager"

    try:
        # 1-2. Load model
        task_type = infer_task_type(model_dir, task_type_hint)
        model, config = load_model_from_disk(model_dir, task_type, dtype_str)
        print(f"    [{mode_label}] Model loaded successfully (task_type={task_type})")

        # 3. Device transfer
        device = get_device_str(backend)
        model = model.to(device).eval()

        # Materialize any remaining meta tensors
        for param in model.parameters():
            if param.device.type == "meta":
                param.data = torch.empty_like(param.data, device=device)

        # 4. Generate fresh inputs
        inputs = make_inputs(config, task_type, device, dtype_str)

        # Resolve dtype for autocast
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        dtype = dtype_map.get(dtype_str, torch.float32)

        # Optional: compile the model
        if is_compile:
            try:
                print(f"    [{mode_label}] Compiling model with torch.compile()...")
                model = torch.compile(model)
                print(f"    [{mode_label}] Model compiled successfully")
            except Exception:
                tb_text = traceback.format_exc()
                print(f"    [{mode_label}] COMPILE_ERROR: {tb_text[-200:]}")
                result_queue.put(
                    {
                        "status": classify_exception(tb_text, is_compile=True),
                        "eager_latency_ms": 0.0,
                        "compile_latency_ms": 0.0,
                        "speedup": 0.0,
                        "compile_p99_ms": 0.0,
                        "precision_match": False,
                        "error_message": tb_text[-1000:],
                        "output": None,
                    }
                )
                return

        # 5. Warmup
        warmup_iters = max(warmup_iters, 3)
        print(f"    [{mode_label}] Warming up ({warmup_iters} iterations)...")
        with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
            for _ in range(warmup_iters):
                _ = model(**inputs)
        synchronize_if_torch_device(device)
        print(f"    [{mode_label}] Warmup completed")

        # 6. Timed inference
        print(f"    [{mode_label}] Running timed inference ({test_iters} iterations)...")
        latencies: list[float] = []
        outputs = []
        with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):  # TODO
            for _ in range(test_iters):
                synchronize_if_torch_device(device)
                t0 = time.perf_counter()
                out = model(**inputs)
                synchronize_if_torch_device(device)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)
                outputs.append(out)

        # Pick the last output for comparison
        last_output = outputs[-1]

        # 7. Serialize to CPU / numpy
        serializable_output = serialize_output(last_output)

        avg_latency = float(np.mean(latencies))
        p99_latency = float(np.percentile(latencies, 99))
        print(
            f"    [{mode_label}] Inference completed: "
            f"avg={avg_latency:.2f}ms, p99={p99_latency:.2f}ms"
        )

        result_queue.put(
            {
                "status": "SUCCESS",
                "eager_latency_ms": avg_latency if not is_compile else 0.0,
                "compile_latency_ms": avg_latency if is_compile else 0.0,
                "speedup": 0.0,  # computed by orchestrator
                "compile_p99_ms": p99_latency if is_compile else 0.0,
                "precision_match": True,  # provisional; compare.py overrides
                "error_message": "",
                "output": serializable_output,
            }
        )

    except RuntimeError:
        tb_text = traceback.format_exc()
        print(f"    [{mode_label}] RUNTIME_ERROR: {tb_text[-200:]}")
        result_queue.put(
            {
                "status": classify_exception(tb_text, is_compile),
                "eager_latency_ms": 0.0,
                "compile_latency_ms": 0.0,
                "speedup": 0.0,
                "compile_p99_ms": 0.0,
                "precision_match": False,
                "error_message": tb_text[-1000:],
                "output": None,
            }
        )
    except Exception:
        tb_text = traceback.format_exc()
        print(f"    [{mode_label}] CRASH: {tb_text[-200:]}")
        result_queue.put(
            {
                "status": "CRASH",
                "eager_latency_ms": 0.0,
                "compile_latency_ms": 0.0,
                "speedup": 0.0,
                "compile_p99_ms": 0.0,
                "precision_match": False,
                "error_message": tb_text[-1000:],
                "output": None,
            }
        )
    finally:
        cleanup_device(backend)
