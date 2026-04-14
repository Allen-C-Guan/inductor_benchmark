"""Isolated sandbox worker for benchmark inference.

All hardware interaction (device init, model load, forward pass) happens
inside this module, which is run in a dedicated ``spawn`` subprocess.
"""

from __future__ import annotations

import sys
import time
import traceback
from queue import Queue
from typing import Any

import numpy as np

from .model_loader import (
    load_model,
    make_inputs,
)


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


def synchronize_if_torch_device(device: str) -> None:
    """Call the appropriate synchronize API for the device."""
    import torch
    torch.npu.synchronize()


def cleanup_device() -> None:
    """Release cached device memory."""
    import torch
    torch.npu.empty_cache()


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
    if "cann memory allocation failed" in lower:
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
    dtype_str: str = task["dtype"]
    warmup_iters: int = task["warmup_iters"]
    test_iters: int = task["test_iters"]
    task_type_hint: str = task["task"]
    inductor_backend: str = task["inductor_backend"]
    mode_label = "compile" if is_compile else "eager"

    try:
        print(f" ***********************************************{task['model_dir']} 开始测试 ******************************************************")
        # 1-2. Load model (task_type is inferred internally)
        model, config, task_type = load_model(model_dir, task_type_hint, dtype_str)
        print(f"    [{mode_label}] Model loaded successfully (task_type={task_type})")

        # 3. Device transfer
        device = "npu:0"
        model = model.to(device).eval()

        # Materialize any remaining meta tensors
        for param in model.parameters():
            if param.device.type == "meta":
                param.data = torch.empty_like(param.data, device=device)

        # 4. Generate fresh inputs
        inputs = make_inputs(config, task_type, device, dtype_str)


        # Optional: compile the model
        if is_compile:
            try:
                import torch._dynamo
                torch._dynamo.reset()
                print(f"    [{mode_label}] Compiling model with torch.compile()...")
                model = torch.compile(model, backend="inductor", options={"npu_backend": inductor_backend})
                print(f"    [{mode_label}] Model compiled successfully (Wrapping done)")
            except Exception:
                tb_text = traceback.format_exc()
                print(f"    [{mode_label}] COMPILE_ERROR: {tb_text}")
                result_queue.put(
                    {
                        "status": classify_exception(tb_text, is_compile=True),
                        "latency_ms": 0.0,
                        "p99_latency_ms": 0.0,
                        "compile_time_ms": 0.0, 
                        "error_message": tb_text[-1000:],
                        "output": None,
                    }
                )
                return

        # 5. Warmup
        print(f"    [{mode_label}] Warming up ({warmup_iters} iterations)...")
        
        first_pass_ms = 0.0 

        with torch.no_grad():
            if is_compile:
                synchronize_if_torch_device(device)
                t0_compile = time.perf_counter()
                _ = model(**inputs) # 这里触发真正的底层编译
                synchronize_if_torch_device(device)
                t1_compile = time.perf_counter()
                
                first_pass_ms = (t1_compile - t0_compile) * 1000.0
                print(f"    [{mode_label}] First forward pass (JIT trigger) took: {first_pass_ms:.2f} ms")
                
                # 执行剩余的 Warmup
                for _ in range(max(0, warmup_iters - 1)):
                    _ = model(**inputs)
            else:
                # Eager 模式正常执行
                for _ in range(warmup_iters):
                    _ = model(**inputs)
                    
        synchronize_if_torch_device(device)
        print(f"    [{mode_label}] Warmup completed")

        print(f"    [{mode_label}] Running timed inference ({test_iters} iterations)...")
        last_output = None
        
        # 将同步放在循环外面
        synchronize_if_torch_device(device)
        t0_total = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(test_iters):
                last_output = model(**inputs) 
                
        synchronize_if_torch_device(device)
        t1_total = time.perf_counter()

        # 7. 计算与返回
        # 这种算出来的平均时间才能真正体现 Compile 的并行与融合优势
        avg_latency = ((t1_total - t0_total) * 1000.0) / test_iters 
        p99_latency = avg_latency # 吞吐量模式下，用 avg 代替 p99 评估整体性能
        
        serializable_output = serialize_output(last_output)
        compile_time_ms = 0.0
        if is_compile:
            compile_time_ms = max(0.0, first_pass_ms - avg_latency)

        print(
            f"    [{mode_label}] Inference completed: "
            f"avg={avg_latency:.2f}ms, p99={p99_latency:.2f}ms | Pure Compile Time={compile_time_ms:.2f}ms"
        )

        result_queue.put(
            {
                "status": "SUCCESS",
                "latency_ms": avg_latency,
                "p99_latency_ms": p99_latency,
                "compile_time_ms": compile_time_ms, 
                "error_message": "",
                "output": serializable_output,
            }
        )
        print(f"*********************************************** {task['model_dir']} 测试完成 ******************************************************")

    except RuntimeError:
        tb_text = traceback.format_exc()
        print(f"    [{mode_label}] RUNTIME_ERROR: {tb_text}")
        result_queue.put(
            {
                "status": classify_exception(tb_text, is_compile),
                "latency_ms": 0.0,
                "p99_latency_ms": 0.0,
                "compile_time_ms": 0.0, 
                "error_message": tb_text[-1000:],
                "output": None,
            }
        )
    except Exception:
        tb_text = traceback.format_exc()
        print(f"    [{mode_label}] CRASH: {tb_text}")
        result_queue.put(
            {
                "status": "CRASH",
                "latency_ms": 0.0,
                "p99_latency_ms": 0.0,
                "compile_time_ms": 0.0, # 
                "error_message": tb_text[-1000:],
                "output": None,
            }
        )
    finally:
        cleanup_device()
