# Eager vs Compile 模式 Benchmark 完整避坑指南

> 本文档综合 PyTorch 官方文档、社区实践与华为昇腾 NPU 开发经验，系统梳理 eager 与 `torch.compile`（Inductor）模式下性能测试的架构设计、关键陷阱与解决方案。

---

## 目录

1. [架构设计原则](#1-架构设计原则)
2. [进程隔离与生命周期管理](#2-进程隔离与生命周期管理)
3. [输入数据管理](#3-输入数据管理)
4. [设备同步与计时精度](#4-设备同步与计时精度)
5. [内存管理（GPU/NPU 显存）](#5-内存管理gpunpu-显存)
6. [Warmup 策略](#6-warmup-策略)
7. [Compile 模式特有陷阱](#7-compile-模式特有陷阱)
8. [精度校验](#8-精度校验)
9. [IPC 与结果序列化](#9-ipc-与结果序列化)
10. [错误分类与容灾](#10-错误分类与容灾)
11. [NPU（昇腾 Ascend）专项陷阱](#11-npu昇腾-ascend专项陷阱)
12. [常见错误模式速查表](#12-常见错误模式速查表)

---

## 1. 架构设计原则

### 1.1 Master-Worker 无状态架构

**原则**：主进程是纯粹的调度器，严禁引入任何硬件运行时（CUDA/CANN）或深度学习全局状态。

| 层 | 职责 | 禁止 |
|---|---|---|
| 调度管控层（主进程） | 解析测试矩阵、分发任务、Watchdog 超时、聚合报告 | `import torch`、初始化 CUDA/CANN |
| 沙箱执行层（子进程） | 模型加载、设备初始化、推理测速、结果序列化 | 与主进程共享任何 tensor/device 对象 |
| 校验度量层（主进程） | 精度比对、统计计算、报告生成 | 在主进程持有任何 GPU/NPU 内存引用 |

**为什么**：
- 一旦主进程初始化了 CUDA/CANN runtime，它会持有 GPU context 且无法完全释放。后续子进程可能继承或冲突该 context，导致不可预测的死锁或 OOM。
- PyTorch 的 Dynamo 图缓存、caching allocator 等全局状态在进程间不隔离，会导致状态泄漏。

### 1.2 `torch.no_grad()` 和 `model.eval()` 是推理 Benchmark 的硬性前提

**`torch.no_grad()` 不可省略**：

```python
# ✅ 正确
with torch.no_grad():
    output = model(**inputs)
```

**为什么**：
- 如果不使用 `torch.no_grad()`，PyTorch autograd 会为每一次前向传播构建完整的计算图，保存所有中间激活（intermediate activations）用于反向传播。
- 这会：① 显存占用翻数倍（大模型可能因此 OOM）；② 前向传播本身变慢（记录 op 的开销）；③ 这些计算图永远不会被使用，纯属浪费。
- 可以用 `@torch.no_grad()` 装饰器或 `with torch.no_grad():` 上下文管理器。

**`model.eval()` 不可省略**：

```python
model.to(device).eval()  # 顺序：先转移设备，再切换到评估模式
```

**为什么**：
- `Dropout` 层在 `train` 模式下会随机丢弃神经元，导致：① 每次推理结果不确定；② 实际计算量波动；③ 精度校验必然失败。
- `BatchNorm` 层在 `train` 模式下使用当前 batch 统计量，在 `eval` 模式下使用运行时统计量。不切换会导致输出不正确。
- 某些模型（如带有 `DropPath`、`StochasticDepth` 的 ViT 系）在 `train` 模式下有额外的随机分支，影响延迟测量的稳定性。

### 1.3 串行独占执行

Eager 和 compile benchmark 必须**严格串行**执行，绝对不能并行。

**为什么**：
- PCIe 总线带宽和显存带宽是共享资源，并行执行会导致结果波动、数据不可信。
- NPU/CUDA 的算子编译会占用额外的临时显存，并行可能导致虚假 OOM。
- 两个模式共享同一个 device，设备端执行队列互相干扰会使计时失真。

---

## 2. 进程隔离与生命周期管理

### 2.1 必须使用 `spawn` 启动方式

```python
ctx = multiprocessing.get_context("spawn")
```

**坑**：使用 `fork` 会导致：
- CUDA/CANN context 被子进程继承，产生死锁（PyTorch 官方明确声明 `fork` 在 CUDA 多进程场景下不受支持）
- Python GIL 状态不一致
- 文件描述符泄漏

### 2.2 每个任务独占一个短生命周期子进程

**不要**在同一个子进程中顺序执行多个模型或多种模式。每个 benchmark 任务（如 model_A 的 eager 测试）应独占一个子进程，用完即毁。

**为什么**：
- `torch.compile` 会修改 Dynamo 全局缓存，在进程内无法完全重置（即使调用 `torch._dynamo.reset()`），可能影响后续模型的行为。
- 模型的 CUDA graph、Triton kernel cache 等底层状态无法在进程内彻底清理。
- 一个模型的崩溃（如 C++ 级别 segfault）如果与下一个模型共享进程，会污染后续执行环境。

### 2.3 Worker 生命周期 7 步标准流程

```
1. import torch（在 worker 内首次导入）
2. 加载模型 + tie_weights() + 实例化 meta tensor
3. model.to(device).eval()
4. 在 worker 内生成新鲜输入（随机种子固定）
5. Warmup >= 3 次 + 设备同步计时
6. Output 移至 CPU → 序列化为 NumPy
7. 推入 Queue → sys.exit(0)（进程物理销毁）
```

### 2.4 Watchdog 超时机制

```python
from multiprocessing import Queue, get_context
from queue import Empty

# ...
try:
    result = result_queue.get(timeout=timeout_seconds)
except Empty:
    proc.terminate()
    proc.join(timeout=300)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=300)
    return {"status": "TIMEOUT", ...}
```

**坑**：
- `proc.terminate()` 发送 SIGTERM，允许子进程做有限的清理。但如果子进程卡在 CUDA/CANN 驱动调用中，SIGTERM 可能无法中断。
- 必须在 `terminate()` 之后加 `join(timeout)` 和条件 `kill()`（SIGKILL），确保进程真正销毁。
- `join` 的 timeout 不要设太短（建议 30-300s），因为 CUDA context 清理本身可能耗时。
- 当子进程被 `kill()` 强杀后，GPU 显存回收依赖操作系统和驱动。如果驱动没有正确实现 orphaned context 清理，显存可能不会立即释放。

---

## 3. 输入数据管理

### 3.1 严禁跨进程传递 tensor

**绝对不要**：
```python
# ❌ 错误：在主进程创建 tensor 传给 worker
inputs = torch.randn(1, 10, device="npu")
worker_fn(inputs, queue)
```

**必须**：
```python
# ✅ 正确：只传描述信息，worker 内部生成
task = {"input_shape": [1, 10], "dtype": "float32", "device": "npu"}
# worker 内部：
inputs = torch.randn(*task["input_shape"], device=task["device"], dtype=getattr(torch, task["dtype"]))
```

**为什么**：
- 跨进程传递 CUDA/NPU tensor 会触发 PyTorch 的 `multiprocessing` tensor 共享机制（基于 IPC handle + file descriptor），这本身引入了隐式的设备同步和内存映射。
- 如果主进程持有 tensor 引用，即使子进程已退出，GPU 显存可能不会释放（因为 Python GC 在主进程中仍持有 tensor 对象的引用计数）。
- 跨进程传递的 tensor 可能带有 stale 的 computation graph 或 gradient 状态。

### 3.2 严禁 eager 和 compile 共用输入

**绝对不要**在 eager 测试中生成的 tensor 对象直接用于 compile 测试。即使是同一 shape，也必须在各自的 worker 中分别生成。

**为什么**：
- PyTorch tensor 可能带有 autograd 信息、`_base` 指针、storage 共享等隐式状态。
- `torch.compile` 可能修改 tensor 的 storage 布局（如 layout transformation），影响后续 eager 模式的正确性。
- 共享输入会使得 eager 和 compile 之间不再是独立实验，引入 confounding variable。

### 3.3 随机种子必须固定但独立

```python
# 在 worker 内：
torch.manual_seed(42)  # 固定种子，确保精度可比较
inputs = torch.randn(batch, seq_len, device=device, dtype=dtype)
```

**坑**：
- 种子必须在 worker 进程内部设置，因为 `spawn` 后子进程的随机状态与主进程独立。
- 如果 eager 和 compile 使用不同的随机种子，输出将不可比较，精度校验毫无意义。
- 但如果 eager 和 compile 各自 spawn 独立子进程（推荐做法），只要在各自的 worker 内使用相同种子，就能保证输入一致。

---

## 4. 设备同步与计时精度

### 4.1 最常见的错误：不正确的计时

**错误示范**：
```python
# ❌ 只测量了 kernel launch 时间，不是实际执行时间
start = time.perf_counter()
output = model(inputs)
end = time.perf_counter()
latency = end - start
```

**为什么错误**：
- GPU/NPU 操作是异步的。`model(inputs)` 只是向设备提交了计算任务，CPU 立即返回。
- `perf_counter` 测量的是 CPU 端的 wall time，可能只包含 kernel launch 的微秒级开销。
- 真正的设备端计算可能还没完成。

### 4.2 正确的计时模式

**CUDA 场景（使用 Event）**：
```python
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
output = model(inputs)
end_event.record()
torch.cuda.synchronize()  # 阻塞等待设备端所有操作完成

latency_ms = start_event.elapsed_time(end_event)
```

**NPU 场景（使用 synchronize + perf_counter）**：
```python
torch.npu.synchronize()
start = time.perf_counter()
output = model(inputs)
torch.npu.synchronize()
end = time.perf_counter()

latency_ms = (end - start) * 1000
```

### 4.3 同步位置的错误

**坑 1：只在末尾同步**
```python
# ❌ start 计时点不精确
start = time.perf_counter()  # 可能在此之前 GPU 还有未完成的操作
output = model(inputs)
torch.npu.synchronize()
end = time.perf_counter()
```

**正确**：计时前也必须同步，确保设备端为空闲状态。

**坑 2：在计时区间内调用同步操作**
```python
# ❌ print(output) 会隐式触发 CPU-GPU 同步
start_event.record()
output = model(inputs)
print(output)  # 隐式同步！
end_event.record()
```

以下操作会隐式触发同步，绝对不能出现在计时区间内：
- `print(device_tensor)` — 隐式 `.cpu()` 转换
- `tensor.item()` — 需要等待设备端结果
- `tensor.cpu()` — 显式设备到主机传输
- `tensor.nonzero()` — 需要 CPU 端数据
- 任何依赖设备端 tensor 值的 Python 控制流（`if (tensor > 0).all()`）

### 4.4 统计指标选择

- **不要**只报 mean，GPU/NPU 推理延迟通常有长尾。
- **推荐**：报 `median`（P50）+ `P99`（长尾延迟）。
- 至少运行 10 次正式测量（不含 warmup），取统计值。
- 使用 `torch.utils.benchmark.Timer.blocked_autorange`（自动确定 warmup/测量分割，总计至少 0.2 秒）作为更严谨的替代方案。

---

## 5. 内存管理（GPU/NPU 显存）

### 5.1 CPU 侧持有 tensor 引用导致显存不释放

**这是 NPU benchmark 中最容易踩、也最难定位的坑。**

```python
# ❌ 主进程间接持有 NPU tensor 引用
result = {
    "output": output_tensor,  # 这是一个 NPU tensor!
}
queue.put(result)
# 即使子进程退出，主进程通过 queue 拿到的 result["output"]
# 指向 NPU 设备内存。如果主进程没有及时释放这个引用，
# NPU 驱动不会回收这块显存。
```

**机制解释**：
- PyTorch tensor 在 CPU 端有一个 Python wrapper 对象，在设备端有实际的存储。
- 即使子进程退出，如果 `multiprocessing.Queue` 将 tensor 的 IPC handle 传到了主进程，主进程的 Python GC 持有该 handle。
- CUDA/NPU 驱动只有在所有引用都被释放后，才会回收设备端内存。
- 主进程不会主动释放这些引用（因为主进程不知道何时该释放），导致显存泄漏。

**正确做法**：
```python
# ✅ Worker 端：移至 CPU → 转为 NumPy → 删除 tensor 引用
output_cpu = output_tensor.cpu()
numpy_data = output_cpu.numpy()
del output_cpu, output_tensor
result = {"output": numpy_data}  # 纯 NumPy，无任何设备端引用
queue.put(result)
```

### 5.2 `empty_cache()` 的正确使用

```python
import gc

# 在 worker 的 finally 块中：
try:
    # ... benchmark logic ...
finally:
    # 1. 删除大对象引用
    del model, inputs, output
    # 2. 强制 Python GC 回收循环引用（关键！）
    gc.collect()
    # 3. 释放设备缓存（必须在 gc.collect() 之后）
    if device_type == "cuda":
        torch.cuda.empty_cache()
    elif device_type == "npu":
        torch.npu.empty_cache()
```

**坑**：
- `gc.collect()` 必须在 `empty_cache()` **之前**调用。Python 的引用计数机制无法回收循环引用（如模型中的 `self.attn = self.attn` 这类自引用结构），只有 `gc.collect()` 才能断开这些循环。如果不先 `gc.collect()`，循环引用中的 tensor 仍被 Python 持有，`empty_cache()` 无法释放对应的设备内存。
- `empty_cache()` 只释放 caching allocator 中未被占用的缓存块。如果仍有 Python 引用指向 tensor，对应显存不会被释放。
- `empty_cache()` 本身有性能开销（触发设备端同步和内存整理），不要在计时区间内调用。
- 在 `finally` 中调用 `empty_cache()` 后，如果子进程随后 `sys.exit(0)`，操作系统会回收所有资源。这是双保险策略。

### 5.3 显存碎片化

**坑**：PyTorch 的 caching allocator 可能总空闲显存足够，但没有连续的大块来满足分配请求，导致虚假 OOM。

**缓解措施**：
- 每个模型使用独立子进程，进程退出后显存完整释放，彻底避免碎片化。
- 如果必须在同一进程中测试多个模型，在模型之间调用 `empty_cache()` + `gc.collect()`。
- 使用 `torch.cuda.memory._snapshot()` （或 NPU 等价 API）诊断碎片化问题。

### 5.4 模型加载过程中的显存峰值

**坑**：模型加载到设备时的显存峰值可能远大于稳态推理时的占用。

- 加载过程中，权重从 CPU → 设备传输时，可能同时存在 CPU 副本和设备副本。
- 某些模型的 `tie_weights()` 或 `resize_token_embeddings()` 会触发额外的显存分配。
- 建议：先在 CPU 上完成所有模型初始化（`tie_weights()`、meta tensor 实例化），再 `.to(device)`。

---

## 6. Warmup 策略

### 6.1 为什么需要 Warmup

设备端第一次执行操作时会有大量一次性开销：

| 开销来源 | 延迟量级 | 说明 |
|---|---|---|
| cuBLAS/cuDNN lazy init | ~2-3 ms | 首次 CUDA 调用时加载（官方教程实测约 2775 μs，但具体值取决于硬件和 CUDA 版本） |
| Triton kernel JIT 编译 | 秒级 | `torch.compile` 首次执行每个 kernel 时触发 |
| CUDA Graph capture | 秒级 | `reduce-overhead` 模式 |
| NPU 算子编译 | 秒级 | CANN 算子库首次匹配 |
| 硬件频率爬升 | ~100ms | GPU/NPU 从低功耗状态提升频率 |
| 内存分配器预热 | ~10ms | Caching allocator 首次分配 block |

### 6.2 Warmup 次数

- **最少 3 次**前向传播。
- 如果使用 `torch.compile`，首次 forward 包含编译开销（可能 30s+），建议 warmup 包含至少 1 次 compiled forward。
- 对于 autotuning 模式（`max-autotune`），Triton 会尝试多种 kernel 配置，warmup 需要更多次数。
- 推荐 5-10 次 warmup，然后 10-50 次正式测量。

### 6.3 Warmup 的常见错误

```python
# ❌ Warmup 使用与正式测试不同的 input shape
for _ in range(3):
    model(torch.randn(1, 10))  # warmup shape: [1, 10]
# 正式测量
for _ in range(10):
    model(torch.randn(1, 128))  # 测量 shape: [1, 128] → 触发重新编译!
```

**必须**确保 warmup 和正式测量使用完全相同的 input shape 和 dtype。不同的 shape 会触发 `torch.compile` 重新编译。

---

## 7. Compile 模式特有陷阱

### 7.1 编译开销不在 latency 内

`torch.compile` 是 lazy（JIT）编译：`torch.compile(model)` 本身**只创建 wrapper，不触发编译**。编译发生在 **compiled model 的首次前向传播**时（即第一次 `compiled_model(**inputs)` 调用），耗时可能从几秒到几分钟不等（取决于模型复杂度和 backend）。官方教程中 DenseNet121 的首次 compiled forward 耗时 77.75 秒（含编译），后续仅需 ~0.35ms。

**正确做法**：
```python
compiled_model = torch.compile(model, backend=backend_str)

# 编译时间单独记录
compile_start = time.perf_counter()
_ = compiled_model(**warmup_inputs)  # 触发编译
torch.npu.synchronize()
compile_end = time.perf_counter()
compile_time_ms = (compile_end - compile_start) * 1000

# Warmup（编译已完成）
for _ in range(warmup_iters - 1):
    _ = compiled_model(**warmup_inputs)
torch.npu.synchronize()

# 正式测量（不含编译时间）
latencies = []
for _ in range(test_iters):
    torch.npu.synchronize()
    start = time.perf_counter()
    _ = compiled_model(**inputs)
    torch.npu.synchronize()
    latencies.append((time.perf_counter() - start) * 1000)
```

### 7.2 compile_time 的正确记录方式

**坑**：不要用 "首次 forward 时间 - 稳态均值" 来估算 compile time。首次 forward 和稳态 forward 的测量方式不同（单次 vs 逐次同步），相减会产生偏差——尤其是稳态均值可能因 kernel 流水线重叠而偏低，导致 compile time 被高估。

**正确做法**：直接记录首次 forward 的总耗时作为 `compile_time_ms`，不做减法。首次 forward = Dynamo 追踪 + Inductor codegen + kernel launch + 设备执行，这正是用户关心的 "编译一次要等多久"。稳态推理延迟则由后续逐次测量独立提供。

```python
# 首次 forward（含编译）
synchronize_if_torch_device(device)
t0 = time.perf_counter()
_ = compiled_model(**inputs)
synchronize_if_torch_device(device)
compile_time_ms = (time.perf_counter() - t0) * 1000.0  # 直接使用，不减 avg_latency

# 后续逐次测量（不含编译时间）
latencies = []
for _ in range(test_iters):
    synchronize_if_torch_device(device)
    t0 = time.perf_counter()
    _ = compiled_model(**inputs)
    synchronize_if_torch_device(device)
    latencies.append((time.perf_counter() - t0) * 1000)

avg_latency = sum(latencies) / len(latencies)
```

### 7.3 Dynamo 缓存污染

**坑**：如果在同一进程中先 compile 模型 A，再 compile 模型 B，Dynamo 的字节码分析缓存可能残留模型 A 的信息。

**解决方案**：
- 每个模型使用独立子进程（推荐）。
- 如果必须在同一进程中，在两个模型之间调用：
  ```python
  torch._dynamo.reset()
  ```

### 7.4 Graph Break

`torch.compile` 在遇到不支持的 Python 特性时会触发 graph break，导致编译后的代码被分段执行，性能可能反而不如 eager。

**常见触发 graph break 的操作**：
- Python `print()` 在模型 forward 中
- 数据依赖的控制流（如 `if tensor.item() > 0`）
- 不支持的 Python 内置函数（如 `id()`, `type()` 在 tensor 上）
- 外部 C extension 调用
- 某些 HuggingFace 模型中的自定义操作

**检测方法**：
```python
# 直接用 explain 分析原始模型，不需要先 torch.compile
torch._dynamo.explain(model)(*inputs)  # 输出 graph break 信息
```

### 7.5 Inductor Backend 选择

| Backend / Mode | 适用场景 | 注意事项 |
|---|---|---|
| `inductor`（默认 mode=`default`） | GPU（Triton 后端） | 不适用于 NPU |
| `inductor` + mode=`reduce-overhead` | GPU 小 batch | 使用 CUDA Graphs 减少 Python 开销，要求 CUDA-only 图且不修改输入 |
| `inductor` + mode=`max-autotune` | GPU 最高性能 | 尝试多种 Triton/模板策略，默认启用 CUDA Graphs |
| `inductor` + mode=`max-autotune-no-cudagraphs` | GPU 最高性能但不用 CUDA Graphs | 与 `max-autotune` 相同的 autotuning，但不启用 CUDA Graphs |
| `inductor` + `dvm` | 华为昇腾 NPU | 需要 torch_npu 支持 |
| `inductor` + `mlir` | 华为昇腾 NPU | MLIR-based 后端 |
| `eager` | 调试/基准对比 | 无编译优化 |
| `aot_eager` | AOT Autograd 但不做 codegen | 比 `eager` 快，但比 `inductor` 慢 |

### 7.6 Compile 缓存管理

如果需要测量 compile 时间（而非仅测量推理延迟），缓存清理策略有重大区别：

**坑**：不要在 worker 内部清除缓存。在 worker 内调用 `shutil.rmtree()` 会把文件 I/O 开销算入 compile 时间，引入测量噪声。正确做法是通过**外层 shell 脚本**控制缓存清理和 worker 启动。

**推荐做法 — shell 脚本控制**：
```bash
#!/bin/bash
# run_compile_benchmark.sh

# 每次运行前清空所有编译缓存
rm -rf $TORCHINDUCTOR_CACHE_DIR
rm -rf $TRITON_CACHE_DIR
rm -rf $ASCEND_CACHE_PATH

# 启动 benchmark（worker 在干净环境下启动）
python -m src.benchmark.run_benchmark --model $1 --inductor-backend $2
```

**为什么不用 worker 内清除**：
- Worker 内 `shutil.rmtree()` 的耗时取决于缓存目录大小（可能几百 MB），这会被误计入 compile_time_ms。
- 缓存清理应该在 worker spawn **之前**完成，确保 worker 从零开始编译，而非"边删边编译"。
- Shell 脚本还能确保每次运行的环境一致性（无残留编译产物）。

**Worker 内的正确处理**：Worker 本身不应该碰缓存目录，只负责 `torch._dynamo.reset()` 清除 Dynamo 的内存态缓存。

---

## 8. 精度校验

### 8.1 为什么 Eager 和 Compile 输出可能不同

`torch.compile` / Inductor 会：
- **融合算子**：将多个 elementwise 操作合并为单个 kernel，改变计算顺序。
- **重排计算**：交换操作顺序（如将 activation 融合到 matmul 中），导致浮点累加顺序不同。
- **精度提升**：某些 reduction 操作在 Triton kernel 中可能使用更高精度的 accumulator。
- **生成自定义 kernel**：Triton 生成的 kernel 可能使用不同的 tiling 策略。

这些都是**合法的浮点差异**，不是 bug。但需要用适当的容差来判断。

### 8.2 容差标准

> **注意**：以下容差是**针对 eager vs compile 比对的实践推荐值**，并非 PyTorch 默认值。PyTorch `torch.allclose` 默认容差为 `rtol=1e-5, atol=1e-8`，对于 compile 比对来说过于严格。以下值是基于 TF32 精度损失（Ampere+ tensor cores 仅读取 10 位尾数）和 BF16 低精度（仅 7 位尾数）的经验值。

| 类型 | rtol | atol | 说明 |
|---|---|---|---|
| Int / Bool | exact match | exact match | 不允许任何偏差 |
| Float32 | 1e-4 | 1e-4 | 考虑 TF32 和算子融合带来的累加顺序差异 |
| BFloat16 | 1e-2 | 1e-2 | BF16 尾数只有 7 位，舍入误差显著大于 FP32 |
| Float16 | 5e-3 | 5e-3 | FP16 尾数 10 位 |
| NaN / Inf | `equal_nan=True` | — | 相同位置的 NaN/Inf 视为相等 |

### 8.3 递归比较嵌套结构

HuggingFace 模型的输出通常是 `ModelOutput` 对象，可能嵌套 dict / tuple / list：

```python
def compare_outputs(expected, actual, dtype_str, path="root"):
    """递归比较嵌套结构"""
    if isinstance(expected, (dict, ModelOutput)):
        for key in expected:
            compare_outputs(expected[key], actual[key], dtype_str, f"{path}.{key}")
    elif isinstance(expected, (tuple, list)):
        for i, (e, a) in enumerate(zip(expected, actual)):
            compare_outputs(e, a, dtype_str, f"{path}[{i}]")
    elif isinstance(expected, torch.Tensor):
        compare_tensors(expected, actual, dtype_str, path)
```

### 8.4 错误报告

精度失败时，必须报告：
- `max_absolute_error`：最大绝对误差
- `max_relative_error`：最大相对误差
- `failing_path`：出错的张量路径（如 `past_key_values.layer_3.key`）
- 出错张量的 shape 和 dtype

---

## 9. IPC 与结果序列化

### 9.1 Worker → 主进程的结果序列化规则

**必须序列化为 NumPy**：

```python
def serialize_output(value):
    """将 PyTorch tensor 转为可跨进程传输的纯数据结构"""
    if isinstance(value, torch.Tensor):
        return {"__type__": "tensor", "data": value.cpu().numpy(), "dtype": str(value.dtype)}
    elif isinstance(value, dict):
        return {k: serialize_output(v) for k, v in value.items()}
    elif isinstance(value, (tuple, list)):
        return [serialize_output(v) for v in value]
    elif isinstance(value, (int, float, str, bool)):
        return value
    elif hasattr(value, '__dict__'):  # ModelOutput 等
        return serialize_output({k: v for k, v in value.items()})
    return None
```

### 9.2 为什么必须转为 NumPy

- `multiprocessing.Queue` 使用 pickle 序列化。
- PyTorch CUDA/NPU tensor 的 pickle 会使用 IPC handle（文件描述符或共享内存句柄），这些在跨进程后可能导致：
  - 设备端内存不释放（因为反序列化端持有引用）
  - 死锁（某些平台上 CUDA IPC 不支持）
  - 文件描述符泄漏
- NumPy array 存储在 CPU 主存中，pickle 后是纯数据，无任何设备端依赖。

### 9.3 主进程 → Worker 的 payload 规则

只传递**基础类型**（str, int, bool, float）：

```python
task = {
    "model_dir": "/path/to/model",  # str
    "is_compile": True,             # bool
    "dtype": "float32",             # str
    "warmup_iters": 5,              # int
    "test_iters": 10,               # int
    "device": "npu",                # str
    "seed": 42,                     # int
}
```

**绝对不要**传递：model 对象、tensor、optimizer、dataloader 等任何复杂对象。

---

## 10. 错误分类与容灾

### 10.1 错误分类体系

| 状态 | 含义 | 触发条件 |
|---|---|---|
| `SUCCESS` | 执行成功且精度通过 | latency > 0 且 precision_match == True |
| `OOM` | 显存不足 | traceback 包含 `CUDA out of memory` 或 `CANN memory allocation failed` |
| `TIMEOUT` | 超时 | `Queue.get(timeout=...)` 超时 |
| `PRECISION_FAIL` | 精度校验失败 | eager/compile 都成功但输出不匹配 |
| `COMPILE_ERROR` | 编译器错误 | Inductor/Triton/CANN 算子编译失败 |
| `CRASH` | 其他崩溃 | 未分类的异常 |

### 10.2 OOM 的识别

```python
OOM_PATTERNS = [
    "CUDA out of memory",
    "CANN memory allocation failed",
    "OutOfMemoryError",
    "Unable to find a valid cuDNN algorithm",
]

def classify_exception(traceback_text: str, is_compile: bool) -> str:
    for pattern in OOM_PATTERNS:
        if pattern in traceback_text:
            return "OOM"
    if is_compile and ("Triton" in traceback_text or "inductor" in traceback_text.lower()):
        return "COMPILE_ERROR"
    return "CRASH"
```

### 10.3 Traceback 截断的必要性

**为什么 error_message 必须截断到 1000 字符**：
- 大型模型（如 LLaMA、Qwen）的 traceback 嵌套层数极深（transformers → modeling_utils → module forward → attention → ...），完整 traceback 可能达到数万字符。
- `multiprocessing.Queue` 内部使用 pickle 序列化后通过 pipe 传输。超大 payload 会阻塞 pipe 的写入端，如果主进程此时正在等待 `Queue.get(timeout=...)`，可能导致超时误判。
- 截断时保留**尾部**（`tb[-1000:]`），因为 Python traceback 的关键错误信息（异常类型 + 错误消息）在最末尾。

### 10.4 `COMPILE_ERROR` 的升级工作流

`COMPILE_ERROR` 不是普通的运行时错误，它意味着编译器（Inductor / Triton / CANN 算子库）无法处理当前模型架构。处理流程：

1. **记录完整的 compile 日志**：设置 `torch._logging.set_logs(dynamo=True, aot=True, inductor=True)` 捕获编译全过程。
2. **提取关键信息**：失败的算子名称、输入 shape、dtype。
3. **自动生成 issue 模板**：包含模型名称、PyTorch/torch_npu 版本、完整 compile log。
4. **升级给编译器团队**：标记为 `COMPILE_ERROR` 的结果不参与 speedup 统计（避免拉低平均 speedup），但需要在报告中单独列出，供编译器团队排障。

### 10.5 子进程清理顺序

```python
import gc
import traceback

def worker_main(task, result_queue):
    try:
        # ... benchmark logic ...
        result_queue.put(result)
    except Exception as e:
        tb = traceback.format_exc()
        result_queue.put({
            "status": classify_exception(tb, task["is_compile"]),
            "error_message": tb[-1000:],  # 截断！取尾部，关键信息在末尾
            ...
        })
    finally:
        # 1. 删除大对象引用
        del model, inputs, output
        # 2. 强制 GC 回收循环引用（必须在 empty_cache 之前）
        gc.collect()
        # 3. 释放设备缓存
        if device_type == "cuda":
            torch.cuda.empty_cache()
        elif device_type == "npu":
            torch.npu.empty_cache()
        # 4. 进程退出，操作系统回收所有资源
        sys.exit(0)
```

---

## 11. NPU（昇腾 Ascend）专项陷阱

### 11.1 torch_npu 与 PyTorch 版本兼容性

torch_npu 版本必须与 PyTorch 版本和 CANN 版本严格匹配。参考 [Ascend/pytorch](https://github.com/Ascend/pytorch) 的兼容矩阵。

**坑**：
- CANN 环境变量未设置会导致 `import torch_npu` 静默失败或设备不可用。
- 必须先 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`。

### 11.2 NPU 设备操作的特殊性

```python
# NPU tensor 创建方式
x = torch.randn(2, 2).npu()  # 方式 1
x = torch.randn(2, 2, device="npu")  # 方式 2（推荐）
```

**坑**：
- `tensor.npu()` 是先在 CPU 创建再拷贝到 NPU，比直接在 NPU 上创建多一次 CPU→NPU 传输。
- 必须使用 `torch.npu.synchronize()` 而非 `torch.cuda.synchronize()`。
- `torch.npu.Event` 可能不完全支持 `enable_timing=True`（根据实际经验，取决于 torch_npu 版本；如不可用则用 `synchronize + perf_counter` 替代）。

### 11.3 NPU 显存管理的差异

- `torch.npu.empty_cache()` 的行为可能与 `torch.cuda.empty_cache()` 不同，具体取决于 CANN 版本。
- NPU 驱动层面的 context 管理可能与 CUDA 不同，进程强杀后的显存回收可能不完全。
- 建议在 NPU benchmark 中增加显存使用监控：
  ```python
  if torch.npu.is_available():
      allocated = torch.npu.memory_allocated() / 1024**3
      reserved = torch.npu.memory_reserved() / 1024**3
  ```

### 11.4 NPU Compile 后端

华为昇腾 NPU 的 `torch.compile` 后端选项：
- **dvm**：华为自研的深度向量机后端
- **mlir**：基于 MLIR 的后端

```python
# NPU compile 方式
compiled_model = torch.compile(model, backend="inductor")  # 可能不工作
# 需要设置正确的 backend 和环境变量
```

### 11.5 NPU 上的 known issues

- 某些 PyTorch 算子在 NPU 上可能有数值差异（如某些 reduction 操作的累加精度）。
- NPU 的 op fusion 规则与 CUDA 不同，可能导致 `torch.compile` 生成的计算图不同。
- NPU 驱动级别的 watchdog 可能会在长时间无响应时主动 kill 进程，这与应用层的 timeout 机制可能冲突。

### 11.6 模型加载安全性与 `trust_remote_code`

HuggingFace 的 `AutoModel.from_pretrained()` 有一个 `trust_remote_code` 参数：

```python
# 默认行为（安全）
model = AutoModel.from_pretrained(model_dir, local_files_only=True)
# trust_remote_code=False（默认）会拒绝加载包含自定义代码的模型

# 允许自定义代码（有风险）
model = AutoModel.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
```

**坑**：
- `trust_remote_code=True` 会执行模型目录中的 Python 文件（如 `modeling_*.py`、`configuration_*.py`），这些文件可能包含任意代码。在自动化 benchmark 环境中，这相当于执行不受信任的代码。
- 某些模型（如 Qwen 系列的早期版本）**必须**启用 `trust_remote_code=True` 才能加载，否则会抛出 `ValueError`。
- **推荐策略**：Benchmark 初版设置 `trust_remote_code=False`，遇到需要自定义代码的模型时明确标记为 `CRASH` 并记录错误信息。后续根据需要逐个模型白名单放行。
- 加载失败时的 traceback 中搜索 `"trust_remote_code"` 关键字，可以自动判断是否是此问题。

---

## 12. 常见错误模式速查表

| 症状 | 根因 | 解决方案 |
|---|---|---|
| Eager 正常但 Compile 后 OOM | Compile 的 Triton kernel 需要额外临时显存 | 减少 batch size 或换用 `reduce-overhead` 模式 |
| 第一个模型通过，后续模型全部 OOM | 前一个子进程的显存未释放 | 确保每个模型用独立子进程 + `empty_cache()` + `sys.exit(0)` |
| Latency 波动巨大（±50%） | 未做 warmup 或同步位置不对 | 增加 warmup 次数，确保 `synchronize()` 在计时前后 |
| Compile 比 Eager 慢 | Graph break 或 backend 不匹配 | 用 `torch._dynamo.explain()` 检查 graph break |
| 精度校验全部失败 | Eager 和 Compile 使用了不同输入 | 确保两者使用相同随机种子 |
| 精度校验 BFloat16 失败但 Float32 正常 | BFloat16 容差太严格 | 使用 `rtol=1e-2, atol=1e-2` |
| Worker 超时且无法回收 | CUDA/CANN 驱动级死锁 | 使用 `kill()` (SIGKILL) 强杀，依赖 OS 回收 |
| 主进程卡在 Queue.get() | Worker 在序列化输出时 crash | 使用 `Queue.get(timeout=...)` |
| `speedup < 1`（compile 反而更慢） | 测量包含了编译时间 | 将编译时间单独记录，仅测量 warmup 后的推理延迟 |
| NPU 上的 latency 远高于预期 | 使用了 `.npu()` 而非 `device="npu"` 创建 tensor | 使用 `torch.randn(shape, device="npu")` |
| 多次运行结果不一致 | 随机种子未固定或硬件频率未稳定 | 固定种子 + 增加 warmup |
| 显存异常高/推理异常慢 | 忘记 `torch.no_grad()` | 用 `with torch.no_grad():` 包裹推理 |
| 每次 forward 结果不同 | 忘记 `model.eval()` | 加载后立即调用 `.eval()` |
| `empty_cache()` 后显存仍未释放 | 未先调用 `gc.collect()` | `del` → `gc.collect()` → `empty_cache()` |
| Compile 时间测量波动大 | 在 worker 内清除缓存引入 I/O 噪声 | 用 shell 脚本在 worker 启动前清缓存 |
| `Queue.get()` 超时但 worker 已完成 | Traceback 过大导致 IPC pipe 阻塞 | 截断 traceback 到 1000 字符 |
| 模型加载失败 `ValueError` | 需要 `trust_remote_code=True` | 白名单管理，默认 `False` |

---

## 参考资料

- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html) — 同步、内存管理、算子融合
- [Introduction to torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) — 官方 timing 模式、编译模式
- [PyTorch Benchmark Tutorial](https://pytorch.org/tutorials/recipes/recipes/benchmark.html) — `torch.utils.benchmark.Timer`，`blocked_autorange`
- [Understanding GPU Memory](https://pytorch.org/blog/understanding-gpu-memory-1/) — Caching allocator、Memory Snapshot API
- [torch.compiler API](https://pytorch.org/docs/stable/torch.compiler.html) — Backend 列表、Dynamo/Inductor 架构
- [Ascend/pytorch (torch_npu)](https://github.com/Ascend/pytorch) — 昇腾 NPU 适配器
- [PyTorch Multiprocessing Notes](https://pytorch.org/docs/stable/notes/multiprocessing.html) — spawn vs fork、CUDA IPC
