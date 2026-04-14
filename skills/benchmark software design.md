# NPU Benchmark 软件设计文档

> 通用 NPU 推理 benchmark 框架设计，覆盖 eager 与 torch.compile（Inductor）模式。
> 综合自 PyTorch 官方 benchmark 设计（`torch.utils.benchmark`）、HuggingFace optimum-benchmark 架构、以及昇腾 NPU 工程实践。

---

## 1. 设计目标

| 目标 | 说明 |
|---|---|
| **正确性** | 测量值必须反映真实设备端延迟，不受异步执行、缓存、数据污染影响 |
| **隔离性** | 每个 benchmark 任务在独立子进程中完成，互不干扰；主进程不持有任何硬件 runtime 状态 |
| **可复现性** | 相同配置下多次运行结果一致（固定种子、确定性输入、充分 warmup） |
| **可扩展性** | 新增模型类型只需实现 Adapter 接口，无需修改编排逻辑 |
| **可观测性** | 全链路日志、精度诊断报告、统计质量警告，出问题可追溯 |

---

## 2. 系统架构 — 三层分离

```
┌─────────────────────────────────────────────────────────┐
│                  调度管控层 (Orchestrator)                │
│  ┌─────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │ CLI/Config│  │ Task Scheduler│  │ Report Aggregator   │  │
│  └─────────┘  └──────────────┘  └─────────────────────┘  │
│  职责: 解析配置 → 生成任务矩阵 → 串行调度 → 聚合报告       │
│  禁止: import torch / 初始化 CUDA/CANN                   │
└──────────────────────┬──────────────────────────────────┘
                       │ spawn subprocess
                       │ Queue.put(task_payload)
                       │ Queue.get(timeout=T)
┌──────────────────────▼──────────────────────────────────┐
│                  沙箱执行层 (Worker)                      │
│  ┌──────────────────────────────────────────────────┐    │
│  │ 1. import torch (worker 内首次)                    │    │
│  │ 2. ModelAdapter.load() → model.to(device).eval()  │    │
│  │ 3. ModelAdapter.make_inputs() (固定种子)            │    │
│  │ 4. Optional: torch.compile(model)                  │    │
│  │ 5. Warmup (>=3) + synchronize + 逐次计时            │    │
│  │ 6. serialize_output() → NumPy                      │    │
│  │ 7. Queue.put(result_dict) → 进程退出               │    │
│  └──────────────────────────────────────────────────┘    │
│  职责: 模型加载、设备初始化、推理、计时、结果序列化           │
│  禁止: 传递任何 tensor/device 对象到主进程                  │
└──────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│                  校验度量层 (Verifier)                    │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────┐ │
│  │ Precision     │  │ Statistics     │  │ Report       │ │
│  │ Verifier      │  │ Calculator     │  │ Renderer     │ │
│  └──────────────┘  └───────────────┘  └──────────────┘ │
│  职责: 精度比对、统计计算、报告渲染                         │
│  禁止: 在主进程持有任何 GPU/NPU 内存引用                    │
└─────────────────────────────────────────────────────────┘
```

### 2.1 分层的核心动机

参考 `torch.utils.benchmark` 的设计哲学：测量工具自身不能引入干扰。三层分离确保：

1. **调度管控层**是纯 Python 调度器，不引入任何设备运行时。一旦主进程初始化了 CUDA/CANN runtime，它会持有 GPU context 且无法完全释放。后续子进程可能继承或冲突该 context，导致不可预测的死锁或 OOM。
2. **沙箱执行层**是短生命周期的"一次性"进程。每个 benchmark 任务独占一个进程，用完即毁，彻底消除状态泄漏风险。
3. **校验度量层**只处理 NumPy 数值数据，不涉及任何设备端操作，确保精度比对不引入额外显存占用。

---

## 3. 模块分解

```
src/benchmark/
├── run_benchmark.py      # CLI 入口 + 编排逻辑 (Orchestrator)
├── worker.py             # 子进程沙箱 (Worker)
├── model_loader.py       # 模型加载 + 输入生成 (表驱动注册)
├── compare.py            # 精度比对 (Verifier)
└── __init__.py
```

### 3.1 模块依赖关系

```
run_benchmark.py
    ├── worker.py          (通过 spawn 子进程调用，不直接 import torch)
    ├── compare.py         (纯 NumPy，主进程内直接调用)
    └── model_loader.py    (仅在 worker 子进程内使用)

worker.py
    ├── model_loader.py    (子进程内 import)
    └── [torch, transformers]  (子进程内首次导入)

compare.py
    └── [numpy]            (无设备依赖)
```

### 3.2 设计约束

| 约束 | 原因 |
|---|---|
| `run_benchmark.py` 不得 `import torch` | 防止主进程初始化设备 runtime |
| `worker.py` 中的 torch 相关 import 必须在函数体内部 | 防止模块级 import 被 `spawn` 序列化时触发 |
| `model_loader.py` 的 `import torch` 在模块级（安全） | 因为只有 worker 子进程会 import model_loader |
| `compare.py` 不得 `import torch` | 只处理 NumPy 数据，无设备依赖 |

---

## 4. 配置设计 — 任务矩阵

### 4.1 任务矩阵生成

参考 optimum-benchmark 的复合配置（BenchmarkConfig）理念，框架从 CLI 参数生成一个笛卡尔积式的任务矩阵：

```
models × modes = tasks
```

对于 N 个模型，每个模型生成 2 个任务（eager + compile），共 2N 个任务。

### 4.2 任务 Payload Schema

主进程 → Worker 的 payload 只允许基础类型（`str | int | float | bool`），严禁传递任何复杂对象：

```python
task_payload = {
    "model_dir": str,           # 模型目录绝对路径
    "is_compile": bool,         # 是否 compile 模式
    "dtype": str,               # 数据类型 ("float32" / "float16" / "bfloat16")
    "warmup_iters": int,        # warmup 次数 (>= 3)
    "test_iters": int,          # 正式测量次数 (>= 10)
    "task": str,                # 任务类型提示 ("auto" 或具体类型)
    "inductor_backend": str,    # Inductor 后端 ("triton" / "dvm" / "mlir")
}
```

**为什么只允许基础类型**：
- `multiprocessing.Queue` 使用 pickle 序列化
- 传递 model 对象、tensor、optimizer 等会导致 pickle 失败或隐式设备同步
- 保持 payload 可序列化、可日志记录、可调试

### 4.3 Worker 结果 Schema

Worker → 主进程的结果同样只包含基础类型和 NumPy 数组：

```python
worker_result = {
    "status": str,              # SUCCESS | OOM | TIMEOUT | COMPILE_ERROR | CRASH
    "latency_ms": float,        # 平均推理延迟 (ms)
    "p99_latency_ms": float,    # P99 延迟 (ms)
    "compile_time_ms": float,   # 编译耗时 (ms)，eager 模式为 0.0
    "error_message": str,       # 错误信息 (截断到 1000 字符)
    "output": Any,              # 序列化后的 NumPy 数据 (用于精度比对) 或 None
}
```

---

## 5. Worker 生命周期协议

### 5.1 七步标准流程

每个 benchmark 任务（eager 或 compile）在独立子进程中按以下步骤执行：

```
步骤 1: import torch — 在 worker 函数体内首次导入
步骤 2: 加载模型 — AutoModel.from_pretrained() + tie_weights()
步骤 3: model.to(device).eval() — 设备转移 + 评估模式
步骤 4: make_inputs() — 在 worker 内部生成输入 (固定种子，不共享)
步骤 5: Warmup + 计时 — 编译完成后 warmup >= 3 次，然后逐次计时
步骤 6: serialize_output() — 输出移至 CPU → 转为 NumPy
步骤 7: Queue.put(result) → sys.exit(0) — 进程物理销毁
```

### 5.2 步骤详解

#### 步骤 1：延迟导入 torch

```python
def worker_main(task, result_queue):
    import torch  # 必须在函数体内，不能在模块级
```

**关键**：`import torch` 必须在 `worker_main()` 函数体内，而非模块顶层。因为 `spawn` 启动方式会重新 import 子进程模块，如果 torch 在模块级导入，子进程的模块初始化就会触发 CUDA/CANN runtime 初始化。

#### 步骤 2：模型加载

```python
from transformers import AutoConfig, AutoModelForXxx

config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
model = AutoModelForXxx.from_pretrained(
    model_dir,
    config=config,
    torch_dtype=str_to_torch_dtype(dtype_str),
    local_files_only=True,      # 禁止网络下载
    trust_remote_code=False,    # 安全默认值
)
if hasattr(model, "tie_weights"):
    model.tie_weights()
```

**表驱动设计**：每种任务类型（causal-lm、masked-lm 等）注册自己的 loader 和 input generator，新增模型类型无需修改编排逻辑：

```python
_MODEL_LOADERS: dict[str, ModelLoaderFunc] = {}
_INPUT_GENERATORS: dict[str, InputGeneratorFunc] = {}

@register_model_loader("causal-lm")
def load_causal_lm_model(model_dir, config, dtype_str): ...

@register_input_generator("causal-lm")
def make_causal_lm_inputs(config, device, dtype, seed=42): ...
```

#### 步骤 3：设备转移

```python
model = model.to(device).eval()
```

**顺序不能颠倒**：先 `.to(device)` 再 `.eval()`。必须在 CPU 上完成所有模型初始化（`tie_weights()`、meta tensor 实例化），再 `.to(device)`。

**meta tensor 处理**：某些模型加载后可能有未实例化的 meta tensor，需要显式替换：

```python
for param in model.parameters():
    if param.device.type == "meta":
        param.data = torch.empty_like(param.data, device=device)
```

#### 步骤 4：输入生成

```python
torch.manual_seed(42)
if hasattr(torch, "npu"):
    torch.npu.manual_seed_all(42)
inputs = make_inputs(config, task_type, device, dtype_str)
```

**三个硬性约束**：
1. 输入必须在 worker 内部生成，严禁跨进程传递 tensor
2. Eager 和 compile 必须各自在独立 worker 中生成输入，严禁共用
3. 随机种子固定（相同种子保证 eager/compile 输入一致，精度可比）

#### 步骤 5：Warmup + 计时

详见第 6 节"计时与度量方法论"。

#### 步骤 6：序列化输出

```python
def serialize_output(value):
    if isinstance(value, torch.Tensor):
        # 保留原始 dtype 信息，供精度比对器选择正确容差
        return {
            "__type__": "tensor",
            "data": value.detach().cpu().numpy(),
            "dtype": str(value.dtype),
        }
    if isinstance(value, ModelOutput):
        return {k: serialize_output(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {k: serialize_output(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_output(v) for v in value]
    if isinstance(value, (int, float, bool, str, type(None))):
        return value
    return str(value)
```

**为什么必须转为 NumPy**：
- PyTorch tensor 的 pickle 使用 IPC handle，跨进程后可能导致显存不释放或死锁
- NumPy array 存储在 CPU 主存中，pickle 后是纯数据，无设备端依赖

#### 步骤 7：清理与退出

```python
finally:
    try:
        del model
    except NameError:
        pass
    try:
        del inputs
    except NameError:
        pass
    gc.collect()                    # 先 GC 断开循环引用
    if device.startswith("npu"):
        torch.npu.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()    # 再释放设备缓存
    sys.exit(0)                     # 进程物理销毁，操作系统回收所有资源
```

**清理顺序不可颠倒**：`gc.collect()` 必须在 `empty_cache()` 之前。Python 的引用计数无法回收循环引用（模型中的自引用结构），只有 `gc.collect()` 才能断开这些循环。否则 `empty_cache()` 无法释放对应的设备内存。

---

## 6. 计时与度量方法论

### 6.1 计时原理

GPU/NPU 操作是异步的：`model(inputs)` 只向设备提交计算任务，CPU 立即返回。直接用 `time.perf_counter()` 包裹 `model(inputs)` 测量的是 kernel launch 时间（微秒级），而非实际设备端计算时间。

### 6.2 NPU 计时方案

NPU 场景下使用 `synchronize + perf_counter`（因为 `torch.npu.Event` 的 `enable_timing=True` 支持程度取决于 torch_npu 版本）：

```python
# 逐次计时模式（推荐）
latencies = []
with torch.no_grad():
    for _ in range(test_iters):
        torch.npu.synchronize()           # 确保前序操作完成
        t0 = time.perf_counter()
        output = model(**inputs)
        torch.npu.synchronize()           # 等待本次推理完成
        latencies.append((time.perf_counter() - t0) * 1000.0)
```

**为什么选择逐次计时而非批量计时**：
- 批量计时（一次 synchronize 前后，中间跑 N 次 forward）只能得到总时间除以 N 的均值
- 逐次计时可以得到每次推理的真实延迟分布，从而计算 P99 等分位数
- 参考 `torch.utils.benchmark.Timer` 的设计：每次测量都是独立的前后 synchronize 包裹

### 6.3 CUDA 场景计时方案

CUDA 场景下优先使用 Event 计时（精度更高，不阻塞 CPU）：

```python
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
output = model(**inputs)
end_event.record()
torch.cuda.synchronize()

latency_ms = start_event.elapsed_time(end_event)
```

### 6.4 统计指标

参考 `torch.utils.benchmark.Measurement` 的设计，报告以下指标：

| 指标 | 计算方式 | 用途 |
|---|---|---|
| **median (P50)** | `sorted_lat[len//2]` | 反映典型延迟，对长尾不敏感 |
| **mean** | `sum(latencies) / len(latencies)` | 反映总吞吐效率 |
| **P99** | `sorted_lat[int(len * 0.99)]` | 反映长尾延迟（对延迟敏感场景重要） |
| **IQR** | Q3 - Q1 | 反映测量稳定性（参考 torch.utils.benchmark 的 IQR 预警） |

**至少 10 次正式测量**（不含 warmup），才能得到有统计意义的 P99。

### 6.5 Compile 模式的特殊计时

`torch.compile` 是 lazy（JIT）编译：`torch.compile(model)` 只创建 wrapper，编译发生在 compiled model 的首次 forward pass。编译耗时可能从几秒到几分钟。

```python
with torch.no_grad():
    if is_compile:
        # 编译时间单独记录
        torch.npu.synchronize()
        t0_compile = time.perf_counter()
        _ = compiled_model(**inputs)  # 触发真正的底层编译
        torch.npu.synchronize()
        compile_time_ms = (time.perf_counter() - t0_compile) * 1000.0

        # 剩余 warmup（编译已完成）
        for _ in range(warmup_iters - 1):
            _ = compiled_model(**inputs)

    torch.npu.synchronize()  # warmup 结束后同步

    # 正式测量（不含编译时间）
    latencies = []
    for _ in range(test_iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        _ = model(**inputs)
        torch.npu.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)
```

**compile_time_ms 直接使用首次 forward 耗时，不做减法**：首次 forward = Dynamo tracing + Inductor codegen + kernel launch + 设备执行，这正是用户关心的"编译一次要等多久"。用首次 forward 减去稳态均值来估算 compile time 会产生偏差（两种测量的同步模式不同）。

### 6.6 计时区间内的禁忌操作

以下操作会隐式触发 CPU-GPU 同步，**绝对不能**出现在计时区间内：

| 操作 | 触发原因 |
|---|---|
| `print(device_tensor)` | 隐式 `.cpu()` 转换 |
| `tensor.item()` | 需要等待设备端结果 |
| `tensor.cpu()` | 显式设备到主机传输 |
| `tensor.nonzero()` | 需要 CPU 端数据 |
| `if (tensor > 0).all()` | 数据依赖的 Python 控制流 |

---

## 7. IPC 与序列化协议

### 7.1 进程间通信模型

```
主进程                              Worker 子进程
  │                                      │
  │──── spawn Process ──────────────────>│
  │                                      │
  │──── Queue ← task_payload ────────────│  (主进程不主动 put，task 通过 Process args 传递)
  │                                      │
  │                                      │  ... benchmark 执行 ...
  │                                      │
  │<─── Queue.get(timeout=T) ───────────│  (Worker put 结果)
  │                                      │
  │──── proc.join() ────────────────────>│  (等待进程退出)
  │                                      │
```

### 7.2 序列化规则

| 方向 | 允许的类型 | 禁止的类型 |
|---|---|---|
| 主进程 → Worker | `str, int, float, bool` | `torch.Tensor, model, optimizer` |
| Worker → 主进程 | `str, int, float, bool, numpy.ndarray, list, dict, None` | `torch.Tensor, torch.Device` |

### 7.3 Traceback 截断

Worker 发生异常时，traceback 必须截断到 **尾部 1000 字符**（`tb[-1000:]`），原因：

- 大型模型的 traceback 嵌套层数极深，完整 traceback 可达数万字符
- 超大 payload 会阻塞 IPC pipe，可能导致 `Queue.get(timeout=...)` 超时误判
- Python traceback 的关键信息（异常类型 + 错误消息）在最末尾

---

## 8. 精度校验设计

### 8.1 为什么 Eager 和 Compile 输出不同

`torch.compile` / Inductor 会：
- **融合算子**：将多个 elementwise 操作合并为单个 kernel，改变计算顺序
- **重排计算**：交换操作顺序，导致浮点累加顺序不同
- **精度变化**：某些 reduction 操作在 Triton kernel 中可能使用更高精度的 accumulator

这些都是合法的浮点差异，不是 bug。

### 8.2 容差标准

> 以下容差是针对 eager vs compile 比对的**实践推荐值**，并非 PyTorch 默认值（`torch.allclose` 默认 `rtol=1e-5, atol=1e-8`）。

| 数据类型 | rtol | atol | 说明 |
|---|---|---|---|
| Int / Bool | exact match | exact match | 不允许任何偏差 |
| Float32 | 1e-4 | 1e-4 | 考虑 TF32 和算子融合带来的累加顺序差异 |
| BFloat16 | 1e-2 | 1e-2 | BF16 尾数只有 7 位，舍入误差显著大于 FP32 |
| Float16 | 5e-3 | 5e-3 | FP16 尾数 10 位 |
| NaN / Inf | `equal_nan=True` | — | 相同位置的 NaN/Inf 视为相等 |

### 8.3 递归比较算法

HuggingFace 模型的输出通常是嵌套结构（`ModelOutput` / `dict` / `tuple` / `list`），需要递归遍历：

```python
def compare_node(path, left, right, summary, dtype_str):
    if isinstance(left, dict):
        for key in left:
            compare_node(f"{path}.{key}", left[key], right[key], summary, dtype_str)
    elif isinstance(left, (list, tuple)):
        for i, (lv, rv) in enumerate(zip(left, right)):
            compare_node(f"{path}[{i}]", lv, rv, summary, dtype_str)
    elif isinstance(left, np.ndarray):
        _compare_arrays(path, left, right, summary, dtype_str)
```

### 8.4 错误报告

精度失败时必须报告：
- `max_absolute_error`：最大绝对误差
- `max_relative_error`：最大相对误差
- `failing_path`：出错的张量路径（如 `root.last_hidden_state`）
- 出错张量的 shape 和 dtype

---

## 9. 错误分类与容错设计

### 9.1 错误分类体系

```
Worker 异常
├── SUCCESS           # 执行成功且精度通过
├── OOM               # 显存不足
├── TIMEOUT           # Watchdog 超时
├── COMPILE_ERROR     # 编译器错误 (Inductor/Triton/CANN 算子编译失败)
├── PRECISION_FAIL    # Eager/Compile 输出不匹配
└── CRASH             # 其他未分类异常
```

### 9.2 OOM 识别

通过 traceback 文本模式匹配识别 OOM：

```python
OOM_PATTERNS = [
    "CUDA out of memory",
    "CANN memory allocation failed",
    "OutOfMemoryError",
    "Unable to find a valid cuDNN algorithm",
]
```

### 9.3 Watchdog 超时机制

```python
try:
    result = result_queue.get(timeout=timeout_seconds)
except Exception:
    proc.terminate()              # SIGTERM，允许子进程有限清理
    proc.join(timeout=300)        # 等待清理（CUDA context 清理可能耗时）
    if proc.is_alive():
        proc.kill()               # SIGKILL 强杀
        proc.join(timeout=300)
    return {"status": "TIMEOUT", ...}
```

**分级清理**：先 SIGTERM（优雅终止）→ 等待 → 必要时 SIGKILL（强制终止）。CUDA/CANN 驱动级调用可能无法被 SIGTERM 中断，必须准备 SIGKILL。

### 9.4 COMPILE_ERROR 升级工作流

`COMPILE_ERROR` 不是普通运行时错误，它意味着编译器无法处理当前模型架构。处理流程：

1. **记录完整 compile 日志**：设置 `torch._logging.set_logs(dynamo=True, aot=True, inductor=True)`
2. **提取关键信息**：失败的算子名称、输入 shape、dtype
3. **不参与 speedup 统计**：标记为 `COMPILE_ERROR` 的结果不拉低平均 speedup
4. **在报告中单独列出**，供编译器团队排障

### 9.5 结果聚合策略

对于每个模型，编排层运行 eager 和 compile 两个独立任务，然后聚合为宽表行：

```python
final_result = {
    "model_id": str,
    "dtype": str,
    "status": str,               # 整体状态
    "eager_status": str,
    "eager_latency_ms": float,
    "eager_p99_ms": float,
    "eager_error_message": str,
    "compile_status": str,
    "compile_latency_ms": float,
    "compile_p99_ms": float,
    "compile_time_ms": float,
    "compile_error_message": str,
    "speedup": float,            # eager_latency / compile_latency
    "precision_match": bool,
    "precision_error": str,
    "error_message": str,
}
```

**整体状态判定优先级**：

```
eager 或 compile CRASH → CRASH
eager 或 compile OOM → OOM
eager 或 compile TIMEOUT → TIMEOUT
compile COMPILE_ERROR → COMPILE_ERROR
两者 SUCCESS 但精度不匹配 → PRECISION_FAIL
两者 SUCCESS 且精度匹配 → SUCCESS
```

---

## 10. Warmup 策略设计

### 10.1 一次性开销来源

| 开销来源 | 延迟量级 | 说明 |
|---|---|---|
| cuBLAS/cuDNN lazy init | ~2-3 ms | 首次 CUDA 调用时加载 |
| Triton kernel JIT 编译 | 秒级 | `torch.compile` 首次执行每个 kernel |
| NPU 算子编译 | 秒级 | CANN 算子库首次匹配 |
| 硬件频率爬升 | ~100 ms | GPU/NPU 从低功耗状态提升频率 |
| 内存分配器预热 | ~10 ms | Caching allocator 首次分配 block |

### 10.2 Warmup 参数推荐

| 场景 | warmup_iters | test_iters | 说明 |
|---|---|---|---|
| Eager 模式 | >= 3 | >= 10 | 覆盖 lazy init + 频率爬升 |
| Compile 模式 | >= 5 | >= 10 | 首次 forward 含编译，需更多 warmup |
| max-autotune 模式 | >= 10 | >= 20 | Triton 尝试多种 kernel 配置 |

### 10.3 Warmup 的硬性约束

- Warmup 和正式测量必须使用**完全相同**的 input shape 和 dtype
- 不同 shape 会触发 `torch.compile` 重新编译
- Warmup 后必须 `synchronize()` 确保所有 warmup 操作完成

---

## 11. 进程隔离设计

### 11.1 必须使用 `spawn` 启动方式

```python
ctx = multiprocessing.get_context("spawn")
```

**禁止使用 `fork`**：
- CUDA/CANN context 被子进程继承 → 死锁
- Python GIL 状态不一致
- 文件描述符泄漏

### 11.2 每个任务独占子进程

**不要**在同一个子进程中顺序执行多个模型或多种模式。原因：

| 原因 | 说明 |
|---|---|
| Dynamo 缓存污染 | `torch.compile` 修改 Dynamo 全局缓存，`torch._dynamo.reset()` 无法完全清理 |
| 显存碎片化 | 进程内多模型加载/卸载导致 caching allocator 碎片化，虚假 OOM |
| 故障隔离 | C++ 级别 segfault 会污染后续执行环境 |
| 算子缓存残留 | Triton kernel cache、CUDA graph 等底层状态无法在进程内彻底清理 |

### 11.3 串行独占执行

所有任务严格串行执行，禁止并行。原因：
- PCIe 总线带宽和显存带宽是共享资源，并行导致结果波动
- NPU/CUDA 的算子编译占用额外临时显存，并行可能导致虚假 OOM
- 设备端执行队列互相干扰，计时失真

---

## 12. 内存管理策略

### 12.1 显存泄漏的根因

CPU 侧持有 tensor 引用 → Python GC 保留引用计数 → 设备端内存不释放。这是 NPU benchmark 中最常见的坑。

### 12.2 内存清理三步法

```
1. del model, inputs, output       # 删除大对象引用
2. gc.collect()                    # 断开循环引用（关键！）
3. torch.npu.empty_cache()         # 释放设备缓存
```

**为什么 `gc.collect()` 必须在 `empty_cache()` 之前**：模型中的自引用结构（如 `self.attn = self.attn`）构成循环引用，Python 引用计数无法回收。只有 `gc.collect()` 能断开这些循环，之后 `empty_cache()` 才能释放对应的设备内存。

### 12.3 子进程退出的双保险

Worker 的 `finally` 块中执行清理三步法后，子进程 `sys.exit(0)` 退出，操作系统回收所有资源（包括 GPU/NPU 显存）。这是双保险策略：
- 正常情况：`empty_cache()` 已释放大部分显存
- 异常情况：进程退出后操作系统强制回收

---

## 13. CLI 设计

### 13.1 命令行接口

```bash
python -m benchmark.run_benchmark [OPTIONS]
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model` | (无) | 指定模型别名（model/ 目录下的子目录名），不指定则运行全部 |
| `--all` | false | benchmark 所有 models/ 下的模型 |
| `--dtype` | float32 | 数据类型 (float32 / float16 / bfloat16) |
| `--warmup-iters` | 3 | warmup 迭代次数 |
| `--test-iters` | 10 | 正式测量迭代次数 |
| `--timeout-seconds` | 600 | 每个模式（eager/compile）的 watchdog 超时（秒） |
| `--task` | auto | 任务类型提示（auto 自动推断） |
| `--inductor-backend` | triton | Inductor 后端 (triton / dvm / mlir) |

### 13.2 自动模型发现

不指定 `--model` 时，自动扫描 `model/` 目录下的所有子目录作为候选模型：

```python
def discover_models():
    model_root = project_root() / "model"
    return sorted([
        entry.name for entry in model_root.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
    ])
```

---

## 14. 结果报告与持久化

### 14.1 CSV 宽表输出

每个模型一行，包含 eager/compile 双侧指标和比对结果：

```csv
model_id,dtype,status,eager_status,eager_latency_ms,eager_p99_ms,compile_status,compile_latency_ms,compile_p99_ms,compile_time_ms,speedup,precision_match,precision_error,error_message
```

### 14.2 输出文件命名

```
output/benchmark_{dtype}_{inductor_backend}_{timestamp}.csv
```

### 14.3 退出码

| 退出码 | 含义 |
|---|---|
| 0 | 所有模型 benchmark 成功 |
| 1 | 至少一个模型失败（任何非 SUCCESS 状态） |

---

## 15. 可扩展性设计

### 15.1 新增模型类型的步骤

只需两步，无需修改编排逻辑：

**第一步**：注册 loader（`model_loader.py`）

```python
@register_model_loader("new-task-type")
def load_new_type_model(model_dir, config, dtype_str):
    model = AutoModelForNewType.from_pretrained(...)
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    return model
```

**第二步**：注册 input generator（`model_loader.py`）

```python
@register_input_generator("new-task-type")
def make_new_type_inputs(config, device, dtype, seed=42):
    torch.manual_seed(seed)
    return {"input_ids": ..., "attention_mask": ...}
```

**第三步**（可选）：在 `infer_task_type()` 中添加自动推断规则。

### 15.2 任务类型自动推断

基于 `config.json` 中的 `architectures` 和 `model_type` 字段自动推断任务类型：

```
architectures 中的关键字 → 任务类型
─────────────────────────────────────
"SequenceClassification" → sequence-classification
"CausalLM" / "LMHeadModel" → causal-lm
"MaskedLM" / "FillMask" → masked-lm
"ConditionalGeneration" → seq2seq-lm
"SpeechSeq2Seq" → speech-seq2seq
"VisionEncoderDecoder" → vision2seq
(无任务关键字且以 Model 结尾) → base
```

### 15.3 扩展 Inductor 后端

`--inductor-backend` 参数支持后端级别的扩展。当前支持：

| 后端 | 适用设备 | 说明 |
|---|---|---|
| `triton` | NVIDIA GPU | Inductor 默认 Triton 后端 |
| `dvm` | 华为昇腾 NPU | 华为自研深度向量机后端 |
| `mlir` | 华为昇腾 NPU | MLIR-based 后端 |

---

## 16. NPU（昇腾 Ascend）专项设计

### 16.1 设备操作适配

```python
def synchronize_if_torch_device(device: str) -> None:
    import torch
    if device.startswith("npu"):
        torch.npu.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()
```

### 16.2 NPU 特有注意事项

| 事项 | 说明 |
|---|---|
| Tensor 创建 | 使用 `torch.randn(shape, device="npu")` 而非 `.npu()` 方法 |
| 显存监控 | `torch.npu.memory_allocated()` / `torch.npu.memory_reserved()` |
| Event 支持 | `torch.npu.Event` 的 `enable_timing=True` 可能不完全支持，用 `synchronize + perf_counter` 替代 |
| 版本兼容 | torch_npu 版本必须与 PyTorch 版本和 CANN 版本严格匹配 |
| 环境变量 | 必须 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |

### 16.3 Compile 缓存管理

测量 compile 时间时，缓存清理必须通过**外层 shell 脚本**控制，不在 worker 内清除：

```bash
#!/bin/bash
rm -rf $TORCHINDUCTOR_CACHE_DIR
rm -rf $TRITON_CACHE_DIR
rm -rf $ASCEND_CACHE_PATH
python -m benchmark.run_benchmark --model $1 --inductor-backend $2
```

**原因**：Worker 内 `shutil.rmtree()` 的耗时取决于缓存目录大小（可能几百 MB），会被误计入 compile_time_ms。

---

## 17. 统计质量预警（参考 torch.utils.benchmark）

参考 `torch.utils.benchmark.Measurement` 的 IQR 预警机制，为 benchmark 结果增加统计质量检查：

### 17.1 测量稳定性检查

```python
def check_measurement_quality(latencies: list[float]) -> list[str]:
    warnings = []
    sorted_lat = sorted(latencies)
    q1 = sorted_lat[len(sorted_lat) // 4]
    q3 = sorted_lat[3 * len(sorted_lat) // 4]
    iqr = q3 - q1
    median = sorted_lat[len(sorted_lat) // 2]

    if iqr > 0.1 * median:
        warnings.append(f"IQR ({iqr:.2f}ms) > 10% of median ({median:.2f}ms), measurements unstable")
    if len(latencies) < 10:
        warnings.append(f"Only {len(latencies)} measurements, recommend >= 10 for reliable P99")

    return warnings
```

### 17.2 预警级别

| 预警 | 触发条件 | 含义 |
|---|---|---|
| IQR 过大 | IQR > 10% × median | 测量不稳定，可能 warmup 不够或存在后台干扰 |
| 样本不足 | test_iters < 10 | P99 不可靠 |
| Compile 比 Eager 慢 | speedup < 1.0 | 可能存在 graph break 或 backend 不匹配 |

---

## 18. 设计决策记录

| 决策 | 选择 | 替代方案 | 理由 |
|---|---|---|---|
| 进程启动方式 | `spawn` | `fork` / `forkserver` | CUDA/CANN 不支持 fork，spawn 保证干净状态 |
| 计时方式 | per-iteration synchronize | batch timing | 可得分位数分布，P99 有实际意义 |
| IPC 载体 | `multiprocessing.Queue` | `shared_memory` / `Manager` | Queue 自带序列化和同步，满足需求 |
| 输出序列化 | NumPy | pickle tensor | 避免 IPC handle 和显存泄漏 |
| 模型注册 | 装饰器 + 字典 | if-elif 链 | 开闭原则，新增类型不改编排逻辑 |
| Compile 时间 | 首次 forward 总耗时 | 首次 forward - 稳态均值 | 避免同步模式不同导致的偏差 |
| 配置传递 | CLI args | YAML / Hydra | 当前规模简单，CLI 足够；未来可扩展 |
| 报告格式 | CSV 宽表 | JSON / HTML | 方便后处理（pandas/excel），一行一模型 |

---

## 参考资料

- [PyTorch torch.utils.benchmark](https://pytorch.org/docs/stable/benchmark_utils.html) — Timer、Measurement、blocked_autorange、Compare
- [HuggingFace optimum-benchmark](https://github.com/huggingface/optimum-benchmark) — 复合配置、Launcher 隔离、Backend 策略模式
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html) — 同步、内存管理、算子融合
- [Introduction to torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) — 编译模式、计时模式
- [PyTorch Multiprocessing Notes](https://pytorch.org/docs/stable/notes/multiprocessing.html) — spawn vs fork、CUDA IPC
- [Understanding GPU Memory](https://pytorch.org/blog/understanding-gpu-memory-1/) — Caching allocator、Memory Snapshot API
- [Ascend/pytorch (torch_npu)](https://github.com/Ascend/pytorch) — 昇腾 NPU 适配器
