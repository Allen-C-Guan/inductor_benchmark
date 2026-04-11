
# benchmark

### 标准benchmark流程



## 2. 系统总体架构 (System Architecture)

系统采用经典的 **Master-Worker（主控-沙箱）无状态架构**。严禁在主进程中引入任何硬件运行时（CUDA/CANN）或深度学习框架的全局状态。

架构分为三层：

1. **调度管控层 (Orchestrator Layer)**：运行在主进程。负责解析测试矩阵、分发任务、设置超时看门狗（Watchdog）、聚合报告。
2. **无菌沙箱层 (Isolated Sandbox Layer)**：运行在子进程（`spawn` 模式启动）。每个任务（如某模型在 Eager 下的测试）独占一个短暂的子进程，用完即毁。
3. **校验与度量层 (Metrics & Verifier Layer)**：运行在主进程。负责跨进程数据（NumPy）的重建、严苛的精度比对以及统计学计算。

------

## 3. 核心模块详细设计 (Detailed Design)

### 3.1 进程调度与防灾模块 (Dispatcher & Watchdog)

**设计原则**：假设任何涉及硬件底层调用的代码都会遭遇不可捕获的内核级崩溃（Core Dump）。

- **启动隔离**：系统入口强制执行 `multiprocessing.set_start_method('spawn', force=True)`，彻底阻断 C++ 硬件上下文的 Fork 污染。
- **超时看门狗 (Watchdog)**：主进程在调用 `Queue.get(timeout=T)` 时必须设置时间阈值。如果子进程底层驱动死锁（Deadlock），主进程将记录 `TIMEOUT_CRASH` 并强行回收资源。
- **串行独占**：为避免 PCIe 总线与显存带宽争抢，测试队列（Eager 测速 -> Compile 测速）必须**严格串行执行**。

### 3.2 沙箱工作者模块 (Isolated Worker Lifecycle)

这是系统中最脆弱但也最核心的部分。Worker 的生命周期被严格定义为以下 7 个标准动作：

1. **环境初始化**：进入子进程后，首次 `import torch`，确保 Dynamo 缓存为出厂状态。
2. **模型装载与净化**：
   - 从磁盘/HF Hub 加载模型。
   - 触发 `tie_weights()` 绑定共享内存。
   - 扫描并强制实例化所有残留的 `Meta` Tensor。
3. **硬件转移**：调用 `model.to(device).eval()`。
4. **一次性数据生成**：调用工厂函数生成新鲜的输入数据。**严禁复用数据，严禁从主进程传递张量。**
5. **高精度测速闭环**：
   - **预热 (Warmup)**：至少 3 次前向传播，触发 JIT 编译并稳定硬件频率。
   - **硬件同步**：在 `perf_counter` 前后，强制调用硬件队列同步 API（如 `torch.npu.synchronize()`）。
6. **防死锁序列化 (IPC De-coupling)**：
   - 将模型 Output 移至 CPU。
   - **强制转换为 NumPy 字节流**，彻底切断与 PyTorch 文件描述符（File Descriptor）及共享内存的绑定，防止跨进程 Queue 瘫痪。
7. **物理毁灭**：将包含 NumPy 数据的 Dict 推入 Queue，调用 `sys.exit(0)`，将显存回收权交还给 Linux 内核。

### 3.3 精度校验器 (Precision Verifier)

**设计原则**：不信任任何浮点数比对，必须区分硬件越界与舍入误差。

- **递归解析树**：支持对 HuggingFace 复杂的 `ModelOutput`、嵌套 Dict/Tuple/List 进行深度优先遍历。
- **分类隔离容差**：
  - Integer / Boolean 标签矩阵：要求绝对相等（`equal`）。
  - Float32 张量：阈值严格（`rtol=1e-4`, `atol=1e-4`）。
  - BFloat16 张量：由于尾数截断机制，阈值必须放宽（`rtol=1e-2`, `atol=1e-2`）。
- **抗 NaN/Inf 干扰**：如果基准输出与编译输出在相同索引处同时出现合法的 NaN/Inf，校验器应视为匹配（`equal_nan=True`）。
- **深度诊断**：发生错误时，不仅返回 `False`，必须抛出 `Max Absolute Error`、`Max Relative Error` 以及出错张量的路径（如 `past_key_values.layer_3.key`），以供算子开发者定位定界。

------

## 4. 接口与数据流规约 (Data Flow & Interfaces)

### 4.1 任务描述输入 (Task Payload)

主进程传递给 Worker 的配置必须是纯基础类型（String, Int, Boolean），严禁传递复杂的实例对象。

```
{
  "task_id": "llama-1b-test",
  "repo_id": "meta-llama/Llama-3.2-1B",
  "is_compile": true,
  "device_str": "npu",
  "dtype_str": "bfloat16",
  "warmup_iters": 5,
  "test_iters": 50
}
```

### 4.2 标准化报告输出 (Result Schema)

压测结束后，系统输出具有高度确定性的宽表数据，可直接对接到前端大盘或 CI/CD (Jenkins/GitLab) 流水线。

```
{
  "model_id": "Llama-3.2-1B",
  "dtype": "bfloat16",
  "status": "SUCCESS",            // 可选值: SUCCESS, OOM, TIMEOUT, PRECISION_FAIL, CRASH
  "eager_latency_ms": 25.40,
  "compile_latency_ms": 12.10,
  "speedup": 2.10,                // 严格遵守: eager_latency / compile_latency
  "compile_p99_ms": 12.50,        // 反映长尾延迟抖动
  "precision_match": true,
  "error_message": ""             // 如果失败，保留最后 1000 字符的 Traceback
}
```

------

## 5. 异常处理与容灾机制 (Error Handling)

1. **OOM (Out of Memory) 翻译**： 子进程捕获到异常时，需分析 `Traceback` 的字符串。如果是分配显存失败（`CUDA out of memory` / `CANN memory allocation failed`），需将系统状态强制置为 `OOM`，而不是模糊的 `FAIL`。
2. **算子库降级与 Panic 追踪**： 当底层 Triton 内核生成失败或 NPU 算子图融合失败时，记录特殊的 `COMPILE_ERROR`，此类错误通常表明编译器规则不支持当前模型架构，需提报给编译器研发团队。
3. **残留资源清理**： 即使 Python 发生了异常，`finally` 块中必须包含跨设备的空闲显存释放指令（`empty_cache`），尽人事听天命，最终由操作系统兜底。



