## Context

需要在 `src/benchmark/` 下新增一个尽量简单但正确的 benchmark 脚本，用本地 `model/<alias>/` 平铺模型目录做推理基准，并且严格遵守仓库里的 benchmark 架构约束：主进程只做调度与超时看门狗，真正的模型加载/设备初始化/推理都放到 `spawn` 子进程里完成。用户还要求通过参数选择后端，只支持 `cpu` 和华为 `npu`。

## Recommended approach

### 1. 新增文件

- `src/benchmark/__init__.py`
- `src/benchmark/run_benchmark.py`
- `src/benchmark/worker.py`
- `src/benchmark/compare.py`
- `tests/test_benchmark_compare.py`
- `tests/test_benchmark_run.py`

### 2. CLI 设计

只做一个简单入口：`python -m src.benchmark.run_benchmark`

建议参数：

- `--model`：必填，`model/<alias>/` 的目录名
- `--backend`：必填，`cpu` 或 `npu`
- `--dtype`：可选，默认 `float32`
- `--warmup-iters`：可选，默认 `3`
- `--test-iters`：可选，默认 `10`
- `--timeout-seconds`：可选，默认 `600`
- `--task`：可选，`auto|causal-lm|seq2seq-lm|masked-lm`，默认 `auto`

### 3. 主进程职责（只做调度）

在 `src/benchmark/run_benchmark.py` 中实现：

- 解析参数
- 解析仓库根目录与 `model/<alias>/`
- 构造只包含基础类型的 task payload
- 使用 `multiprocessing.get_context("spawn")`
- 严格串行启动两个短生命周期子进程：
  1. eager 模式
  2. compile 模式
- 用 `Queue.get(timeout=...)` 做看门狗
- 若超时，归类为 `TIMEOUT`
- 收集两个子进程返回的 CPU/NumPy 结果
- 调用精度比较逻辑
- 生成统一宽表结果：`status / eager_latency_ms / compile_latency_ms / speedup / compile_p99_ms / precision_match / error_message`

建议主文件中的关键函数：

- `parse_args()`
- `project_root()`（复用现有 `src/model_download/download_hf_model.py` 的模式）
- `resolve_model_dir(alias)`
- `build_task_payload(args, is_compile)`
- `run_mode_with_timeout(task, timeout_seconds)`
- `assemble_final_result(eager_result, compile_result, compare_result)`
- `main()`

### 4. Worker 职责（真正执行 benchmark）

在 `src/benchmark/worker.py` 中实现：

- `worker_main(task, result_queue)` 作为子进程入口
- 进入子进程后第一件事 `import torch`
- 再导入 `transformers`
- 从本地目录加载模型
- 如果模型有 `tie_weights()`，就调用
- `model.to(device).eval()`
- 在 worker 内生成新输入，绝不从主进程传张量
- 预热 >= 3 次
- timing 前后做设备同步
- eager / compile 各自独立执行
- 输出搬到 CPU，并转换为可跨进程传输的纯 Python / NumPy 结构
- 在 `finally` 里做 `empty_cache`

建议关键函数：

- `worker_main(task, result_queue)`
- `infer_task_type(model_dir)`
- `load_model_from_disk(model_dir, task_type, dtype_str)`
- `make_inputs(config, task_type, device)`
- `synchronize_if_needed(torch_module, backend)`
- `serialize_output(value)`
- `cleanup_device(torch_module, backend)`
- `classify_exception(traceback_text, is_compile)`

### 5. 模型加载策略（KISS）

直接把 `model/<alias>/` 当作 Hugging Face 本地模型目录使用，不做复杂适配。

推荐流程：

- `AutoConfig.from_pretrained(model_dir, local_files_only=True)`
- `--task auto` 时根据 config 推断任务类型
- 仅支持三类自动加载：
  - `AutoModelForCausalLM`
  - `AutoModelForSeq2SeqLM`
  - `AutoModelForMaskedLM`

这样可以覆盖当前仓库里最可能的模型：

- Qwen -> causal lm
- T5 -> seq2seq lm
- BERT -> masked lm

限制明确保留：

- 不尝试支持所有 Hugging Face 架构
- 初版不启用 `trust_remote_code=True`
- 如果本地目录缺关键文件，直接明确失败并返回错误信息

### 6. 输入生成策略（保持简单）

为了避免 tokenizer/processor 复杂度，使用 worker 内合成输入：

- 文本模型统一生成固定 shape 的 `input_ids` 和 `attention_mask`
- causal/masked/seq2seq 分别按最小必需字段构造
- 不从主进程传 tensor
- 不做真实文本预处理

这更符合用户要求的 KISS，也符合 benchmark notes 的“一次性数据生成”。

### 7. compile 路径策略

保持最简单实现：

- eager：直接 `model(**inputs)`
- compile：若 `torch.compile` 可用，则 `compiled_model = torch.compile(model)` 后执行
- 如果当前环境（尤其是 NPU）不支持 compile，则在 worker 中标记 `COMPILE_ERROR`

不额外加入 backend-specific compile 选项或复杂优化开关。

### 8. 精度比较

在 `src/benchmark/compare.py` 中实现纯函数比较器：

- 递归支持 dict / list / tuple / Hugging Face 风格输出结构
- int/bool 必须完全相等
- float32: `rtol=1e-4, atol=1e-4`
- bfloat16: `rtol=1e-2, atol=1e-2`
- `equal_nan=True`
- 失败时报告：
  - failing path
  - max absolute error
  - max relative error

建议函数：

- `compare_outputs(expected, actual, dtype_str)`
- `compare_node(path, left, right, summary)`
- `tensor_tolerances(dtype_name)`

### 9. 状态与错误归类

遵循 notes：

- worker 内部可返回：`SUCCESS`, `OOM`, `CRASH`, `COMPILE_ERROR`
- 主进程超时映射为：`TIMEOUT`
- eager/compile 都成功但精度失败时，最终状态为：`PRECISION_FAIL`
- `OOM` 通过 traceback 文本匹配：
  - `CUDA out of memory`
  - `CANN memory allocation failed`

最终输出 schema 统一为宽表，包含：

- `model_id`
- `dtype`
- `status`
- `eager_latency_ms`
- `compile_latency_ms`
- `speedup`
- `compile_p99_ms`
- `precision_match`
- `error_message`

### 10. 测试策略

不做真实硬件端到端测试，避免脆弱。

新增：

- `tests/test_benchmark_compare.py`
  - 覆盖 int/bool 精确比较
  - float32 / bfloat16 容差
  - 嵌套结构遍历
  - NaN/Inf 对齐情况
  - 错误路径与误差统计
- `tests/test_benchmark_run.py`
  - payload 只含基础类型
  - `Queue.get(timeout=...)` 超时映射 `TIMEOUT`
  - eager/compile 成功 + compare 成功 -> `SUCCESS`
  - eager/compile 成功 + compare 失败 -> `PRECISION_FAIL`
  - 错误文本映射 `OOM` / `COMPILE_ERROR`

## Critical files to modify

- `src/benchmark/__init__.py`
- `src/benchmark/run_benchmark.py`
- `src/benchmark/worker.py`
- `src/benchmark/compare.py`
- `tests/test_benchmark_compare.py`
- `tests/test_benchmark_run.py`

## Existing code/patterns to reuse

- `src/model_download/download_hf_model.py`
  - 复用其 `project_root()` 风格来定位仓库根目录和 `model/`
- `benchmark notes.md`
  - 作为 benchmark 架构与状态机的硬约束
- `tests/test_download_hf_model.py`
  - 复用其 pytest + monkeypatch 风格做 orchestration 级测试

## Verification

1. 运行单元测试：`python -m pytest tests/test_benchmark_compare.py tests/test_benchmark_run.py`
2. 运行全量测试：`python -m pytest`
3. 运行 lint：`python -m ruff check .`
4. 手工执行脚本验证参数与结果 schema：
   - `python -m src.benchmark.run_benchmark --model t5_small --backend cpu`
   - `python -m src.benchmark.run_benchmark --model qwen3_0_6b --backend npu`
5. 重点检查：
   - 主进程不导入 `torch`
   - 子进程使用 `spawn`
   - 超时能正确返回 `TIMEOUT`
   - 输出字段完整且 `speedup = eager / compile`