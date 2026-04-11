#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

MODEL_REGISTRY = {
    "t5_small": "google-t5/t5-small",
    "trocr_for_causal_lm": "microsoft/trocr-base-handwritten",
    "bert_for_masked_lm": "google-bert/bert-base-uncased",
    "qwen3_0_6b": "Qwen/Qwen3-0.6B",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_snapshot_download():
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError:
        print("缺少依赖 huggingface_hub，请先执行: pip install huggingface_hub", file=sys.stderr)
        raise SystemExit(1)

    return snapshot_download


EXCLUDE_FILES = [
    "*.msgpack",
    "*.h5",
    "tf_model.h5",
    "flax_model.msgpack",
    "*.gguf",  # 排除量化版（除非你专门测 GGUF）
    "*.pdf",  # 排除论文等说明文件
    "onnx/",  # 排除整个 ONNX 文件夹
]


def download_all_models() -> None:
    snapshot_download = load_snapshot_download()
    model_root = project_root() / "model"

    for alias, repo_id in MODEL_REGISTRY.items():
        target_dir = model_root / alias
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"开始下载 {repo_id} -> {target_dir}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir),
            ignore_patterns=EXCLUDE_FILES,
            local_dir_use_symlinks=False,
            resume_download=True,
        )


def main() -> int:
    download_all_models()
    print("全部模型下载完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
