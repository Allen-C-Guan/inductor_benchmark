"""Model loading and input generation module.

Uses table-driven design to handle different model types.
Each model type has its own loader and input generator function.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import torch

# ============================================================================
# Type registry
# ============================================================================

ModelLoaderFunc = Callable[[str, Any, str], Any]
InputGeneratorFunc = Callable[[Any, str, str, int], dict[str, Any]]

# Registry: task_type -> loader function
_MODEL_LOADERS: dict[str, ModelLoaderFunc] = {}

# Registry: task_type -> input generator function
_INPUT_GENERATORS: dict[str, InputGeneratorFunc] = {}


def register_model_loader(task_type: str) -> Callable:
    """Decorator to register a model loader function for a task type."""

    def decorator(func: ModelLoaderFunc) -> ModelLoaderFunc:
        _MODEL_LOADERS[task_type] = func
        return func

    return decorator


def register_input_generator(task_type: str) -> Callable:
    """Decorator to register an input generator function for a task type."""

    def decorator(func: InputGeneratorFunc) -> InputGeneratorFunc:
        _INPUT_GENERATORS[task_type] = func
        return func

    return decorator


# ============================================================================
# Dtype utilities
# ============================================================================


def str_to_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch.dtype.

    Supports: float32, float16, bfloat16, fp32, fp16, bf16
    Returns torch.float32 for unknown dtypes.
    """
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str, torch.float32)


# ============================================================================
# Task type inference
# ============================================================================


def infer_task_type(model_dir: str, task_hint: str = "auto") -> str:
    """
    Infer the task type from config.json.

    Returns one of:
      - 'base' (no specific task head)
      - 'causal-lm'
      - 'seq2seq-lm'
      - 'masked-lm'
      - 'sequence-classification'
      - 'speech-seq2seq'
      - 'vision2seq'
    """
    if task_hint != "auto":
        return task_hint

    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {model_dir}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    architectures = config.get("architectures", [])
    model_type = config.get("model_type", "").lower()

    arch_str = " ".join(architectures).lower()

    # Vision Encoder-Decoder
    if model_type == "vision-encoder-decoder" or "visionencoderdecoder" in arch_str:
        return "vision2seq"


    speech_types = ("whisper", "speech_to_text", "speech_to_text_2")
    if model_type in speech_types or "speechseq2seq" in arch_str or "speech2textforconditionalgeneration" in arch_str:
        return "speech-seq2seq"

    # Sequence Classification
    if "sequenceclassification" in arch_str:
        return "sequence-classification"

    # Base Models (no task head)
    if architectures:
        has_task_head = any(
            keyword in arch
            for arch in architectures
            for keyword in ["For", "With", "Head", "ConditionalGeneration", "LM"]
        )
        if not has_task_head and arch_str.endswith("model"):
            return "base"

    # Causal LM
    causal_types = (
        "gpt2", "llama", "qwen", "qwen2", "qwen3", "mistral", "gemma", "phi", "opt", 
        "gptj" 
    )
    if "causallm" in arch_str or "lmheadmodel" in arch_str or model_type in causal_types:
        return "causal-lm"

    # Masked LM
    if "maskedlm" in arch_str or "fillmask" in arch_str:
        return "masked-lm"

    # Text Seq2Seq LM
    seq2seq_types = ("t5", "mt5", "bart", "m2m_100", "mbart", "pegasus")
    if "conditionalgeneration" in arch_str or "seq2seq" in arch_str or model_type in seq2seq_types:
        return "seq2seq-lm"


    masked_types = (
        "bert", "roberta", "xlm-roberta", "electra", "albert", 
        "deberta-v2", "mobilebert", "layoutlm",
        "deberta",    # 适配 deberta-base
        "longformer", # 适配 longformer-base-4096
        "fnet"        # 适配 fnet-base
    )
    if model_type in masked_types:
        return "masked-lm"

    raise ValueError(
        f"Cannot infer task type from config.json.\n"
        f"  architectures={architectures}\n"
        f"  model_type={model_type}\n"
        f"Please manually specify using task_hint parameter."
    )

# ============================================================================
# Vocab size extraction
# ============================================================================


def get_exact_vocab_size(config: Any) -> int:
    """
    Extract vocab_size from config.
    Raises ValueError if not found.
    """
    # Direct keys
    direct_keys = [
        "vocab_size",
        "target_vocab_size",
        "src_vocab_size",
        "word_vocab_size",
        "text_vocab_size",
    ]
    for key in direct_keys:
        if hasattr(config, key) and getattr(config, key) is not None:
            return getattr(config, key)

    # Multi-modal models (e.g., TrOCR)
    if hasattr(config, "text_config"):
        if hasattr(config.text_config, "vocab_size") and config.text_config.vocab_size is not None:
            return config.text_config.vocab_size

    # Encoder-Decoder models
    if hasattr(config, "decoder"):
        if hasattr(config.decoder, "vocab_size") and config.decoder.vocab_size is not None:
            return config.decoder.vocab_size

    raise ValueError(
        f"Cannot find vocab_size in config.\n"
        f"Config type: {type(config)}\n"
        f"Please check config.json and add the field to get_exact_vocab_size()."
    )


# ============================================================================
# Model loaders (table-driven)
# ============================================================================


@register_model_loader("base")
def load_base_model(model_dir: str, config: Any, dtype_str: str) -> Any:
    """Load base model (no task head)."""
    from transformers import AutoModel

    torch_dtype = str_to_torch_dtype(dtype_str)
    model = AutoModel.from_pretrained(
        model_dir,
        config=config,
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    return model


@register_model_loader("causal-lm")
def load_causal_lm_model(model_dir: str, config: Any, dtype_str: str) -> Any:
    """Load causal language model."""
    from transformers import AutoModelForCausalLM

    torch_dtype = str_to_torch_dtype(dtype_str)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=config,
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    return model


@register_model_loader("seq2seq-lm")
def load_seq2seq_lm_model(model_dir: str, config: Any, dtype_str: str) -> Any:
    """Load sequence-to-sequence language model."""
    from transformers import AutoModelForSeq2SeqLM

    torch_dtype = str_to_torch_dtype(dtype_str)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir,
        config=config,
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    return model


@register_model_loader("masked-lm")
def load_masked_lm_model(model_dir: str, config: Any, dtype_str: str) -> Any:
    """Load masked language model."""
    from transformers import AutoModelForMaskedLM

    torch_dtype = str_to_torch_dtype(dtype_str)
    model = AutoModelForMaskedLM.from_pretrained(
        model_dir,
        config=config,
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    return model


@register_model_loader("sequence-classification")
def load_sequence_classification_model(model_dir: str, config: Any, dtype_str: str) -> Any:
    """Load sequence classification model."""
    from transformers import AutoModelForSequenceClassification

    torch_dtype = str_to_torch_dtype(dtype_str)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        config=config,
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    return model


@register_model_loader("speech-seq2seq")
def load_speech_seq2seq_model(model_dir: str, config: Any, dtype_str: str) -> Any:
    """Load speech sequence-to-sequence model (e.g., Whisper)."""
    from transformers import AutoModelForSpeechSeq2Seq

    torch_dtype = str_to_torch_dtype(dtype_str)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_dir,
        config=config,
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    return model


@register_model_loader("vision2seq")
def load_vision2seq_model(model_dir: str, config: Any, dtype_str: str) -> Any:
    """Load vision encoder-decoder model (e.g., TrOCR)."""
    from transformers import VisionEncoderDecoderModel

    torch_dtype = str_to_torch_dtype(dtype_str)
    model = VisionEncoderDecoderModel.from_pretrained(
        model_dir,
        config=config,
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    return model


# ============================================================================
# Input generators (table-driven)
# ============================================================================


@register_input_generator("base")
def make_base_inputs(config: Any, device: str, dtype: str, seed: int = 42) -> dict[str, Any]:
    """Generate inputs for base model."""
    torch.manual_seed(seed)
    if hasattr(torch, "npu"):
        torch.npu.manual_seed_all(seed)

    tensor_dtype = str_to_torch_dtype(dtype)
    batch_size = 1
    seq_len = 128
    exact_vocab_size = get_exact_vocab_size(config)

    input_ids = torch.randint(0, exact_vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)

    return {"input_ids": input_ids, "attention_mask": attention_mask}


@register_input_generator("causal-lm")
def make_causal_lm_inputs(config: Any, device: str, dtype: str, seed: int = 42) -> dict[str, Any]:
    """Generate inputs for causal language model."""
    torch.manual_seed(seed)
    if hasattr(torch, "npu"):
        torch.npu.manual_seed_all(seed)

    tensor_dtype = str_to_torch_dtype(dtype)
    batch_size = 1
    seq_len = 128
    exact_vocab_size = get_exact_vocab_size(config)

    input_ids = torch.randint(0, exact_vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)

    return {"input_ids": input_ids, "attention_mask": attention_mask}


@register_input_generator("seq2seq-lm")
def make_seq2seq_lm_inputs(config: Any, device: str, dtype: str, seed: int = 42) -> dict[str, Any]:
    """Generate inputs for sequence-to-sequence language model."""
    torch.manual_seed(seed)
    if hasattr(torch, "npu"):
        torch.npu.manual_seed_all(seed)

    tensor_dtype = str_to_torch_dtype(dtype)
    batch_size = 1
    seq_len = 128
    exact_vocab_size = get_exact_vocab_size(config)

    input_ids = torch.randint(0, exact_vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)

    decoder_seq_len = 32
    decoder_input_ids = torch.randint(
        0, exact_vocab_size, (batch_size, decoder_seq_len), device=device
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "decoder_input_ids": decoder_input_ids,
    }


@register_input_generator("masked-lm")
def make_masked_lm_inputs(config: Any, device: str, dtype: str, seed: int = 42) -> dict[str, Any]:
    """Generate inputs for masked language model."""
    torch.manual_seed(seed)
    if hasattr(torch, "npu"):
        torch.npu.manual_seed_all(seed)

    batch_size = 1
    seq_len = 128
    exact_vocab_size = get_exact_vocab_size(config)

    # 1. 永远需要生成的必选项：input_ids
    input_ids = torch.randint(0, exact_vocab_size, (batch_size, seq_len), device=device)
    
    inputs = {"input_ids": input_ids}

    # 2. 动态判断：检查模型类型是否为 FNet
    model_type = getattr(config, "model_type", "").lower()
    
    if model_type != "fnet":
        # 如果不是 FNet（即普通的 BERT, RoBERTa 等），则正常添加 attention_mask
        attention_mask = torch.ones(
            (batch_size, seq_len), 
            dtype=torch.long, 
            device=device
        )
        inputs["attention_mask"] = attention_mask
    else:
        # 如果是 FNet，记录一条日志（可选），并且不去组装 attention_mask
        print(f"    [Input Gen] Detected FNet architecture, skipping attention_mask.")

    return inputs


@register_input_generator("sequence-classification")
def make_sequence_classification_inputs(config: Any, device: str, dtype: str, seed: int = 42) -> dict[str, Any]:
    """Generate inputs for sequence classification model."""
    torch.manual_seed(seed)
    if hasattr(torch, "npu"):
        torch.npu.manual_seed_all(seed)

    tensor_dtype = str_to_torch_dtype(dtype)
    batch_size = 1
    seq_len = 128
    exact_vocab_size = get_exact_vocab_size(config)

    input_ids = torch.randint(0, exact_vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)

    return {"input_ids": input_ids, "attention_mask": attention_mask}


@register_input_generator("speech-seq2seq")
def make_speech_seq2seq_inputs(config: Any, device: str, dtype: str, seed: int = 42) -> dict[str, Any]:
    """Generate inputs for speech sequence-to-sequence model (e.g., Whisper, S2T)."""
    torch.manual_seed(seed)
    if hasattr(torch, "npu"):
        torch.npu.manual_seed_all(seed)

    tensor_dtype = str_to_torch_dtype(dtype)
    batch_size = 1
    exact_vocab_size = get_exact_vocab_size(config)

    # 兼容不同配置文件的特征维度命名（S2T 可能叫 input_feat_per_channel）
    num_mel_bins = getattr(config, "num_mel_bins", getattr(config, "input_feat_per_channel", 80))
    model_type = getattr(config, "model_type", "").lower()

    if model_type == "whisper":
        audio_seq_len = 3000
        # Whisper 期望形状: (batch, features, sequence)
        feature_shape = (batch_size, num_mel_bins, audio_seq_len)
        
    elif model_type in ["speech_to_text", "speech_to_text_2"]:
        # S2T 默认 max_source_positions 通常为 6000
        audio_seq_len = getattr(config, "max_source_positions", 6000)
        # S2T 期望形状: (batch, sequence, features)
        feature_shape = (batch_size, audio_seq_len, num_mel_bins)
        
    else:
        audio_seq_len = getattr(config, "max_source_positions", 3000)
        feature_shape = (batch_size, num_mel_bins, audio_seq_len)

    input_features = torch.randn(
        feature_shape,
        device=device,
        dtype=tensor_dtype,
    )

    decoder_seq_len = 32
    decoder_input_ids = torch.randint(
        0, exact_vocab_size, (batch_size, decoder_seq_len), device=device
    )

    return {"input_features": input_features, "decoder_input_ids": decoder_input_ids}

@register_input_generator("vision2seq")
def make_vision2seq_inputs(config: Any, device: str, dtype: str, seed: int = 42) -> dict[str, Any]:
    """Generate inputs for vision encoder-decoder model (e.g., TrOCR)."""
    torch.manual_seed(seed)
    if hasattr(torch, "npu"):
        torch.npu.manual_seed_all(seed)

    tensor_dtype = str_to_torch_dtype(dtype)
    batch_size = 1
    exact_vocab_size = get_exact_vocab_size(config)

    # TrOCR default image size
    image_size = 384
    if hasattr(config, "encoder") and hasattr(config.encoder, "image_size"):
        image_size = config.encoder.image_size

    pixel_values = torch.randn(
        (batch_size, 3, image_size, image_size),
        device=device,
        dtype=tensor_dtype,
    )

    decoder_seq_len = 32
    decoder_input_ids = torch.randint(
        0, exact_vocab_size, (batch_size, decoder_seq_len), device=device
    )

    return {"pixel_values": pixel_values, "decoder_input_ids": decoder_input_ids}


# ============================================================================
# Public API
# ============================================================================


def load_model(model_dir: str, task_hint: str, dtype_str: str) -> tuple[Any, Any, str]:
    """Load a model, automatically inferring the task type.

    Args:
        model_dir: Path to model directory.
        task_hint: Task type hint (e.g., 'causal-lm', 'masked-lm', or 'auto' for auto-detection).
        dtype_str: Data type string (e.g., 'float32').

    Returns:
        Tuple of (model, config, task_type).
    """
    from transformers import AutoConfig

    # Infer task type internally
    task_type = infer_task_type(model_dir, task_hint)

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)

    if task_type not in _MODEL_LOADERS:
        raise ValueError(
            f"No loader registered for task_type='{task_type}'. "
            f"Available types: {list(_MODEL_LOADERS.keys())}"
        )

    loader = _MODEL_LOADERS[task_type]
    return loader(model_dir, config, dtype_str), config, task_type


def make_inputs(config: Any, task_type: str, device: str, dtype: str, seed: int = 42) -> dict[str, Any]:
    """Generate inputs using the registered generator for the given task type.

    Args:
        config: Model config.
        task_type: Task type (e.g., 'causal-lm', 'masked-lm').
        device: Device string (e.g., 'npu:0').
        dtype: Data type string (e.g., 'float32').
        seed: Random seed for reproducibility.

    Returns:
        Dict of inputs suitable for model(**inputs).
    """
    if task_type not in _INPUT_GENERATORS:
        raise ValueError(
            f"No input generator registered for task_type='{task_type}'. "
            f"Available types: {list(_INPUT_GENERATORS.keys())}"
        )

    generator = _INPUT_GENERATORS[task_type]
    return generator(config, device, dtype, seed)
