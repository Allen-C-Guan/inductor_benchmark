"""Precision verifier for benchmark outputs.

Recursively compares nested dict/list/tuple/tensor structures with
dtype-specific tolerances and detailed error reporting.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def tensor_tolerances(dtype_name: str) -> tuple[float, float]:
    """Return (rtol, atol) for a given dtype string."""
    dtype_lower = dtype_name.lower()
    if "float32" in dtype_lower or "float" == dtype_lower:
        return 1e-4, 1e-4
    if "bfloat16" in dtype_lower or "bf16" in dtype_lower:
        return 1e-2, 1e-2
    if "float16" in dtype_lower or "fp16" in dtype_lower or "half" in dtype_lower:
        return 1e-2, 1e-2
    # Default: strict
    return 1e-4, 1e-4


def _is_tensor_like(value: Any) -> bool:
    return isinstance(value, (np.ndarray,))


def _is_exact_dtype(value: Any) -> bool:
    """Check if a tensor-like holds int/bool data that requires exact match."""
    if isinstance(value, np.ndarray):
        return np.issubdtype(value.dtype, np.integer) or np.issubdtype(value.dtype, np.bool_)
    return False


class CompareSummary:
    """Accumulates comparison results across a recursive traversal."""

    def __init__(self) -> None:
        self.precision_match = True
        self.error_message = ""

    def record_fail(self, path: str, max_abs: float, max_rel: float) -> None:
        self.precision_match = False
        self.error_message = (
            f"Precision mismatch at '{path}': "
            f"max_abs_error={max_abs:.6e}, max_rel_error={max_rel:.6e}"
        )


def compare_node(
    path: str,
    left: Any,
    right: Any,
    summary: CompareSummary,
    dtype_str: str = "float32",
) -> None:
    """Recursively compare two nodes, updating *summary* on mismatch."""
    if not summary.precision_match:
        return

    if type(left) is not type(right):
        # Try to handle tuple/list interop
        if not (isinstance(left, (list, tuple)) and isinstance(right, (list, tuple))):
            summary.record_fail(path, float("nan"), float("nan"))
            summary.error_message = (
                f"Type mismatch at '{path}': {type(left).__name__} vs {type(right).__name__}"
            )
            return

    if isinstance(left, dict):
        if set(left.keys()) != set(right.keys()):
            summary.record_fail(path, float("nan"), float("nan"))
            summary.error_message = (
                f"Dict key mismatch at '{path}': {set(left.keys())} vs {set(right.keys())}"
            )
            return
        for key in left:
            compare_node(f"{path}.{key}", left[key], right[key], summary, dtype_str)

    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            summary.record_fail(path, float("nan"), float("nan"))
            summary.error_message = (
                f"Sequence length mismatch at '{path}': {len(left)} vs {len(right)}"
            )
            return
        for i, (lv, rv) in enumerate(zip(left, right)):
            compare_node(f"{path}[{i}]", lv, rv, summary, dtype_str)

    elif _is_tensor_like(left):
        _compare_arrays(path, left, right, summary, dtype_str)

    elif isinstance(left, (int, bool, str, float)):
        if left != right:
            if not (isinstance(left, float) and isinstance(right, float)):
                summary.record_fail(path, float(abs(left - right)), float("inf"))
            else:
                rtol, atol = tensor_tolerances(dtype_str)
                if not math.isclose(left, right, rel_tol=rtol, abs_tol=atol):
                    summary.record_fail(path, abs(left - right), _rel_err(left, right))
    else:
        # Fallback: direct equality
        if left != right:
            summary.record_fail(path, float("nan"), float("nan"))
            summary.error_message = f"Unsupported type at '{path}': {type(left).__name__}"


def _rel_err(a: float, b: float) -> float:
    denom = max(abs(a), abs(b))
    if denom == 0:
        return float("inf")
    return abs(a - b) / denom


def _compare_arrays(
    path: str,
    left: np.ndarray,
    right: np.ndarray,
    summary: CompareSummary,
    dtype_str: str,
) -> None:
    if left.shape != right.shape:
        summary.record_fail(path, float("nan"), float("nan"))
        summary.error_message = f"Shape mismatch at '{path}': {left.shape} vs {right.shape}"
        return

    if _is_exact_dtype(left):
        if not np.array_equal(left, right, equal_nan=False):
            summary.record_fail(path, float("nan"), float("nan"))
            summary.error_message = f"Exact mismatch at '{path}': int/bool tensors differ"
        return

    rtol, atol = tensor_tolerances(dtype_str)
    # Check with tolerances; aligned NaN/Inf count as equal
    close = np.isclose(left, right, rtol=rtol, atol=atol, equal_nan=True)
    if not np.all(close):
        abs_err = np.abs(left.astype(np.float64) - right.astype(np.float64))
        rel_denom = np.maximum(np.abs(right.astype(np.float64)), abs_err)
        rel_err = np.where(rel_denom > 0, abs_err / rel_denom, abs_err)
        # Mask out where both are inf/nan (those are considered equal)
        both_special = ~np.isfinite(left) & ~np.isfinite(right) & (np.sign(left) == np.sign(right))
        if not np.all(both_special | close):
            max_abs = float(np.max(abs_err))
            max_rel = float(np.max(rel_err))
            summary.record_fail(path, max_abs, max_rel)


def compare_outputs(expected: Any, actual: Any, dtype_str: str = "float32") -> CompareSummary:
    """Top-level comparison entry point.

    Returns a CompareSummary with precision_match and optional error_message.
    """
    summary = CompareSummary()
    compare_node("root", expected, actual, summary, dtype_str)
    return summary
