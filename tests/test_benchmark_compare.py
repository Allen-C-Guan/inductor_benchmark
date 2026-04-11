"""Tests for the precision verifier (src/benchmark/compare.py)."""

from __future__ import annotations

import numpy as np

from src.benchmark.compare import compare_outputs, tensor_tolerances


class TestTensorTolerances:
    def test_float32(self):
        rtol, atol = tensor_tolerances("float32")
        assert rtol == 1e-4
        assert atol == 1e-4

    def test_bfloat16(self):
        rtol, atol = tensor_tolerances("bfloat16")
        assert rtol == 1e-2
        assert atol == 1e-2

    def test_float16(self):
        rtol, atol = tensor_tolerances("float16")
        assert rtol == 1e-2
        assert atol == 1e-2

    def test_default(self):
        rtol, atol = tensor_tolerances("int64")
        assert rtol == 1e-4
        assert atol == 1e-4


class TestExactMatch:
    def test_identical_arrays(self):
        a = np.array([1, 2, 3], dtype=np.int64)
        result = compare_outputs(a, a.copy(), "float32")
        assert result.precision_match is True

    def test_bool_exact(self):
        a = np.array([True, False, True])
        b = np.array([True, False, True])
        result = compare_outputs(a, b, "float32")
        assert result.precision_match is True

    def test_bool_mismatch(self):
        a = np.array([True, False, True])
        b = np.array([True, True, True])
        result = compare_outputs(a, b, "float32")
        assert result.precision_match is False
        assert "Exact mismatch" in result.error_message

    def test_int_mismatch(self):
        a = np.array([1, 2, 3], dtype=np.int32)
        b = np.array([1, 9, 3], dtype=np.int32)
        result = compare_outputs(a, b, "float32")
        assert result.precision_match is False


class TestFloat32Tolerance:
    def test_within_tolerance(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([1.0 + 1e-5, 2.0 + 1e-5, 3.0 + 1e-5], dtype=np.float32)
        result = compare_outputs(a, b, "float32")
        assert result.precision_match is True

    def test_outside_tolerance(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([1.1, 2.1, 3.1], dtype=np.float32)
        result = compare_outputs(a, b, "float32")
        assert result.precision_match is False
        assert "max_abs_error" in result.error_message


class TestBFloat16Tolerance:
    def test_bf16_within_tolerance(self):
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([1.005, 2.005], dtype=np.float32)
        result = compare_outputs(a, b, "bfloat16")
        assert result.precision_match is True

    def test_bf16_outside_tolerance(self):
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([1.5, 2.5], dtype=np.float32)
        result = compare_outputs(a, b, "bfloat16")
        assert result.precision_match is False


class TestNaNInf:
    def test_aligned_nan(self):
        a = np.array([1.0, float("nan"), 3.0])
        b = np.array([1.0, float("nan"), 3.0])
        result = compare_outputs(a, b, "float32")
        assert result.precision_match is True

    def test_aligned_inf(self):
        a = np.array([float("inf"), 1.0])
        b = np.array([float("inf"), 1.0])
        result = compare_outputs(a, b, "float32")
        assert result.precision_match is True

    def test_misaligned_nan(self):
        a = np.array([1.0, float("nan"), 3.0])
        b = np.array([1.0, 2.0, 3.0])
        result = compare_outputs(a, b, "float32")
        assert result.precision_match is False


class TestNestedStructures:
    def test_dict_comparison(self):
        a = {"logits": np.array([1.0, 2.0])}
        b = {"logits": np.array([1.0, 2.0])}
        result = compare_outputs(a, b, "float32")
        assert result.precision_match is True

    def test_list_comparison(self):
        a = [np.array([1.0]), np.array([2.0])]
        b = [np.array([1.0]), np.array([2.0])]
        result = compare_outputs(a, b, "float32")
        assert result.precision_match is True

    def test_nested_dict_list(self):
        a = {"past_key_values": [{"key": np.array([1.0, 2.0])}]}
        b = {"past_key_values": [{"key": np.array([1.0, 2.0])}]}
        result = compare_outputs(a, b, "float32")
        assert result.precision_match is True

    def test_nested_mismatch_reports_path(self):
        a = {"layer_0": {"key": np.array([1.0, 2.0])}}
        b = {"layer_0": {"key": np.array([1.0, 9.0])}}
        result = compare_outputs(a, b, "float32")
        assert result.precision_match is False
        assert "layer_0.key" in result.error_message


class TestErrorReporting:
    def test_shape_mismatch(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0])
        result = compare_outputs(a, b, "float32")
        assert result.precision_match is False
        assert "Shape mismatch" in result.error_message

    def test_type_mismatch(self):
        a = np.array([1.0])
        b = "not_a_tensor"
        result = compare_outputs(a, b, "float32")
        assert result.precision_match is False
        assert "Type mismatch" in result.error_message

    def test_dict_key_mismatch(self):
        a = {"logits": np.array([1.0])}
        b = {"loss": np.array([1.0])}
        result = compare_outputs(a, b, "float32")
        assert result.precision_match is False
        assert "Dict key mismatch" in result.error_message

    def test_sequence_length_mismatch(self):
        a = [np.array([1.0]), np.array([2.0])]
        b = [np.array([1.0])]
        result = compare_outputs(a, b, "float32")
        assert result.precision_match is False
        assert "Sequence length mismatch" in result.error_message


class TestErrorPathAndStats:
    def test_reports_max_abs_and_rel_error(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.5, 3.0])
        result = compare_outputs(a, b, "float32")
        assert result.precision_match is False
        assert "max_abs_error" in result.error_message
        assert "max_rel_error" in result.error_message
