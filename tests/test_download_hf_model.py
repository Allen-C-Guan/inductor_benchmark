from pathlib import Path

import pytest

from src.model_download import download_hf_model


def test_project_root_points_to_repo_root() -> None:
    expected_root = Path(__file__).resolve().parents[1]

    assert download_hf_model.project_root() == expected_root


def test_load_snapshot_download_exits_when_dependency_missing(monkeypatch, capsys) -> None:
    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise ModuleNotFoundError("No module named 'huggingface_hub'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(SystemExit) as excinfo:
        download_hf_model.load_snapshot_download()

    assert excinfo.value.code == 1
    assert "缺少依赖 huggingface_hub" in capsys.readouterr().err


def test_download_all_models_calls_snapshot_download_for_each_registry_entry(
    monkeypatch, tmp_path
) -> None:
    calls: list[dict[str, object]] = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(download_hf_model, "load_snapshot_download", lambda: fake_snapshot_download)
    monkeypatch.setattr(download_hf_model, "project_root", lambda: tmp_path)
    monkeypatch.setattr(
        download_hf_model,
        "MODEL_REGISTRY",
        {
            "alpha": "org/model-alpha",
            "beta": "org/model-beta",
        },
    )

    download_hf_model.download_all_models()

    assert [call["repo_id"] for call in calls] == ["org/model-alpha", "org/model-beta"]
    assert [Path(call["local_dir"]) for call in calls] == [
        tmp_path / "model" / "alpha",
        tmp_path / "model" / "beta",
    ]
    assert all(call["ignore_patterns"] == download_hf_model.EXCLUDE_FILES for call in calls)
    assert all(call["local_dir_use_symlinks"] is False for call in calls)
    assert all(call["resume_download"] is True for call in calls)
    assert (tmp_path / "model" / "alpha").is_dir()
    assert (tmp_path / "model" / "beta").is_dir()


def test_main_returns_zero(monkeypatch) -> None:
    monkeypatch.setattr(download_hf_model, "download_all_models", lambda: None)

    assert download_hf_model.main() == 0
