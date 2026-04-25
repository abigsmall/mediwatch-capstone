from pathlib import Path
from types import SimpleNamespace


def test_load_model_uses_packaged_model_when_manifest_has_host_path(tmp_path, monkeypatch):
    from mediwatch import artifacts

    packaged_model = tmp_path / "models" / "model.joblib"
    packaged_model.parent.mkdir()
    packaged_model.write_text("placeholder", encoding="utf-8")
    loaded_paths = []

    monkeypatch.setattr(artifacts, "settings", SimpleNamespace(models_dir=packaged_model.parent))
    monkeypatch.setattr(
        artifacts,
        "read_manifest",
        lambda: {"local_model_path": f"/host/path/{packaged_model.name}", "s3_model_uri": ""},
    )
    monkeypatch.setattr(
        artifacts.joblib,
        "load",
        lambda path: loaded_paths.append(Path(path)) or "loaded-model",
    )

    assert artifacts.load_model() == "loaded-model"
    assert loaded_paths == [packaged_model]
