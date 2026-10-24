"""The MLflow store stays under the project root, whatever the working directory is."""

import mlflow

from geneva_permits.tracking import set_experiment


def test_store_and_artifacts_land_under_the_root(tmp_path, monkeypatch):
    root, cwd = tmp_path / "mlruns", tmp_path / "elsewhere"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    set_experiment("probe", root=root)
    set_experiment("probe", root=root)  # reselecting an existing experiment
    with mlflow.start_run():
        mlflow.log_metric("macro_f1", 0.5)
        mlflow.log_text("x", "note.txt")

    assert (root / "mlflow.db").is_file()
    assert any((root / "probe").rglob("note.txt"))
    assert list(cwd.iterdir()) == []
