"""Giao dịch đánh giá trên fixture tổng hợp tmp_path, không mở dữ liệu thật."""

import json
from types import SimpleNamespace

import pytest

import src.evaluate as evaluation
import src.pipeline as pipeline_module
import src.visualization as visualization
from src.pipeline import sha256_file


@pytest.fixture
def synthetic_run(tmp_path, monkeypatch):
    """Model/data giả lập điều phối; metric thật có đáp án tay hai dự đoán đúng."""
    run_id = "synthetic-integrity"
    calls = {"predict": 0, "prepare": [], "render": 0, "load": 0}

    def put(relative, text):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    csv_path = put("synthetic.csv", "synthetic fixture,not real water data\n")
    data_path = put("data/manifest.json", json.dumps({"seed": 137, "test_fraction": 0.2, "n_folds": 5}))
    model_path = put(f"artifacts/{run_id}/pipeline.json", "synthetic model stub\n")
    cv = put(f"results/metrics/{run_id}/cv_results.json", '{"synthetic": true}\n')
    eda = put(f"results/figures/{run_id}/eda.png", "synthetic EDA bytes\n")
    files = {path.relative_to(tmp_path).as_posix(): sha256_file(path) for path in (model_path, cv, eda)}
    manifest = {"run_id": run_id, "stage": "trained_not_tested", "files": files,
                "model_path": model_path.relative_to(tmp_path).as_posix(),
                "model_sha256": sha256_file(model_path), "csv_sha256": sha256_file(csv_path),
                "data_manifest_sha256": sha256_file(data_path), "baseline_prior_1": 0.5,
                "environment": {"dirty": True}}
    run_path = put(f"results/runs/{run_id}.json", json.dumps(manifest))

    def predict(rows):
        calls["predict"] += 1
        assert rows == [[None] * 9, [2.0] * 9]
        return [[0.8, 0.2], [0.1, 0.9]]

    def load(*args, **kwargs):
        calls["load"] += 1
        return SimpleNamespace(run_id=run_id, predict_proba=predict, metadata={"source_verified": False})

    def prepare(path, root, *, seed):
        calls["prepare"].append(seed)
        return {"manifest_path": data_path, "test": [0, 1],
                "clean": {"X": [[None] * 9, [2.0] * 9], "y": [0, 1], "row_ids": [4, 9]}}

    def render(metrics, directory):
        calls["render"] += 1
        assert metrics["confusion_matrix"] == [[1, 0], [0, 1]]
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "confusion_matrix.png").write_bytes(b"synthetic rendered figure")
        (directory / "evaluation_manifest.json").write_text('{"synthetic": true}', encoding="utf-8")

    monkeypatch.setattr(pipeline_module, "load_pipeline", load)
    monkeypatch.setattr(evaluation, "prepare_data", prepare)
    monkeypatch.setattr(visualization, "create_evaluation_figures", render)
    return SimpleNamespace(root=tmp_path, run_id=run_id, csv=csv_path, data=data_path,
                           model=model_path, cv=cv, eda=eda, run=run_path, original=manifest,
                           calls=calls, render=render)


def run(fixture):
    """Chỉ gọi evaluate với các đường dẫn synthetic đã khóa trong tmp_path."""
    return evaluation.evaluate_run(fixture.run_id, fixture.csv, fixture.root, verification_confirmed=True)


def test_verified_seed_and_preserved_old_hashes(synthetic_run):
    fixture = synthetic_run
    stale = fixture.eda.parent / "untracked-old.png"
    stale.write_bytes(b"must not sign unrelated files")
    result = run(fixture)
    assert result["model"]["accuracy"] == 1.0
    assert result["baseline"]["roc_auc"] == 0.5
    assert fixture.calls["prepare"] == [137]
    assert fixture.calls["predict"] == 1
    manifest = json.loads(fixture.run.read_text(encoding="utf-8"))
    assert manifest["stage"] == "evaluated"
    for relative, checksum in fixture.original["files"].items():
        assert manifest["files"][relative] == checksum
    assert stale.relative_to(fixture.root).as_posix() not in manifest["files"]
    for relative, checksum in manifest["files"].items():
        assert sha256_file(fixture.root / relative) == checksum
    predictions = fixture.root / f"results/metrics/{fixture.run_id}/test_predictions.csv"
    assert "4,0,0,0.8,0.2" in predictions.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="đã đánh giá"):
        run(fixture)
    assert fixture.calls["predict"] == 1


@pytest.mark.parametrize("attribute", ["cv", "eda", "model", "data", "csv"])
def test_corruption_rejected_before_prediction(synthetic_run, attribute):
    fixture = synthetic_run
    path = getattr(fixture, attribute)
    path.write_text("corrupted", encoding="utf-8")
    with pytest.raises(ValueError, match="Checksum|checksum|khớp"):
        run(fixture)
    assert fixture.calls["predict"] == 0
    assert fixture.calls["prepare"] == []
    assert json.loads(fixture.run.read_text(encoding="utf-8")) == fixture.original


@pytest.mark.parametrize("relative", ["../escape.txt", "/absolute.txt", "results/../../escape.txt"])
def test_manifest_path_escape_rejected(synthetic_run, relative):
    fixture = synthetic_run
    manifest = dict(fixture.original)
    manifest["files"] = {relative: "0" * 64}
    fixture.run.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="root"):
        run(fixture)
    assert fixture.calls["predict"] == 0
    assert fixture.calls["load"] == 0


def test_render_failure_retries_checkpoint_without_test_access(synthetic_run, monkeypatch):
    fixture = synthetic_run

    def fail_render(metrics, directory):
        checkpoint = fixture.run.with_name(f"{fixture.run_id}.evaluation_checkpoint.json")
        stored = json.loads(checkpoint.read_text(encoding="utf-8"))
        assert stored["model"]["accuracy"] == 1.0
        assert "test_predictions.csv" in stored["outputs"]
        directory.mkdir(parents=True)
        (directory / "partial.png").write_bytes(b"partial")
        raise OSError("synthetic render failure")

    monkeypatch.setattr(visualization, "create_evaluation_figures", fail_render)
    with pytest.raises(OSError, match="render failure"):
        run(fixture)
    assert not (fixture.cv.parent / "test_metrics.json").exists()
    assert not (fixture.eda.parent / "partial.png").exists()
    assert json.loads(fixture.run.read_text(encoding="utf-8"))["stage"] == "evaluation_checkpointed"
    monkeypatch.setattr(visualization, "create_evaluation_figures", fixture.render)
    result = run(fixture)
    assert result["model"]["accuracy"] == 1.0
    assert fixture.calls["predict"] == fixture.calls["load"] == 1
    assert fixture.calls["prepare"] == [137]


def test_inference_failure_blocks_automatic_retry(synthetic_run, monkeypatch):
    fixture = synthetic_run

    def fail_predict(rows):
        fixture.calls["predict"] += 1
        assert json.loads(fixture.run.read_text(encoding="utf-8"))["stage"] == "evaluation_started"
        raise OSError("synthetic inference failure")

    monkeypatch.setattr(pipeline_module, "load_pipeline", lambda *args, **kwargs:
                        SimpleNamespace(run_id=fixture.run_id, predict_proba=fail_predict))
    with pytest.raises(OSError, match="inference failure"):
        run(fixture)
    with pytest.raises(ValueError, match="interrupted.*checkpoint"):
        run(fixture)
    assert fixture.calls["predict"] == 1


@pytest.mark.parametrize("missing", [False, True])
def test_invalid_checkpoint_does_not_reinfer(synthetic_run, monkeypatch, missing):
    fixture = synthetic_run
    monkeypatch.setattr(visualization, "create_evaluation_figures",
                        lambda *args: (_ for _ in ()).throw(OSError("stop render")))
    with pytest.raises(OSError, match="stop render"):
        run(fixture)
    checkpoint = fixture.run.with_name(f"{fixture.run_id}.evaluation_checkpoint.json")
    if missing:
        checkpoint.unlink()
    else:
        checkpoint.write_text("corrupted checkpoint", encoding="utf-8")
    with pytest.raises(ValueError, match="interrupted"):
        run(fixture)
    assert fixture.calls["predict"] == fixture.calls["load"] == 1


def test_exclusive_lock_rejects_second_evaluator(synthetic_run):
    fixture = synthetic_run
    lock_path = fixture.run.with_suffix(".evaluate.lock")
    with evaluation._evaluation_lock(lock_path):
        with pytest.raises(ValueError, match="exclusive lock"):
            run(fixture)
    assert fixture.calls["predict"] == 0
    assert run(fixture)["model"]["accuracy"] == 1.0


def test_interrupted_publish_retries_without_inference(synthetic_run, monkeypatch):
    fixture = synthetic_run
    replace = evaluation.os.replace
    publications = []

    def fail_publish(source, destination):
        if str(destination).endswith("test_metrics.json"):
            publications.append(str(destination))
            raise OSError("synthetic publish failure")
        return replace(source, destination)

    monkeypatch.setattr(evaluation.os, "replace", fail_publish)
    with pytest.raises(OSError, match="publish failure"):
        run(fixture)
    assert publications
    partial = json.loads(fixture.run.read_text(encoding="utf-8"))
    assert partial["stage"] == "evaluation_checkpointed"
    assert partial["files"] == fixture.original["files"]
    monkeypatch.setattr(evaluation.os, "replace", replace)
    assert run(fixture)["model"]["accuracy"] == 1.0
    assert fixture.calls["predict"] == 1


def test_final_manifest_failure_is_recoverable(synthetic_run, monkeypatch):
    fixture = synthetic_run
    durable_json = evaluation._durable_json

    def fail_final(path, value):
        if path == fixture.run and value.get("stage") == "evaluated":
            raise OSError("synthetic final manifest failure")
        return durable_json(path, value)

    monkeypatch.setattr(evaluation, "_durable_json", fail_final)
    with pytest.raises(OSError, match="final manifest failure"):
        run(fixture)
    manifest = json.loads(fixture.run.read_text(encoding="utf-8"))
    assert manifest["stage"] == "evaluation_checkpointed"
    assert manifest["files"] == fixture.original["files"]
    assert (fixture.cv.parent / "test_metrics.json").is_file()
    monkeypatch.setattr(evaluation, "_durable_json", durable_json)
    assert run(fixture)["model"]["accuracy"] == 1.0
    assert fixture.calls["predict"] == fixture.calls["load"] == 1


def test_corrupted_partial_publication_is_not_resigned(synthetic_run, monkeypatch):
    fixture = synthetic_run
    durable_json = evaluation._durable_json

    def fail_final(path, value):
        if path == fixture.run and value.get("stage") == "evaluated":
            raise OSError("stop final manifest")
        return durable_json(path, value)

    monkeypatch.setattr(evaluation, "_durable_json", fail_final)
    with pytest.raises(OSError, match="stop final"):
        run(fixture)
    metrics_path = fixture.cv.parent / "test_metrics.json"
    metrics_path.write_text("corrupted partial publication", encoding="utf-8")
    monkeypatch.setattr(evaluation, "_durable_json", durable_json)
    with pytest.raises(ValueError, match="bị thay đổi"):
        run(fixture)
    assert metrics_path.read_text(encoding="utf-8") == "corrupted partial publication"
    assert fixture.calls["predict"] == 1


def test_second_evaluator_is_blocked_during_inference(synthetic_run, monkeypatch):
    fixture = synthetic_run

    def predict(rows):
        fixture.calls["predict"] += 1
        with pytest.raises(ValueError, match="exclusive lock"):
            run(fixture)
        return [[0.8, 0.2], [0.1, 0.9]]

    monkeypatch.setattr(pipeline_module, "load_pipeline", lambda *args, **kwargs:
                        SimpleNamespace(run_id=fixture.run_id, predict_proba=predict, metadata={}))
    assert run(fixture)["model"]["accuracy"] == 1.0
    assert fixture.calls["predict"] == 1


@pytest.mark.parametrize("field,value", [("test_fraction", 0.3), ("n_folds", 3)])
def test_split_configuration_drift_stops_before_predict(synthetic_run, field, value):
    fixture = synthetic_run
    data_manifest = json.loads(fixture.data.read_text(encoding="utf-8"))
    data_manifest[field] = value
    fixture.data.write_text(json.dumps(data_manifest), encoding="utf-8")
    manifest = dict(fixture.original)
    manifest["data_manifest_sha256"] = sha256_file(fixture.data)
    fixture.run.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="test_fraction/n_folds"):
        run(fixture)
    assert fixture.calls["predict"] == fixture.calls["load"] == 0


def test_unverified_source_never_becomes_candidate(synthetic_run):
    fixture = synthetic_run
    manifest = dict(fixture.original)
    manifest["environment"] = {"dirty": False}
    fixture.run.write_text(json.dumps(manifest), encoding="utf-8")
    run(fixture)
    manifest = json.loads(fixture.run.read_text(encoding="utf-8"))
    assert manifest["acceptance_status"] == "unverified_source_evaluation"
