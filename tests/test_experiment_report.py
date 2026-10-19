"""Experiment reports must follow actual saved metrics and immutable run evidence."""
import importlib.util
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from pulsesplit.reporting import file_sha256, write_evaluation_report


spec = importlib.util.spec_from_file_location(
    "experiment_report", Path(__file__).resolve().parents[1] / "docs" / "build_experiment_report.py"
)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


@pytest.fixture
def experiment(tmp_path):
    artifacts, results = tmp_path / "artifacts", tmp_path / "results"
    artifacts.mkdir()
    (results / "audit").mkdir(parents=True)
    (artifacts / "best.pt").write_bytes(b"immutable test checkpoint fixture, not a trained model")
    split = {"train": ["0001"], "val": ["0002"], "test": ["0031", "0032"], "seed": 42}
    metadata = {
        "recordings": 4, "split_sizes": {"train": 1, "val": 1, "test": 2},
        "epochs_completed": 2, "best_val_logmag_mae": .1, "seed": 42,
        "dataset_label": "UCI-format report unit-test fixture", "batch_size": 4,
        "device": "cpu", "segment_seconds": 2, "base_channels": 4,
    }
    history = [
        {"epoch": 1, "train_logmag_mae": .3, "val_logmag_mae": .2},
        {"epoch": 2, "train_logmag_mae": .15, "val_logmag_mae": .1},
    ]
    for name, value in (("split.json", split), ("history.json", history), ("run_metadata.json", metadata)):
        (artifacts / name).write_text(json.dumps(value), encoding="utf-8")
    frame = pd.DataFrame({
        "id": ["0031", "0032"], "heart_si_sdr": [-7., -9.], "lung_si_sdr": [-18., -9.],
        "heart_baseline_si_sdr": [-10., -8.], "lung_baseline_si_sdr": [-15., -10.],
        "heart_improvement": [3., -1.], "lung_improvement": [-3., 1.],
    })
    frame.to_csv(results / "test_per_recording.csv", index=False)
    summary = {
        "num_test_recordings": 2, "dataset_label": "UCI-format report unit-test fixture",
        "checkpoint_sha256": file_sha256(artifacts / "best.pt"),
        "split_sha256": file_sha256(artifacts / "split.json"), "disclaimer": "Unit-test fixture only",
    }
    write_evaluation_report(frame, summary, results)
    ids = ["0001", "0002", "0031", "0032"]
    audit = {
        "recording_count": 4, "file_count": 12, "status": "warnings",
        "sample_rates": [4000], "channel_counts": [1], "mixed_duration_seconds": 60,
        "recordings": [{"id": key} for key in ids], "issue_counts": {"possible_clipping": 1},
        "files": [{"id": key, "role": role, "sample_rate": 4000, "channels": 1, "duration_seconds": 15}
                  for key in ids for role in ("mixed", "heart", "lung")],
    }
    (results / "audit/dataset_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    return artifacts, results


def test_report_preserves_signed_results_ids_and_native_properties(experiment, tmp_path):
    artifacts, results = experiment
    paths = report.build_experiment_report(artifacts, results, tmp_path / "report", include_docx=False)
    text = paths[0].read_text(encoding="utf-8")
    assert paths[0].name == "Actual_Experiment_Report.md"
    assert "| Heart: gain over mixture | 2 | 0 | 1.000 |" in text
    assert "| Lung: gain over mixture | 2 | 0 | -1.000 |" in text
    assert "`0031`, `0032`" in text
    assert "| Native sample rate(s), Hz | 4,000 |" in text
    assert "22,050 Hz stated" in text
    assert "does not restore frequencies" in text
    assert "not a patient cohort" in text
    assert "not percentage accuracy" in text
    assert file_sha256(artifacts / "best.pt") in text


def test_report_rejects_changed_checkpoint(experiment):
    artifacts, results = experiment
    (artifacts / "best.pt").write_bytes(b"checkpoint replaced after evaluation")
    with pytest.raises(ValueError, match="Checkpoint SHA-256"):
        report.collect_experiment(artifacts, results)


def test_report_rejects_wrong_heldout_ids(experiment):
    artifacts, results = experiment
    path = results / "test_per_recording.csv"
    path.write_text(path.read_text().replace("0031", "0001"), encoding="utf-8")
    with pytest.raises(ValueError, match="held-out test split"):
        report.collect_experiment(artifacts, results)


def test_report_rejects_stale_aggregate_statistics(experiment):
    artifacts, results = experiment
    path = results / "summary.json"
    summary = json.loads(path.read_text())
    summary["statistics"]["heart_improvement"]["mean"] = 99
    path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="heart_improvement/mean"):
        report.collect_experiment(artifacts, results)


def test_report_rejects_wrong_per_recording_gain(experiment):
    artifacts, results = experiment
    path = results / "test_per_recording.csv"
    frame = pd.read_csv(path, dtype={"id": str})
    frame.loc[0, "heart_improvement"] = 30
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Incorrect heart improvement"):
        report.collect_experiment(artifacts, results)


def test_report_rejects_catalogue_rate_substituted_for_measured_rate(experiment):
    artifacts, results = experiment
    path = results / "audit/dataset_audit.json"
    audit = json.loads(path.read_text())
    audit["sample_rates"] = [22050]
    path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="sample-rate summary"):
        report.collect_experiment(artifacts, results)


def _add_source_group_metadata(artifacts, results):
    path = artifacts / "split.json"
    split = json.loads(path.read_text())
    split.update({
        "strategy": "exact-source-connected-components",
        "groups": {"0001": ["0001"], "0002": ["0002"], "0031": ["0031", "0032"]},
        "group_assignments": {"0001": "train", "0002": "val", "0031": "test"},
        "group_sizes": {"0001": 1, "0002": 1, "0031": 2},
        "actual_sizes": {"train": 1, "val": 1, "test": 2},
        "file_hashes": {key: {role: hashlib.sha256(f"{key}-{role}".encode()).hexdigest()
                              for role in ("mixed", "heart", "lung")}
                        for key in ("0001", "0002", "0031", "0032")},
    })
    split["file_hashes"]["0032"]["heart"] = split["file_hashes"]["0031"]["heart"]
    path.write_text(json.dumps(split), encoding="utf-8")
    summary_path = results / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["split_sha256"] = file_sha256(path)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return split


def test_report_describes_exact_source_components_without_claiming_pattern_holdout(experiment):
    artifacts, results = experiment
    _add_source_group_metadata(artifacts, results)
    evidence = report.collect_experiment(artifacts, results)
    text = report.render_experiment_markdown(evidence)
    assert "| Exact-source connected components | 3 |" in text
    assert "| Train / validation / test component counts | 1 / 1 / 1 |" in text
    assert "| Distinct byte hashes (mixture / heart / lung) | 4 / 3 / 4 |" in text
    assert "not a holdout of all simulator sound types" in text
    assert "descriptive per-recording bootstrap" in text


@pytest.mark.parametrize("change", ["assignment", "cross_partition_hash"])
def test_report_rejects_inconsistent_source_group_claims(experiment, change):
    artifacts, results = experiment
    split = _add_source_group_metadata(artifacts, results)
    if change == "assignment":
        split["group_assignments"]["0031"] = "val"
    else:
        split["file_hashes"]["0001"]["lung"] = split["file_hashes"]["0031"]["heart"]
    (artifacts / "split.json").write_text(json.dumps(split), encoding="utf-8")
    summary_path = results / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["split_sha256"] = file_sha256(artifacts / "split.json")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="component spans partitions|source hash spans"):
        report.collect_experiment(artifacts, results)


def test_experiment_word_appendix_is_editable_and_has_measured_metrics(experiment, tmp_path):
    docx = pytest.importorskip("docx")
    paths = report.build_experiment_report(*experiment, tmp_path / "report")
    document = docx.Document(paths[1])
    table_text = "\n".join(cell.text for table in document.tables for row in table.rows for cell in row.cells)
    paragraph_text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    assert "Heart: gain over mixture" in table_text
    assert "-1.000" in table_text
    assert "4,000" in table_text
    assert "clinical validation" in paragraph_text
