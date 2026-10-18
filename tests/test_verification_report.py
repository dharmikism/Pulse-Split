"""Checks that the report follows evidence instead of hardcoded earlier results."""
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "verification_report", Path(__file__).resolve().parents[1] / "docs" / "build_verification_report.py"
)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


@pytest.fixture
def proof(tmp_path):
    directory = tmp_path / "new-run" / "proof"
    directory.mkdir(parents=True)
    artifacts = {
        "verification.json": {
            "data": "Synthetic sine-wave fixtures",
            "created_utc": "20261018T120000Z",
            "inference_shape": [2, 8000],
            "checks": {"pytest": "passed", "checkpoint_reload_and_inference": "passed"},
            "packages": {"torch": "test-version"},
        },
        "run_metadata.json": {
            "recordings": 3, "split_sizes": {"train": 1, "val": 1, "test": 1},
            "seed": 17, "device": "cpu", "best_val_logmag_mae": .125,
        },
        "split.json": {"train": ["01"], "val": ["02"], "test": ["03"], "seed": 17},
        "history.json": [
            {"epoch": 1, "train_logmag_mae": .3, "val_logmag_mae": .25},
            {"epoch": 2, "train_logmag_mae": .2, "val_logmag_mae": .125},
        ],
    }
    for name, content in artifacts.items():
        (directory / name).write_text(json.dumps(content), encoding="utf-8")
    (directory / "tests.xml").write_text(
        '<testsuites><testsuite tests="2" errors="0" failures="0" skipped="1">'
        '<testcase name="passed"/><testcase name="unavailable"><skipped/></testcase>'
        '</testsuite></testsuites>', encoding="utf-8",
    )
    return directory


def test_report_reads_counts_losses_and_hashes_from_this_run(proof):
    evidence = report.collect_evidence(proof)
    assert evidence["tests"] == {"total": 2, "passed": 1, "failed": 0, "errors": 0, "skipped": 1}
    paths = report.build_reports(proof, include_docx=False)
    text = paths[0].read_text(encoding="utf-8")
    assert "1 passed / 2 collected" in text
    assert "| Synthetic recording triplets | 3 |" in text
    assert "| 2 | 0.2 | 0.125 |" in text
    assert "not UCI dataset performance" in text
    assert all(len(digest) == 64 and digest in text for digest in evidence["sources"].values())


def test_report_rejects_disagreement_between_junit_and_pass_claim(proof):
    (proof / "tests.xml").write_text(
        '<testsuites><testsuite errors="0"><testcase name="broken"><failure/>'
        '</testcase></testsuite></testsuites>', encoding="utf-8",
    )
    with pytest.raises(ValueError, match="JUnit records failures"):
        report.collect_evidence(proof)


def test_report_rejects_recording_leakage(proof):
    split = json.loads((proof / "split.json").read_text())
    split["test"] = split["train"]
    (proof / "split.json").write_text(json.dumps(split), encoding="utf-8")
    with pytest.raises(ValueError, match="overlapping"):
        report.collect_evidence(proof)


def test_report_rejects_nonfinite_training_evidence(proof):
    history = json.loads((proof / "history.json").read_text())
    history[0]["train_logmag_mae"] = float("nan")
    (proof / "history.json").write_text(json.dumps(history), encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        report.collect_evidence(proof)


def test_checkpoint_loss_follows_minimum_improvement_policy(proof):
    history = json.loads((proof / "history.json").read_text())
    history[0]["val_logmag_mae"] = .125005
    (proof / "history.json").write_text(json.dumps(history), encoding="utf-8")
    metadata = json.loads((proof / "run_metadata.json").read_text())
    metadata["best_val_logmag_mae"] = .125005
    (proof / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    assert report.collect_evidence(proof)["metadata"]["best_val_logmag_mae"] == .125005
    metadata["best_val_logmag_mae"] = .125
    (proof / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="selection policy"):
        report.collect_evidence(proof)


def test_editable_word_report_matches_evidence(proof):
    docx = pytest.importorskip("docx")
    paths = report.build_reports(proof)
    document = docx.Document(paths[1])
    table_text = "\n".join(cell.text for table in document.tables for row in table.rows for cell in row.cells)
    paragraph_text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    assert "1 passed / 2 collected" in table_text
    assert "0.125" in table_text
    assert "Synthetic audio software verification only" in paragraph_text
    assert "new-run" in paragraph_text
