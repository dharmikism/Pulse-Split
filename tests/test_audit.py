import csv
import json

import numpy as np
import pytest
import soundfile as sf

from pulsesplit.audit import audit_dataset


def _triplet(root, key="0001"):
    root.mkdir(parents=True, exist_ok=True)
    signal = 0.2 * np.sin(np.arange(800) * 0.1)
    for role, scale in (("M", 1), ("H", 0.6), ("L", 0.4)):
        sf.write(root / f"{role}{key}.wav", signal * scale, 8000, subtype="FLOAT")


def test_audit_exports_original_audio_statistics(tmp_path):
    root = tmp_path / "audio"
    _triplet(root)
    _triplet(root, "0002")
    output = tmp_path / "audit"
    summary = audit_dataset(root, output)
    assert summary["status"] == "passed"
    assert summary["recording_count"] == 2
    assert summary["file_count"] == 6
    assert summary["mixed_duration_seconds"] == pytest.approx(0.2)
    assert summary["sample_rates"] == [8000]
    assert summary["channel_counts"] == [1]
    assert summary["issue_counts"] == {}
    assert json.loads((output / "dataset_audit.json").read_text()) == summary
    with (output / "dataset_audit.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6
    assert rows[0]["id"] == "0001"
    assert int(rows[0]["frames"]) == 800
    assert float(rows[0]["peak"]) == pytest.approx(0.2, abs=1e-5)


def test_audit_flags_nonfinite_silent_stereo_and_mismatched_audio(tmp_path):
    _triplet(tmp_path)
    sf.write(tmp_path / "M0001.wav", np.column_stack((np.full(900, 0.2), np.zeros(900))),
             16000, subtype="FLOAT")
    sf.write(tmp_path / "H0001.wav", np.zeros(800), 8000, subtype="FLOAT")
    lung = np.full(800, 0.1)
    lung[:3] = [np.nan, np.inf, -1.0]
    sf.write(tmp_path / "L0001.wav", lung, 8000, subtype="FLOAT")
    summary = audit_dataset(tmp_path, output=tmp_path / "audit")
    assert summary["status"] == "failed"
    assert summary["issue_counts"] == {
        "duration_mismatch": 1, "frame_count_mismatch": 1,
        "multichannel_audio": 1, "nonfinite_audio": 1,
        "possible_clipping": 1, "sample_rate_mismatch": 1, "silent_channel": 2,
    }
    mixed, heart, lung = summary["files"]
    assert mixed["silent_channels"] == [2]
    assert heart["rms"] == 0
    assert lung["nonfinite_samples"] == 2
    assert lung["clipped_samples"] == 1
    assert lung["clipped_fraction"] == pytest.approx(1 / 800)
    assert np.isfinite(lung["rms"])
    # JSON is standards compliant even when the source audio contains NaN/Inf.
    text = (tmp_path / "audit" / "dataset_audit.json").read_text()
    assert "NaN" not in text and "Infinity" not in text


def test_audit_distinguishes_quality_warnings_and_corrupt_files(tmp_path):
    _triplet(tmp_path)
    sf.write(tmp_path / "H0001.wav", np.zeros(800), 8000)
    assert audit_dataset(tmp_path, output=None)["status"] == "warnings"
    (tmp_path / "L0001.wav").write_bytes(b"invalid wav")
    summary = audit_dataset(tmp_path, output=None)
    assert summary["status"] == "failed"
    assert summary["issue_counts"]["unreadable_audio"] == 1
    assert summary["files"][2]["error"]


def test_audit_handles_empty_and_entirely_nonfinite_wavs(tmp_path):
    _triplet(tmp_path)
    sf.write(tmp_path / "H0001.wav", np.array([], dtype=np.float32), 8000)
    sf.write(tmp_path / "L0001.wav", np.full(800, np.nan), 8000, subtype="FLOAT")
    summary = audit_dataset(tmp_path, output=tmp_path / "audit")
    assert summary["status"] == "failed"
    assert summary["issue_counts"]["empty_audio"] == 1
    assert summary["issue_counts"]["nonfinite_audio"] == 1
    assert summary["files"][2]["rms"] is None
    assert summary["files"][2]["peak"] is None


def test_audit_split_manifest_detects_leakage_unknown_and_missing_ids(tmp_path):
    for key in ("0001", "0002", "0003"):
        _triplet(tmp_path, key)
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({"train": ["0001", "0001"], "val": ["0001"],
                                    "test": ["0002", "9999"], "seed": 42}))
    summary = audit_dataset(tmp_path, output=None, splits=manifest)
    assert summary["status"] == "failed"
    result = summary["splits"]
    assert result["valid"] is False
    assert result["overlaps"]["train_val"] == ["0001"]
    assert result["duplicate_ids"]["train"] == ["0001"]
    assert result["unknown_ids"] == ["9999"]
    assert result["unassigned_ids"] == ["0003"]
    valid = {"train": ["0001"], "val": ["0002"], "test": ["0003"]}
    assert audit_dataset(tmp_path, output=None, splits=valid)["splits"]["valid"] is True


def test_audit_rejects_incomplete_triplets_and_invalid_split_formats(tmp_path):
    _triplet(tmp_path)
    with pytest.raises(ValueError, match="list of string recording IDs"):
        audit_dataset(tmp_path, output=None, splits={"train": [1], "val": [], "test": []})
    (tmp_path / "H0001.wav").unlink()
    with pytest.raises(ValueError, match="missing matched audio"):
        audit_dataset(tmp_path, output=None)


@pytest.mark.parametrize("kwargs", [{"silence_threshold": -1}, {"silence_threshold": float("nan")},
                                   {"clipping_threshold": 0}, {"clipping_threshold": 1.1}])
def test_audit_validates_thresholds(tmp_path, kwargs):
    with pytest.raises(ValueError, match="threshold"):
        audit_dataset(tmp_path, output=None, **kwargs)
