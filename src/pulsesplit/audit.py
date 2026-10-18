"""Inspect original WAV quality and recording-level split integrity before training.

The audit measures dataset properties, not separation accuracy. Signals are read at
their native sample rate and channel count so preprocessing cannot hide problems.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .data import find_triplets


def _rms(values):
    peak = float(np.max(np.abs(values)))
    # Scaling keeps even valid DOUBLE WAV amplitudes from overflowing a square.
    return peak * float(np.sqrt(np.mean((values / peak) ** 2))) if peak else 0.0


def _audit_file(path, recording_id, role, silence_threshold, clipping_threshold):
    row = {
        "id": recording_id,
        "role": role,
        "path": str(path),
        "sample_rate": None,
        "channels": None,
        "frames": None,
        "duration_seconds": None,
        "peak": None,
        "rms": None,
        "nonfinite_samples": 0,
        "clipped_samples": 0,
        "clipped_fraction": 0.0,
        "silent_channels": [],
        "issues": [],
    }
    try:
        samples, sr = sf.read(str(path), dtype="float64", always_2d=True)
    except (OSError, RuntimeError) as exc:
        row["issues"].append("unreadable_audio")
        row["error"] = str(exc)
        return row

    row.update(sample_rate=int(sr), channels=int(samples.shape[1]),
               frames=int(samples.shape[0]), duration_seconds=len(samples) / sr)
    if not samples.size:
        row["issues"].append("empty_audio")
        return row

    finite = np.isfinite(samples)
    row["nonfinite_samples"] = int((~finite).sum())
    if row["nonfinite_samples"]:
        row["issues"].append("nonfinite_audio")
    values = samples[finite]
    if values.size:
        row["peak"] = float(np.max(np.abs(values)))
        row["rms"] = _rms(values)
        row["clipped_samples"] = int((np.abs(values) >= clipping_threshold).sum())
        row["clipped_fraction"] = row["clipped_samples"] / samples.size
        if row["clipped_samples"]:
            row["issues"].append("possible_clipping")
    for channel in range(samples.shape[1]):
        values = samples[:, channel]
        if np.isfinite(values).all():
            rms = _rms(values)
            if rms <= silence_threshold:
                row["silent_channels"].append(channel + 1)
    if row["silent_channels"]:
        row["issues"].append("silent_channel")
    if samples.shape[1] != 1:
        row["issues"].append("multichannel_audio")
    return row


def _audit_splits(splits, recording_ids):
    if isinstance(splits, (str, Path)):
        splits = json.loads(Path(splits).read_text(encoding="utf-8"))
    if not isinstance(splits, Mapping):
        raise ValueError("Splits must be a mapping or a path to a split JSON file")
    groups = {}
    for name in ("train", "val", "test"):
        ids = splits.get(name)
        if not isinstance(ids, list) or not all(isinstance(key, str) for key in ids):
            raise ValueError(f"Split {name!r} must contain a list of string recording IDs")
        groups[name] = ids
    overlaps = {
        f"{left}_{right}": sorted(set(groups[left]) & set(groups[right]))
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    assigned = set().union(*(set(ids) for ids in groups.values()))
    duplicates = {name: sorted(key for key, count in Counter(ids).items() if count > 1)
                  for name, ids in groups.items()}
    unknown = sorted(assigned - recording_ids)
    unassigned = sorted(recording_ids - assigned)
    empty = [name for name, ids in groups.items() if not ids]
    return {
        "valid": not (any(overlaps.values()) or any(duplicates.values()) or unknown or unassigned or empty),
        "sizes": {name: len(ids) for name, ids in groups.items()},
        "overlaps": overlaps,
        "duplicate_ids": duplicates,
        "unknown_ids": unknown,
        "unassigned_ids": unassigned,
        "empty_splits": empty,
    }


def audit_dataset(root, output="results/audit", splits=None, *,
                  silence_threshold=1e-6, clipping_threshold=0.999):
    """Return and optionally save original-audio statistics for matched triplets.

    ``splits`` accepts the training ``split.json`` path or the equivalent mapping.
    IDs must be strings such as ``"0001"``. Empty, overlapping, duplicated,
    unknown, or unassigned splits fail the integrity check. Pass ``output=None``
    to inspect without creating files. Near-full-scale samples are flagged as
    *possible* clipping; silence is measured per channel using RMS amplitude.

    Missing or ambiguous H####/L####/M#### pairs raise the same discovery errors
    as training. Corrupt WAVs are included in the report as unreadable audio.
    """
    if not np.isfinite(silence_threshold) or silence_threshold < 0:
        raise ValueError("Silence threshold must be a finite nonnegative amplitude")
    if not np.isfinite(clipping_threshold) or not 0 < clipping_threshold <= 1:
        raise ValueError("Clipping threshold must be in (0, 1]")

    triplets = find_triplets(Path(root))
    files = []
    recordings = []
    for triplet in triplets:
        rows = [_audit_file(getattr(triplet, role), triplet.key, role,
                            silence_threshold, clipping_threshold)
                for role in ("mixed", "heart", "lung")]
        files.extend(rows)
        issues = []
        readable = [row for row in rows if row["sample_rate"] is not None]
        if len(readable) == 3:
            if len({row["sample_rate"] for row in readable}) > 1:
                issues.append("sample_rate_mismatch")
            if len({row["frames"] for row in readable}) > 1:
                issues.append("frame_count_mismatch")
            durations = [row["duration_seconds"] for row in readable]
            # A tolerance of one sample at the lowest rate handles rounding.
            if max(durations) - min(durations) > 1 / min(row["sample_rate"] for row in readable):
                issues.append("duration_mismatch")
        recordings.append({"id": triplet.key, "issues": issues})

    counts = Counter(issue for row in files + recordings for issue in row["issues"])
    split_audit = _audit_splits(splits, {item.key for item in triplets}) if splits is not None else None
    serious = any(counts[key] for key in ("unreadable_audio", "empty_audio", "nonfinite_audio"))
    if split_audit is not None and not split_audit["valid"]:
        serious = True
    summary = {
        "schema_version": 1,
        "root": str(Path(root)),
        "status": "failed" if serious else ("warnings" if counts else "passed"),
        "recording_count": len(triplets),
        "file_count": len(files),
        "mixed_duration_seconds": sum(row["duration_seconds"] or 0 for row in files if row["role"] == "mixed"),
        "sample_rates": sorted({row["sample_rate"] for row in files if row["sample_rate"] is not None}),
        "channel_counts": sorted({row["channels"] for row in files if row["channels"] is not None}),
        "issue_counts": dict(sorted(counts.items())),
        "thresholds": {"silence_rms": float(silence_threshold), "possible_clipping_peak": float(clipping_threshold)},
        "splits": split_audit,
        "recordings": recordings,
        "files": files,
        "note": "Dataset integrity measurements only; these are not model accuracy or clinical performance results.",
    }
    if output is not None:
        destination = Path(output)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "dataset_audit.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        fields = [key for key in files[0] if key != "error"] + ["error"]
        with (destination / "dataset_audit.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in files:
                writer.writerow({**row, "issues": ";".join(row["issues"]),
                                 "silent_channels": ";".join(map(str, row["silent_channels"]))})
    return summary
