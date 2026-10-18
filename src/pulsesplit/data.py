"""Discovers paired HLS-CMDS recordings and forms record-level train/val/test splits."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import re
import zipfile
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np
from .audio import load_audio, align_by_length, shared_scale

@dataclass(frozen=True)
class Triplet:
    key: str
    mixed: Path
    heart: Path
    lung: Path


def unpack_dataset(archive: Path, destination: Path):
    """Unpack a UCI download and recursively unpack contained .zip files safely."""
    archive, destination = Path(archive), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    extracted = set()
    queue = [archive]
    while queue:
        z = queue.pop(0)
        if z.resolve() in extracted:
            continue
        extracted.add(z.resolve())
        with zipfile.ZipFile(z) as f:
            for member in f.infolist():
                target = (destination / member.filename).resolve()
                if not target.is_relative_to(destination.resolve()):
                    raise ValueError(f"Unsafe zip member {member.filename}")
            f.extractall(destination)
        for nested in destination.rglob("*.zip"):
            if nested.resolve() not in extracted and nested.resolve() not in map(lambda p:p.resolve(),queue):
                queue.append(nested)
    return destination


def _kind(path: Path):
    """Infer role from filename prefix H/L/M, independent of folder nesting."""
    m = re.match(r"^([HLM])(\d{4})$", path.stem.upper())
    return (m.group(1), m.group(2)) if m else (None, None)


def find_triplets(root: Path, strict=True):
    root = Path(root)
    wavs = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".wav"]
    indexed = {}
    for path in wavs:
        kind, key = _kind(path)
        if key is None:
            continue
        indexed.setdefault(key, {}).setdefault(kind, []).append(path)
    # Real UCI source pairs have H0001/L0001/M0001 ... H0145/L0145/M0145.
    # Other singleton H/L recordings are ignored unless a matching M exists.
    triplets = []
    for key, group in sorted(indexed.items()):
        if "M" not in group:
            continue
        if any(len(group.get(kind, [])) != 1 for kind in ("M","H","L")):
            if strict:
                raise ValueError(f"Ambiguous/missing matched audio for recording {key}: {group}")
            continue
        triplets.append(Triplet(key, group["M"][0], group["H"][0], group["L"][0]))
    if not triplets:
        raise FileNotFoundError("No M####.wav/H####.wav/L####.wav triplets found; inspect extracted UCI archive.")
    return triplets


def make_splits(triplets, seed=42, val_fraction=.15, test_fraction=.15):
    """Split IDs first; segments from a single recording are never across splits."""
    ids = [t.key for t in triplets]
    if len(ids) < 7:
        raise ValueError("Need at least seven distinct recordings to generate three held-out splits")
    train, holdout = train_test_split(ids, test_size=val_fraction+test_fraction, random_state=seed, shuffle=True)
    val, test = train_test_split(holdout, test_size=test_fraction/(val_fraction+test_fraction), random_state=seed, shuffle=True)
    assert not(set(train)&set(val) or set(train)&set(test) or set(val)&set(test))
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test), "seed": seed}


def save_manifest(triplets, path):
    data = [{"id": t.key, "mixed": str(t.mixed), "heart": str(t.heart), "lung": str(t.lung)} for t in triplets]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2))


class SegmentDataset(Dataset):
    def __init__(self, triplets, sample_rate=8000, seconds=2., training=False, hop_seconds=None):
        self.n = int(sample_rate * seconds)
        self.sr = sample_rate
        self.training = training
        self.records = []
        self.indices = []
        stride = int(sample_rate * (hop_seconds if hop_seconds else seconds))
        for rec in triplets:
            audio = align_by_length(*(load_audio(x, sample_rate) for x in (rec.mixed,rec.heart,rec.lung)))
            audio = shared_scale(*audio)
            idx = len(self.records)
            self.records.append((rec.key, audio))
            if len(audio[0]) <= self.n:
                self.indices.append((idx, 0))
            else:
                starts = list(range(0, len(audio[0])-self.n+1, stride))
                self.indices.extend((idx, pos) for pos in starts)
        if not self.indices:
            raise ValueError("No usable segments")

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        rec_idx, start = self.indices[i]
        key, signals = self.records[rec_idx]
        if self.training and len(signals[0]) > self.n:
            start = min(max(0, start + np.random.randint(-self.n//4,self.n//4+1)),len(signals[0])-self.n)
        batch = []
        for signal in signals:
            slice_ = signal[start:start+self.n]
            batch.append(np.pad(slice_, (0, max(0,self.n-len(slice_)))))
        import torch
        return torch.from_numpy(np.stack(batch).astype(np.float32)), key
