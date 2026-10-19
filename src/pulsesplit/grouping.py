"""Deterministic held-out splits that keep identical recorded sources together."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math
import random


def _file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact_partition(order, sizes, test_target, val_target):
    """Two-dimensional subset sum; every component can enter at most one set."""
    if len(order) > 200 or (test_target + 1) * (val_target + 1) > 250_000:
        return None
    states = {(0, 0): (0, 0)}
    for position, group in enumerate(order):
        size = sizes[group]
        bit = 1 << position
        for (test_size, val_size), (test_mask, val_mask) in list(states.items()):
            if test_size + size <= test_target:
                states.setdefault((test_size + size, val_size), (test_mask | bit, val_mask))
            if val_size + size <= val_target:
                states.setdefault((test_size, val_size + size), (test_mask, val_mask | bit))
        if (test_target, val_target) in states:
            test_mask, val_mask = states[(test_target, val_target)]
            return {
                "test": [group for i, group in enumerate(order) if test_mask & (1 << i)],
                "val": [group for i, group in enumerate(order) if val_mask & (1 << i)],
                "train": [group for i, group in enumerate(order) if not (test_mask | val_mask) & (1 << i)],
            }
    return None


def _closest_subset(order, sizes, target, reserve):
    """Choose a nonempty set while reserving components for other partitions."""
    limit = len(order) - reserve
    if len(order) > 200 or sum(sizes[group] for group in order) > 100_000:
        selected = [min(order, key=lambda group: abs(sizes[group] - target))]
        current = sizes[selected[0]]
        for group in order:
            if (group not in selected and len(selected) < limit
                    and abs(current + sizes[group] - target) < abs(current - target)):
                selected.append(group)
                current += sizes[group]
        return selected
    states = {0: ()}
    for group in order:
        for total, selected in list(states.items()):
            if len(selected) >= limit:
                continue
            value = total + sizes[group]
            candidate = (*selected, group)
            if value not in states or len(candidate) < len(states[value]):
                states[value] = candidate
    best = min((total for total in states if total > 0),
               key=lambda total: (abs(total - target), total > target, len(states[total])))
    return list(states[best])


def _partition(groups, sizes, targets, seed):
    randomizer = random.Random(seed)
    order = sorted(groups)
    randomizer.shuffle(order)
    exact = _exact_partition(order, sizes, targets["test"], targets["val"])
    if exact is not None:
        return exact
    best = None
    best_score = None
    # When exact fractions are impossible, compare sizes only, never outcomes.
    for attempt in range(32 if len(order) <= 200 else 8):
        randomizer.shuffle(order)
        first, second = ("test", "val") if attempt % 2 else ("val", "test")
        candidate = {first: _closest_subset(order, sizes, targets[first], reserve=2)}
        remaining = [group for group in order if group not in candidate[first]]
        candidate[second] = _closest_subset(remaining, sizes, targets[second], reserve=1)
        candidate["train"] = [group for group in remaining if group not in candidate[second]]
        deviations = [abs(sum(sizes[group] for group in candidate[name]) - targets[name])
                      for name in ("train", "val", "test")]
        score = (sum(deviations), max(deviations))
        if best_score is None or score < best_score:
            best, best_score = candidate, score
    return best


def make_source_group_splits(triplets, seed=42, val_fraction=.15, test_fraction=.15):
    """Keep transitive components of byte-identical M/H/L files in one split.

    Hashes are compared across roles too. This protects against reusing the same
    exact recorded source under a different ID; it does not establish independent
    patients or independent simulator sound templates. Fractions target recording
    counts and may differ when indivisible component sizes prevent exact totals.
    """
    if (not math.isfinite(val_fraction) or not math.isfinite(test_fraction)
            or val_fraction <= 0 or test_fraction <= 0 or val_fraction + test_fraction >= 1):
        raise ValueError("Validation and test fractions must be positive and sum to less than one")
    triplets = sorted(triplets, key=lambda item: item.key)
    ids = [item.key for item in triplets]
    if len(ids) != len(set(ids)):
        raise ValueError("Recording IDs must be unique")
    parents = {key: key for key in ids}

    def find(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    hashes = {}
    first_recording = {}
    for item in triplets:
        hashes[item.key] = {}
        for role in ("mixed", "heart", "lung"):
            digest = _file_hash(getattr(item, role))
            hashes[item.key][role] = digest
            if digest in first_recording:
                left, right = find(item.key), find(first_recording[digest])
                parents[max(left, right)] = min(left, right)
            else:
                first_recording[digest] = item.key

    components = defaultdict(list)
    for key in ids:
        components[find(key)].append(key)
    groups = dict(sorted(components.items()))
    if len(groups) < 3:
        raise ValueError(f"Need at least three independent source components for train/val/test; found {len(groups)}")
    sizes = {group: len(members) for group, members in groups.items()}
    targets = {"val": max(1, round(len(ids) * val_fraction)),
               "test": max(1, round(len(ids) * test_fraction))}
    while targets["val"] + targets["test"] >= len(ids):
        larger = max(("val", "test"), key=lambda name: targets[name])
        targets[larger] -= 1
    targets["train"] = len(ids) - targets["val"] - targets["test"]
    partition = _partition(groups, sizes, targets, seed)
    split = {name: sorted(key for group in partition[name] for key in groups[group])
             for name in ("train", "val", "test")}

    assigned_hashes = {}
    for name in ("train", "val", "test"):
        if not split[name]:
            raise ValueError("Could not form three nonempty source-independent splits")
        for key in split[name]:
            for digest in hashes[key].values():
                if digest in assigned_hashes and assigned_hashes[digest] != name:
                    raise RuntimeError("Identical recorded source assigned to different splits")
                assigned_hashes[digest] = name
    return {
        **split, "seed": seed, "strategy": "exact-source-connected-components",
        "groups": groups, "group_sizes": sizes,
        "group_assignments": {group: name for name, members in partition.items() for group in members},
        "target_sizes": targets,
        "actual_sizes": {name: len(keys) for name, keys in split.items()},
        "file_hashes": hashes,
        "limitation": "Groups exclude byte-identical recorded sources across splits; simulator templates and patients are not independently held out.",
    }
