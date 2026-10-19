from pathlib import Path

import pytest

from pulsesplit.data import Triplet
from pulsesplit.grouping import make_source_group_splits


def _records(tmp_path, count):
    records = []
    for index in range(count):
        key = f"{index + 1:04d}"
        paths = []
        for role in ("M", "H", "L"):
            path = tmp_path / f"{role}{key}.wav"
            path.write_bytes(f"unique recorded {role} source {key}".encode())
            paths.append(path)
        records.append(Triplet(key, *paths))
    return records


def _assert_hash_independence(split):
    seen = {}
    for name in ("train", "val", "test"):
        assert split[name]
        for key in split[name]:
            for digest in split["file_hashes"][key].values():
                assert digest not in seen or seen[digest] == name
                seen[digest] = name


def test_source_split_is_deterministic_and_covers_each_record_once(tmp_path):
    records = _records(tmp_path, 20)
    split = make_source_group_splits(records)
    assert split == make_source_group_splits(list(reversed(records)))
    assert split["actual_sizes"] == {"train": 14, "val": 3, "test": 3}
    assert split["strategy"] == "exact-source-connected-components"
    assert sorted(split["train"] + split["val"] + split["test"]) == [item.key for item in records]
    _assert_hash_independence(split)


def test_source_split_unites_transitive_and_cross_role_duplicates(tmp_path):
    records = _records(tmp_path, 9)
    records[1].heart.write_bytes(records[0].heart.read_bytes())
    records[2].lung.write_bytes(records[1].lung.read_bytes())
    # The third and fourth records share bytes under different source roles.
    records[3].mixed.write_bytes(records[2].heart.read_bytes())
    split = make_source_group_splits(records, val_fraction=.2, test_fraction=.2)
    assert split["groups"]["0001"] == ["0001", "0002", "0003", "0004"]
    assert split["group_sizes"]["0001"] == 4
    assert any(set(split[name]) >= {"0001", "0002", "0003", "0004"} for name in ("train", "val", "test"))
    _assert_hash_independence(split)


def test_source_split_finds_exact_two_dimensional_partition(tmp_path):
    records = _records(tmp_path, 16)
    # Components of sizes 5, 4, 3, 2, 1, 1 permit val=5 and test=5.
    start = 0
    for size in (5, 4, 3, 2, 1, 1):
        for record in records[start:start + size]:
            record.heart.write_bytes(f"component-{start}".encode())
        start += size
    split = make_source_group_splits(records, val_fraction=.3125, test_fraction=.3125)
    assert split["actual_sizes"] == {"train": 6, "val": 5, "test": 5}
    _assert_hash_independence(split)


def test_source_split_preserves_components_when_exact_sizes_impossible(tmp_path):
    records = _records(tmp_path, 10)
    for index, record in enumerate(records):
        record.lung.write_bytes(f"group-{min(index // 4, 2)}".encode())
    split = make_source_group_splits(records)
    assert sorted(split["actual_sizes"].values()) == [2, 4, 4]
    assert split == make_source_group_splits(records)
    _assert_hash_independence(split)


def test_source_split_rejects_insufficient_independent_sources(tmp_path):
    records = _records(tmp_path, 5)
    for record in records:
        record.heart.write_bytes(b"same heart everywhere")
    with pytest.raises(ValueError, match="at least three independent source components.*found 1"):
        make_source_group_splits(records)
    with pytest.raises(ValueError, match="IDs must be unique"):
        make_source_group_splits([records[0], records[0]])


@pytest.mark.parametrize("val,test", [(0, .15), (.15, -1), (.6, .4), (float("nan"), .15)])
def test_source_split_rejects_invalid_fractions(tmp_path, val, test):
    with pytest.raises(ValueError, match="fractions"):
        make_source_group_splits(_records(tmp_path, 3), val_fraction=val, test_fraction=test)
