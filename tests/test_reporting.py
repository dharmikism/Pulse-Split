import json
import numpy as np
import pandas as pd
import pytest
from pulsesplit.reporting import metric_statistics, validate_splits, write_evaluation_report


def test_statistics_count_undefined_and_are_reproducible():
    stats = metric_statistics([1, 2, 3, np.nan, np.inf])
    assert stats['count'] == 3 and stats['undefined_count'] == 2
    assert stats['mean'] == 2 and stats['median'] == 2 and stats['std'] == 1
    assert stats['mean_ci95'][0] <= 2 <= stats['mean_ci95'][1]
    assert stats == metric_statistics([1, 2, 3, np.nan, np.inf])
    assert metric_statistics([5])['mean_ci95'] is None
    assert metric_statistics([np.nan])['mean'] is None


@pytest.mark.parametrize('splits', [
    {'train': ['1'], 'val': ['1'], 'test': ['3']},
    {'train': ['1'], 'val': ['2'], 'test': []},
    {'train': ['1'], 'val': ['2'], 'test': ['missing']},
    {'train': ['1', '1'], 'val': ['2'], 'test': ['3']},
])
def test_invalid_splits_rejected(splits):
    with pytest.raises(ValueError):
        validate_splits(splits, ['1', '2', '3'])


def test_evaluation_report_handles_undefined_scores(tmp_path):
    frame = pd.DataFrame({'id': ['1', '2'], 'heart_si_sdr': [float('nan'), float('nan')]})
    summary = {'dataset_label': 'Synthetic fixture', 'num_test_recordings': 2,
               'disclaimer': 'Software check only', 'checkpoint_sha256': 'abc', 'split_sha256': 'def'}
    write_evaluation_report(frame, summary, tmp_path)
    data = json.loads((tmp_path / 'summary.json').read_text())
    assert data['metrics']['heart_si_sdr'] is None
    assert data['statistics']['heart_si_sdr']['undefined_count'] == 2
    assert 'N/A' in (tmp_path / 'evaluation_report.md').read_text()
