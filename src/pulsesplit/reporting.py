"""Recording-level summaries and report exports for completed experiments."""
from pathlib import Path
import hashlib
import json

import numpy as np


def validate_splits(splits, available_ids):
    """Reject leakage, missing recordings, and empty evaluation partitions."""
    seen = set()
    for kind in ('train', 'val', 'test'):
        ids = splits.get(kind)
        if not isinstance(ids, list) or not ids or any(not isinstance(key, str) for key in ids):
            raise ValueError(f'Split {kind} must contain a nonempty list of recording IDs')
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError(f'Duplicate recording IDs or split leakage in {kind}')
        missing = set(ids) - set(available_ids)
        if missing:
            raise ValueError(f'Split {kind} references missing recordings: {sorted(missing)}')
        seen.update(ids)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def metric_statistics(values, seed=42, bootstrap_samples=2000):
    """Bootstrap recordings, not overlapping windows; exclude undefined scores."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    result = {'count': int(len(finite)), 'undefined_count': int(len(values) - len(finite)),
              'mean': None, 'median': None, 'std': None, 'mean_ci95': None}
    if not len(finite):
        return result
    result.update(mean=float(finite.mean()), median=float(np.median(finite)))
    if len(finite) > 1:
        result['std'] = float(finite.std(ddof=1))
        rng = np.random.default_rng(seed)
        # Chunking keeps memory bounded for large evaluation sets.
        means = np.concatenate([
            rng.choice(finite, size=(min(100, bootstrap_samples - start), len(finite)), replace=True).mean(axis=1)
            for start in range(0, bootstrap_samples, 100)
        ])
        result['mean_ci95'] = np.percentile(means, [2.5, 97.5]).tolist()
    return result


def write_evaluation_report(frame, summary, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    statistics = {column: metric_statistics(frame[column]) for column in frame if column != 'id'}
    summary['statistics'] = statistics
    summary['metrics'] = {column: stats['mean'] for column, stats in statistics.items()}
    summary['uncertainty_method'] = 'Percentile bootstrap of recording means; 2000 resamples; seed 42. No interval for n < 2.'
    summary['warning'] = 'Intervals assume independent recordings; repeated simulator patterns can violate this assumption. Undefined scores are excluded and counted.'
    (destination / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    lines = ['# Held-out source separation results', '',
             f'Data label: {summary["dataset_label"]}', '',
             f'Test recordings: {summary["num_test_recordings"]}', '',
             summary['disclaimer'], '',
             '| Metric (dB) | Valid n | Undefined | Mean | Median | SD | Mean 95% CI |',
             '|---|---:|---:|---:|---:|---:|---|']
    def number(value):
        return 'N/A' if value is None else f'{value:.3f}'
    for name, stats in statistics.items():
        interval = stats['mean_ci95']
        ci = 'N/A' if interval is None else f'[{interval[0]:.3f}, {interval[1]:.3f}]'
        lines.append(f'| {name} | {stats["count"]} | {stats["undefined_count"]} | {number(stats["mean"])} | {number(stats["median"])} | {number(stats["std"])} | {ci} |')
    lines.extend(['', summary['uncertainty_method'], '', summary['warning'], '',
                  f'Checkpoint SHA-256: `{summary["checkpoint_sha256"]}`', '',
                  f'Split SHA-256: `{summary["split_sha256"]}`', ''])
    (destination / 'evaluation_report.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary


def plot_training_history(history, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(8, 4))
    for key, label in [('train_logmag_mae', 'Training'), ('val_logmag_mae', 'Validation')]:
        axis.plot([row['epoch'] for row in history], [row[key] for row in history], label=label)
    axis.set(xlabel='Epoch', ylabel='Log-magnitude L1 loss', title='Training and validation loss')
    axis.legend()
    fig.tight_layout()
    fig.savefig(Path(destination) / 'training_history.png', dpi=160)
    plt.close(fig)
