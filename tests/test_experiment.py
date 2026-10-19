import json
import numpy as np
import pytest
import soundfile as sf
import torch
from pulsesplit.pipeline import train, evaluate, separate_signal
from pulsesplit.model import MaskUNet


def test_train_evaluate_exports_and_rejects_leakage(tmp_path):
    torch.set_num_threads(1)
    root = tmp_path / 'audio'
    root.mkdir()
    time = np.arange(1024) / 8000
    for index in range(9):
        heart = .2 * np.sin(2 * np.pi * (70 + index) * time)
        lung = .1 * np.sin(2 * np.pi * (200 + index) * time)
        for kind, signal in [('M', heart + lung), ('H', heart), ('L', lung)]:
            sf.write(root / f'{kind}{index:04d}.wav', signal, 8000)
    artifact = tmp_path / 'model'
    frozen_split = {'train': [f'{i:04d}' for i in range(5)],
                    'val': ['0005', '0006'], 'test': ['0007', '0008'],
                    'seed': 42, 'strategy': 'precomputed source groups'}
    checkpoint = train(root, artifact, epochs=1, base=2, batch_size=2, seconds=.128,
                       dataset_label='Synthetic unit-test fixture',split_manifest=frozen_split)
    summary = evaluate(root, checkpoint, tmp_path / 'evaluation')
    assert summary['dataset_label'] == 'Synthetic unit-test fixture'
    assert summary['num_test_recordings'] == 2
    assert len(summary['checkpoint_sha256']) == 64
    assert summary['statistics']['heart_improvement']['count'] == 2
    assert (artifact / 'training_history.png').exists()
    metadata = json.loads((artifact / 'run_metadata.json').read_text())
    assert metadata['parameter_count'] > 0 and metadata['runtime_seconds'] > 0
    assert metadata['split_strategy'] == 'precomputed source groups'
    assert json.loads((artifact / 'split.json').read_text()) == frozen_split
    assert 'Synthetic unit-test fixture' in (tmp_path / 'evaluation/evaluation_report.md').read_text()
    splits = json.loads((artifact / 'split.json').read_text())
    splits['test'][0] = splits['train'][0]
    (artifact / 'split.json').write_text(json.dumps(splits))
    with pytest.raises(ValueError, match='leakage'):
        evaluate(root, checkpoint, tmp_path / 'bad')


@pytest.mark.parametrize('audio,seconds', [(np.zeros((2, 10)), 1),
                                        (np.array([np.nan]), 1), (np.zeros(10), 0)])
def test_invalid_inference_arguments(audio, seconds):
    with pytest.raises(ValueError):
        separate_signal(MaskUNet(base=2), audio, seconds=seconds)
