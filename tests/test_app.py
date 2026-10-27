"""Exercise the demo with a temporary fixture checkpoint, never a real model."""
from pathlib import Path
from unittest.mock import patch
import io
import numpy as np
import pytest
import torch

pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest
from pulsesplit.demo import wav_bytes
from pulsesplit.model import MaskUNet


def app_source(root):
    source = (Path(__file__).resolve().parents[1] / 'app.py').read_text(encoding='utf-8')
    return source.replace('ROOT = Path(__file__).resolve().parent', f'ROOT = Path({str(root)!r})')


def test_missing_model_guidance(tmp_path):
    app = AppTest.from_string(app_source(tmp_path)).run(timeout=30)
    assert not app.exception
    assert 'trained model' in app.warning[0].value


def test_upload_separation_and_view_switch(tmp_path):
    folder = tmp_path / 'artifacts'
    folder.mkdir()
    torch.save({'state_dict': MaskUNet(base=4).state_dict(), 'base': 4,
                'sample_rate': 8000, 'seconds': .5}, folder / 'best.pt')
    signal = (.2 * np.sin(2 * np.pi * 90 * np.arange(8000) / 8000)).astype(np.float32)
    with patch('streamlit.file_uploader', return_value=io.BytesIO(wav_bytes(signal, 8000))):
        app = AppTest.from_string(app_source(tmp_path)).run(timeout=30)
        app.button[0].click().run(timeout=30)
        assert not app.exception
        assert len(app.get('audio')) == 3
        assert len(app.get('download_button')) == 4
        app.radio(key='visualization').set_value('Waveform').run(timeout=30)
        assert not app.exception
        assert len(app.get('audio')) == 3


def test_saved_examples_without_checkpoint(tmp_path):
    import json
    folder = tmp_path / 'examples'
    folder.mkdir()
    signal = (.2 * np.sin(2 * np.pi * 90 * np.arange(8000) / 8000)).astype(np.float32)
    for name in ('mixture', 'heart', 'lung'):
        (folder / f'{name}.wav').write_bytes(wav_bytes(signal, 8000))
    entry = {'id': 'fixture', 'title': 'Stored fixture', 'description': 'Synthetic UI test only',
             'checkpoint_sha256': 'fixture', 'mixture': 'mixture.wav',
             'heart': 'heart.wav', 'lung': 'lung.wav'}
    (folder / 'manifest.json').write_text(json.dumps([entry]))
    app = AppTest.from_string(app_source(tmp_path)).run(timeout=30)
    assert not app.exception
    assert len(app.get('audio')) == 3
    assert len(app.get('download_button')) == 4
    app.radio(key='visualization').set_value('Waveform').run(timeout=30)
    assert not app.exception
    assert len(app.get('audio')) == 3


def test_switch_from_saved_example_to_upload(tmp_path):
    import json
    folder = tmp_path / 'examples'
    folder.mkdir()
    signal = (.2 * np.sin(2 * np.pi * 90 * np.arange(8000) / 8000)).astype(np.float32)
    for name in ('mixture', 'heart', 'lung'):
        (folder / f'{name}.wav').write_bytes(wav_bytes(signal, 8000))
    entry = {'id': 'fixture', 'title': 'Stored fixture', 'description': 'Synthetic UI test only',
             'checkpoint_sha256': 'fixture', 'mixture': 'mixture.wav',
             'heart': 'heart.wav', 'lung': 'lung.wav'}
    (folder / 'manifest.json').write_text(json.dumps([entry]))
    app = AppTest.from_string(app_source(tmp_path)).run(timeout=30)
    assert not app.exception
    assert len(app.get('audio')) == 3
    app.radio(key='audio_source').set_value('Upload WAV').run(timeout=30)
    assert not app.exception
    assert len(app.get('audio')) == 0
    assert 'trained model' in app.warning[0].value
    app.radio(key='audio_source').set_value('Saved examples').run(timeout=30)
    assert not app.exception
    assert len(app.get('audio')) == 3
