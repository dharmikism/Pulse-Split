import io
import numpy as np
import pytest
import soundfile as sf
from pulsesplit.demo import prepare_upload, wav_bytes


def test_stereo_resampling_and_export():
    time = np.arange(16000) / 16000
    signal = .2 * np.sin(2 * np.pi * 100 * time)
    mixed, info = prepare_upload(wav_bytes(np.column_stack([signal, signal]), 16000))
    assert mixed.shape == (8000,)
    assert info['channels'] == 2
    assert info['duration_seconds'] == 1
    decoded, rate = sf.read(io.BytesIO(wav_bytes(mixed * 1.5, 8000)))
    assert rate == 8000
    assert np.allclose(decoded, mixed * 1.5)


@pytest.mark.parametrize('payload', [b'', b'not audio', wav_bytes(np.zeros(1000), 8000),
                                    wav_bytes(np.ones(8001), 8000)],
                         ids=['empty', 'corrupt', 'silent', 'too-long'])
def test_invalid_uploads(payload):
    with pytest.raises(ValueError):
        prepare_upload(payload, max_seconds=1)


def test_nonfinite_audio_rejected():
    with pytest.raises(ValueError, match='invalid audio'):
        prepare_upload(wav_bytes(np.array([np.nan, .1]), 8000))


def test_browser_playback_encoding_and_boost():
    from pulsesplit.demo import playback_bytes
    signal = (.02 * np.sin(2 * np.pi * 90 * np.arange(8000) / 8000)).astype(np.float32)
    original = signal.copy()
    boosted = playback_bytes(signal, 8000)
    with sf.SoundFile(io.BytesIO(boosted)) as file:
        assert file.subtype == 'PCM_16'
        assert file.samplerate == 48000
        assert file.frames == 48000
        audio = file.read()
    assert np.max(np.abs(audio)) == pytest.approx(.9, abs=1e-4)
    quiet, rate = sf.read(io.BytesIO(playback_bytes(signal, 8000, boost=False)))
    assert np.max(np.abs(quiet)) < .021
    assert np.array_equal(signal, original)


def test_silent_playback_stays_silent():
    from pulsesplit.demo import playback_bytes
    audio, rate = sf.read(io.BytesIO(playback_bytes(np.zeros(8000), 8000)))
    assert np.isfinite(audio).all()
    assert not audio.any()
