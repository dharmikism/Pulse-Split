"""Validated upload handling and float WAV exports."""
import io
from math import gcd
import numpy as np
from scipy.signal import resample_poly
import soundfile as sf


def prepare_upload(payload, target_rate=8000, max_seconds=30):
    if not payload or len(payload) > 20 * 1024 * 1024:
        raise ValueError('Choose a nonempty WAV file smaller than 20 MB.')
    try:
        with sf.SoundFile(io.BytesIO(payload)) as file:
            if file.format not in {'WAV', 'WAVEX', 'RF64'}:
                raise ValueError('Please upload a WAV recording.')
            if file.frames == 0 or file.frames / file.samplerate > max_seconds:
                raise ValueError(f'Audio must be nonempty and at most {max_seconds} seconds.')
            if file.channels > 2:
                raise ValueError('Choose mono or stereo audio.')
            rate, channels = file.samplerate, file.channels
            audio = file.read(dtype='float32', always_2d=True)
    except RuntimeError as error:
        raise ValueError('Could not read this recording. Choose a valid WAV file.') from error
    if not np.isfinite(audio).all():
        raise ValueError('The recording contains invalid audio samples.')
    mono = audio.mean(axis=1)
    if float(np.max(np.abs(mono))) < 1e-7:
        raise ValueError('The recording is silent. Choose an audible signal.')
    if rate != target_rate:
        factor = gcd(rate, target_rate)
        mono = resample_poly(mono, target_rate // factor, rate // factor).astype(np.float32)
    scale = max(float(np.max(np.abs(mono))), 1e-6)
    return mono / scale, {'duration_seconds': len(audio) / rate,
                          'original_sample_rate': rate, 'channels': channels,
                          'sample_rate': target_rate, 'normalization_scale': scale}


def wav_bytes(signal, sample_rate):
    buffer = io.BytesIO()
    sf.write(buffer, signal, sample_rate, format='WAV', subtype='FLOAT')
    return buffer.getvalue()


def playback_bytes(signal, sample_rate, boost=True):
    """Encode browser-friendly PCM at 48 kHz; boost only the listening copy."""
    audio = np.asarray(signal, dtype=np.float32)
    factor = gcd(int(sample_rate), 48000)
    audio = resample_poly(audio, 48000 // factor, int(sample_rate) // factor)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if boost and peak > 1e-7:
        audio = audio * (0.9 / peak)
    audio = np.clip(audio, -1.0, 1.0)
    buffer = io.BytesIO()
    sf.write(buffer, audio, 48000, format='WAV', subtype='PCM_16')
    return buffer.getvalue()
