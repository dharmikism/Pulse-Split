"""Audio utility code. All signals are mono and peak scaling is shared per triplet."""
from __future__ import annotations
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd


def load_audio(path, sample_rate=8000):
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != sample_rate:
        g = gcd(int(sr), int(sample_rate))
        x = resample_poly(x, sample_rate // g, sr // g).astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def shared_scale(mix, heart, lung):
    """Scale all three channels using one factor; never independently normalize labels."""
    scale = max(float(np.max(np.abs(mix))), 1e-6)
    return (mix / scale).astype(np.float32), (heart / scale).astype(np.float32), (lung / scale).astype(np.float32)


def align_by_length(*signals):
    n = min(map(len, signals))
    if n < 1:
        raise ValueError("Empty audio")
    return tuple(x[:n] for x in signals)


def si_sdr(reference, estimate):
    """Scale-invariant SDR (dB); near-silent references return nan."""
    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimate, dtype=np.float64)
    ref, est = align_by_length(ref, est)
    ref, est = ref - ref.mean(), est - est.mean()
    energy = np.dot(ref, ref)
    if energy < 1e-10:
        return float("nan")
    projection = np.dot(est, ref) / (energy + 1e-12) * ref
    noise = est - projection
    return float(10 * np.log10((np.dot(projection, projection) + 1e-10) / (np.dot(noise, noise) + 1e-10)))
