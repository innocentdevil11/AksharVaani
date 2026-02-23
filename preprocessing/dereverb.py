import numpy as np

def spectral_dereverb(x, alpha=0.35, smoothing=0.95):
    S = np.fft.rfft(x)
    mag = np.abs(S)
    phase = np.angle(S)

    decay = np.maximum(alpha * (mag[:-1] - mag[1:]), 0)
    decay = np.pad(decay, (0, 1), mode='edge')

    mag_clean = smoothing * mag + (1 - smoothing) * decay
    out = np.fft.irfft(mag_clean * np.exp(1j * phase))
    return out / (np.max(np.abs(out)) + 1e-8)
