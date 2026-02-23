import numpy as np

def stationary_noise_filter(x, noise, alpha=2.5, beta=0.35, smooth=0.85):
    if len(noise) < len(x):
        r = int(np.ceil(len(x) / len(noise)))
        noise = np.tile(noise, r)[:len(x)]

    X = np.fft.rfft(x)
    N = np.fft.rfft(noise)

    Px = np.abs(X) ** 2
    Pn = np.abs(N) ** 2

    G = Px / (Px + alpha * Pn + 1e-8)
    G = np.clip(G, beta, 1.0)
    G = smooth * G + (1 - smooth) * np.mean(G)

    out = np.fft.irfft(G * X)
    return out / (np.max(np.abs(out)) + 1e-8)


def spectral_gate(x, noise, threshold_db=-22, reduction_db=-12):
    X = np.fft.rfft(x)
    N = np.fft.rfft(noise[:len(x)])

    mag = np.abs(X)
    phase = np.angle(X)
    noise_mag = np.abs(N)

    threshold = noise_mag * (10 ** (threshold_db / 20))
    reduction = 10 ** (reduction_db / 20)

    mask = np.where(mag < threshold, reduction, 1.0)
    out = np.fft.irfft(mask * mag * np.exp(1j * phase))
    return out / (np.max(np.abs(out)) + 1e-8)
