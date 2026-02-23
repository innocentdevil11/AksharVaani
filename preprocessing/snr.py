import numpy as np

def compute_snr(signal, noise):
    return 10 * np.log10(
        np.mean(signal ** 2) / (np.mean(noise ** 2) + 1e-8)
    )
