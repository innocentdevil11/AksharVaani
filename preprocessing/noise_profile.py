import numpy as np

def get_last_noise_block(x, fs):
    one_sec = fs
    if len(x) <= one_sec:
        print("⚠ Audio < 1s → using entire file as noise reference.")
        return x
    noise = x[-one_sec:]
    return noise / (np.max(np.abs(noise)) + 1e-8)
    