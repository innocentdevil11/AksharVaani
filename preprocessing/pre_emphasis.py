import numpy as np

def pre_emphasis(sig, alpha=0.92):
    sig = sig.astype(np.float32)
    if sig.ndim > 1:
        sig = sig.mean(axis=1)
    sig = sig / (np.max(np.abs(sig)) + 1e-8)
    out = np.append(sig[0], sig[1:] - alpha * sig[:-1])
    return out / (np.max(np.abs(out)) + 1e-8)
