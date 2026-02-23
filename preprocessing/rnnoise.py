import numpy as np
from rnnoise import RNNoise

rn = RNNoise()

def apply_rnnoise(audio, fs=16000):
    frame_size = 480
    out = np.zeros_like(audio)

    for i in range(0, len(audio) - frame_size, frame_size):
        out[i:i+frame_size] = rn.process_frame(audio[i:i+frame_size])

    return out / (np.max(np.abs(out)) + 1e-8)
