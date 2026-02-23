import numpy as np
import webrtcvad

def nlms_fast(x, fs, mu=0.03, filter_len=32):
    vad = webrtcvad.Vad(1)
    w = np.zeros(filter_len)
    cleaned = np.zeros_like(x)
    FRAME = int(0.02 * fs)
    FRAME -= FRAME % 2

    for i in range(0, len(x) - FRAME - filter_len, FRAME):
        frame = x[i:i+FRAME]
        x_vec = x[i:i+filter_len]
        y = np.dot(w, x_vec)
        e = frame - y
        cleaned[i:i+FRAME] = e

        fb = (frame * 32767).astype(np.int16).tobytes()
        is_speech = vad.is_speech(fb, fs)
        adapt = 0.5 if is_speech else 1.0

        w += adapt * (mu / (np.dot(x_vec, x_vec) + 1e-6)) * np.mean(e) * x_vec

    return cleaned / (np.max(np.abs(cleaned)) + 1e-8)
