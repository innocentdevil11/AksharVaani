import numpy as np
from scipy.signal import butter, lfilter

def highpass_filter(sig, cutoff=250, fs=16000, order=3):
    nyq = fs / 2
    b, a = butter(order, cutoff / nyq, btype='highpass')
    return lfilter(b, a, sig).astype(np.float32)
