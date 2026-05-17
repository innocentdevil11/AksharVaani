"""
Audio Denoising Module
======================
Provides multiple denoising strategies (Wiener, spectral subtraction,
bandpass filtering) with SNR-based comparison.

Designed for Indian classroom environments with high ambient noise,
fan hum, and student chatter.

Usage:
    python -m aksharvaani.src.audio.denoiser --input noisy.wav --output denoised/
"""

import os
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, lfilter

try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except ImportError:
    HAS_NOISEREDUCE = False


# ─── Denoising Strategies ───────────────────────────────────────────────────

def denoise_wiener(audio: np.ndarray, sr: int) -> np.ndarray:
    """Non-stationary Wiener filter (adaptive noise estimation)."""
    if not HAS_NOISEREDUCE:
        raise ImportError("noisereduce is required: pip install noisereduce")
    return nr.reduce_noise(y=audio, sr=sr, stationary=False)


def denoise_spectral(audio: np.ndarray, sr: int) -> np.ndarray:
    """Stationary spectral gating (assumes consistent noise floor)."""
    if not HAS_NOISEREDUCE:
        raise ImportError("noisereduce is required: pip install noisereduce")
    return nr.reduce_noise(y=audio, sr=sr, stationary=True)


def denoise_bandpass(audio: np.ndarray, sr: int,
                     lowcut: int = 300, highcut: int = 3400,
                     order: int = 5) -> np.ndarray:
    """Butterworth bandpass filter targeting human speech frequencies."""
    nyq = 0.5 * sr
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return lfilter(b, a, audio)


def spectral_subtraction(audio: np.ndarray, sr: int,
                         noise_frames: int = 30,
                         gain: float = 2.0) -> np.ndarray:
    """STFT-based spectral subtraction with optional amplification.

    Parameters
    ----------
    audio : np.ndarray
        Input audio waveform.
    sr : int
        Sample rate.
    noise_frames : int
        Number of initial STFT frames used to estimate noise spectrum.
    gain : float
        Amplification factor applied after subtraction.
    """
    D = librosa.stft(audio)
    magnitude, phase = librosa.magphase(D)

    noise_mag = np.mean(np.abs(magnitude[:, :noise_frames]), axis=1, keepdims=True)
    clean_mag = np.maximum(np.abs(magnitude) - noise_mag, 0.0)
    clean_D = clean_mag * phase
    y_clean = librosa.istft(clean_D)

    return np.clip(y_clean * gain, -1.0, 1.0)


# ─── Evaluation ─────────────────────────────────────────────────────────────

def calculate_snr(original: np.ndarray, processed: np.ndarray) -> float:
    """Estimate Signal-to-Noise Ratio improvement (dB)."""
    noise = processed - original[:len(processed)]
    return float(10 * np.log10(
        np.sum(original[:len(processed)] ** 2) / (np.sum(noise ** 2) + 1e-10)
    ))


# ─── Pipeline ───────────────────────────────────────────────────────────────

def run_denoising_comparison(input_file: str, output_dir: str = "denoised") -> dict:
    """Run all denoising methods and save results with SNR comparison.

    Returns
    -------
    dict
        Mapping of method name → output path.
    """
    os.makedirs(output_dir, exist_ok=True)
    audio, sr = librosa.load(input_file, sr=16000)

    methods = {
        "wiener": denoise_wiener,
        "spectral": denoise_spectral,
        "bandpass": denoise_bandpass,
    }

    results = {}
    for name, fn in methods.items():
        try:
            processed = fn(audio, sr)
            out_path = os.path.join(output_dir, f"{name}.wav")
            sf.write(out_path, processed, sr)
            snr = calculate_snr(audio, processed)
            results[name] = {"path": out_path, "snr_db": round(snr, 2)}
            print(f"  ✅ {name:12s} → SNR {snr:+.2f} dB  ({out_path})")
        except Exception as e:
            results[name] = {"error": str(e)}
            print(f"  ❌ {name:12s} → {e}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare audio denoising methods")
    parser.add_argument("--input", required=True, help="Noisy audio file")
    parser.add_argument("--output", default="denoised", help="Output directory")
    args = parser.parse_args()

    run_denoising_comparison(args.input, args.output)
