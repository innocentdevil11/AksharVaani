"""
Audio Preprocessing Module
==========================
Handles format conversion (MP3/M4A → WAV), resampling to 16 kHz mono,
and waveform normalisation.  Acts as the first stage of the audio pipeline.

Usage:
    python -m aksharvaani.src.audio.preprocessor --input lecture.mp3 --output cleaned.wav
"""

import os
import librosa
import soundfile as sf


def convert_to_wav(input_file: str, output_file: str, sample_rate: int = 16000) -> str:
    """Convert any audio format to 16 kHz mono WAV using ffmpeg.

    Parameters
    ----------
    input_file : str
        Path to the source audio file (MP3, M4A, WAV, etc.).
    output_file : str
        Destination path for the converted WAV file.
    sample_rate : int, optional
        Target sample rate (default 16 000 Hz — Whisper optimal).

    Returns
    -------
    str
        Path to the converted WAV file.
    """
    try:
        import ffmpeg

        ffmpeg.input(input_file).output(
            output_file, ar=sample_rate, ac=1
        ).overwrite_output().run(quiet=True)
    except ImportError:
        # Fallback: use librosa directly
        audio, sr = librosa.load(input_file, sr=sample_rate, mono=True)
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        sf.write(output_file, audio, sr)

    return output_file


def preprocess_audio(input_file: str, output_file: str, sample_rate: int = 16000) -> str:
    """Full preprocessing: convert → load → normalise → save.

    Parameters
    ----------
    input_file : str
        Raw classroom recording.
    output_file : str
        Path for the cleaned WAV output.
    sample_rate : int, optional
        Target sample rate (default 16 000 Hz).

    Returns
    -------
    str
        Path to the cleaned audio file.
    """
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    # Load and resample
    audio, sr = librosa.load(input_file, sr=sample_rate, mono=True)

    # Save cleaned audio
    sf.write(output_file, audio, sr)
    print(f"✅ Preprocessed audio saved → {output_file}  "
          f"(duration: {len(audio) / sr:.1f}s, sr: {sr})")

    return output_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess classroom audio")
    parser.add_argument("--input", required=True, help="Source audio file")
    parser.add_argument("--output", required=True, help="Output WAV path")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate")
    args = parser.parse_args()

    preprocess_audio(args.input, args.output, args.sr)
