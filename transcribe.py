# transcribe.py

import torch
from faster_whisper import WhisperModel
# from preprocessing.voicefingerprint import filter_professor_segments  # your fingerprint module

PROFESSOR_REF = "audio/clean_tanmay.wav"
THRESHOLD = 0.35


def transcribe(audio_path="audio/huClassSample.wav",threshold=THRESHOLD):
    """
    Transcribe with Distil-Whisper + Professor Voice Fingerprint.
    Returns list of professor-only segments: {start, end, text}
    """
    # 1. Load Distil-Whisper model (CPU, CT2 optimized)
    print("🔄 Loading Distil-Whisper...")
    model = WhisperModel("medium", device="cpu")
    
    # 2. Transcribe with translation
    print("🎤 Transcribing...")
    segments, info = model.transcribe(audio_path, task="translate")
    
    print("Detected language:", info.language)
    print("Language probability:", info.language_probability)
    
    # 3. Collect all segments
    all_segments = []
    for segment in segments:
        all_segments.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })
    
    # 4. Voice fingerprint → keep only professor segments
    print("🆔 Running voice fingerprint...")
    prof_segments = filter_professor_segments(audio_path, all_segments, threshold=THRESHOLD)
    
    return prof_segments



def transcribe_without_fingerprint(audio_path="audio/huClassSample.wav"):
    """
    Pure Distil-Whisper transcription (no speaker filtering).
    Returns list of {start, end, text}.
    """
    model = WhisperModel("medium", device="cpu")
    
    print("🎤 Transcribing...")
    segments, info = model.transcribe(audio_path, task="translate")
    
    print("Detected language:", info.language)
    print("Language probability:", info.language_probability)
    
    transcript = []
    for segment in segments:
        transcript.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })
    
    return transcript


if __name__ == "__main__":
    transcript = transcribe_without_fingerprint("audio/huClassSample.wav")
    
    print("\n📝 FULL TRANSCRIPT (NO FINGERPRINT):")
    for seg in transcript:
        print(f"[{seg['start']:.2f}s -> {seg['end']:.2f}s] {seg['text']}")
    
    print(f"\n✅ {len(transcript)} total segments")