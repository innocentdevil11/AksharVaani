"""
Speech-to-Text Transcription Module
====================================
Professor speech isolation + Whisper transcription + speaker diarisation.

Supports two modes:
  1. **Speaker-verified**: Uses a reference voice clip to isolate professor speech
     via ECAPA-TDNN speaker embeddings.
  2. **Diarised**: Uses Pyannote speaker diarisation to label multiple speakers,
     then keeps only the dominant (professor) speaker.

Usage:
    python -m aksharvaani.src.audio.transcriber \\
        --input lecture.wav --ref professor_sample.wav --output transcript.json
"""

import os
import json
import torch
from typing import List, Dict, Optional


def transcribe_with_speaker_isolation(
    audio_path: str,
    professor_ref: str,
    whisper_model: str = "medium",
    threshold: float = 0.35,
    output_json: Optional[str] = None,
) -> List[Dict]:
    """Transcribe audio, keeping only professor segments.

    Parameters
    ----------
    audio_path : str
        Path to the classroom recording (WAV preferred).
    professor_ref : str
        Short clean audio clip of the professor's voice.
    whisper_model : str
        Whisper model size: tiny | base | small | medium | large-v2.
    threshold : float
        Speaker verification confidence threshold.
    output_json : str, optional
        Path to save the transcript JSON.

    Returns
    -------
    list[dict]
        List of ``{start, end, text}`` segments attributed to the professor.
    """
    from faster_whisper import WhisperModel
    from speechbrain.inference import SpeakerRecognition
    from pydub import AudioSegment

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    # Load models
    model = WhisperModel(whisper_model, device=device, compute_type=compute_type)
    verifier = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
    )

    # Transcribe
    print(f"🎙️  Transcribing with Whisper ({whisper_model}) on {device}…")
    segments, info = model.transcribe(audio_path, task="translate")
    print(f"   Detected language: {info.language} (p={info.language_probability:.2f})")

    results = []
    for seg in segments:
        tmp = f"_chunk_{int(seg.start * 100)}.wav"
        audio = AudioSegment.from_file(audio_path)
        chunk = audio[seg.start * 1000 : seg.end * 1000]
        chunk = chunk.set_channels(1).set_frame_rate(16000)
        chunk.export(tmp, format="wav")

        score, prediction = verifier.verify_files(professor_ref, tmp)
        if prediction and score > threshold:
            results.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
            })

        os.remove(tmp)

    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"📂 Transcript saved → {output_json}")

    return results


def transcribe_with_diarisation(
    audio_path: str,
    whisper_model: str = "medium",
    pyannote_model: str = "pyannote/speaker-diarization-3.1",
    output_json: Optional[str] = None,
) -> List[Dict]:
    """Transcribe audio with full speaker diarisation.

    Returns
    -------
    list[dict]
        ``{start, end, speaker, text}`` for every detected segment.
    """
    from faster_whisper import WhisperModel
    from pyannote.audio import Pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    # Whisper transcription
    model = WhisperModel(whisper_model, device=device, compute_type=compute_type)
    segments, info = model.transcribe(audio_path, task="transcribe")
    transcript = [
        {"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments
    ]

    # Pyannote diarisation
    pipeline = Pipeline.from_pretrained(pyannote_model)
    diarization = pipeline(audio_path)
    speakers = [
        {"start": t.start, "end": t.end, "speaker": spk}
        for t, _, spk in diarization.itertracks(yield_label=True)
    ]

    # Merge
    merged = _merge_transcript_speakers(transcript, speakers)

    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        print(f"📂 Diarised transcript saved → {output_json}")

    return merged


def _merge_transcript_speakers(transcript: list, speakers: list) -> list:
    """Align Whisper segments with Pyannote speaker labels by maximum overlap."""
    merged, j = [], 0
    for seg in transcript:
        start, end = seg["start"], seg["end"]
        best_speaker, max_overlap = "Unknown", 0

        while j < len(speakers) and speakers[j]["end"] <= start:
            j += 1

        k = j
        while k < len(speakers) and speakers[k]["start"] < end:
            overlap = min(end, speakers[k]["end"]) - max(start, speakers[k]["start"])
            if overlap > max_overlap:
                max_overlap, best_speaker = overlap, speakers[k]["speaker"]
            k += 1

        merged.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "speaker": best_speaker,
            "text": seg["text"],
        })

    return merged


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe classroom lecture")
    parser.add_argument("--input", required=True, help="Audio file path")
    parser.add_argument("--ref", help="Professor reference audio for isolation")
    parser.add_argument("--output", default="transcript.json", help="Output JSON")
    parser.add_argument("--model", default="medium", help="Whisper model size")
    args = parser.parse_args()

    if args.ref:
        transcribe_with_speaker_isolation(args.input, args.ref, args.model, output_json=args.output)
    else:
        transcribe_with_diarisation(args.input, args.model, output_json=args.output)
