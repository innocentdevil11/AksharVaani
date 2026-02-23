import os
import numpy as np
from scipy.io.wavfile import read, write

# ================== AUDIO PREPROCESSING ==================
from preprocessing.pre_emphasis import pre_emphasis
from preprocessing.highpass import highpass_filter
from preprocessing.noise_profile import get_last_noise_block
from preprocessing.nlms import nlms_fast
from preprocessing.dereverb import spectral_dereverb
from preprocessing.stationary_noise import stationary_noise_filter
from preprocessing.snr import compute_snr

# ================== OCR ==================
from ocr_modules.text_ocr import run_text_ocr
from ocr_modules.formula_ocr import run_formula_ocr

from img_crop.crop_2 import crop_blackboard

# ================== WHISPER ==================
from transcribe import transcribe_without_fingerprint



from llm_merge import merge_modalities


# ================== PATH SETUP ==================
AUDIO_INPUT = "chem_test_audio.wav"
CLEAN_AUDIO_PATH = "audio/AKSHARVAANI_final_clean.wav"
IMAGE_PATH = "chem_test_image.png"

os.makedirs("audio", exist_ok=True)


# # ================== LOAD AUDIO ==================
# fs, audio = read(AUDIO_INPUT)
# audio = audio.astype(np.float32)

# if audio.ndim > 1:
#     audio = audio.mean(axis=1)

# audio /= np.max(np.abs(audio)) + 1e-8


# # ================== AUDIO PIPELINE ==================
# print("\n🔊 Starting Audio Cleaning Pipeline...")

# pre = pre_emphasis(audio)
# hpf = highpass_filter(pre, cutoff=120, fs=fs)

# noise = get_last_noise_block(hpf, fs)
# r = int(np.ceil(len(hpf) / len(noise)))
# noise_long = np.tile(noise, r)[:len(hpf)]

# snr_before = compute_snr(hpf, noise_long)

# nlms_out = nlms_fast(hpf, fs, mu=0.02)
# dereverb = spectral_dereverb(nlms_out, alpha=0.25)

# final = stationary_noise_filter(
#     dereverb,
#     noise_long,
#     alpha=1.4,
#     beta=0.65,
#     smooth=0.92
# )

# snr_after = compute_snr(final, noise_long)

# write(CLEAN_AUDIO_PATH, fs, final)

# print("\n🎧 AUDIO CLEANING COMPLETE")
# print(f"📈 SNR Before : {snr_before:.2f} dB")
# print(f"📈 SNR After  : {snr_after:.2f} dB")
# print(f"🚀 Improvement: {snr_after - snr_before:.2f} dB")


# # ================== WHISPER TRANSCRIPTION ==================
# print("\n🎙️ STARTING TRANSCRIPTION")

# prof_segments = transcribe_without_fingerprint(
#     audio_path=CLEAN_AUDIO_PATH
# )

# print("\n🧠 TRANSCRIBED SEGMENTS:")
# for seg in prof_segments:
#     print(f"[{seg['start']:.2f}s → {seg['end']:.2f}s] {seg['text']}")

# print(f"\n✅ {len(prof_segments)} total segments")


# ================== OCR PIPELINE ==================
print("\n🧾 STARTING OCR WITH IMAGE CROPPING")

image_path = "ocr test.jpeg"

# Crop image into quadrants + rows
quadrant_imgs, row_imgs = crop_blackboard(image_path)

print(f"🖼️ Quadrants: {len(quadrant_imgs)} images")
print(f"🖼️ Rows: {len(row_imgs)} images")

# Run OCR on BOTH
text_out = run_text_ocr(quadrant_imgs + row_imgs)
formula_out = run_formula_ocr(quadrant_imgs + row_imgs)

print("\n📄 OCR COMPLETE")
print(f"📝 Text OCR → {text_out}")
print(f"📐 Formula OCR → {formula_out}")



# ================== LLM MERGE ==================
# print("\n🧠 MERGING AUDIO + OCR USING GROQ LLM")

# final_notes = merge_modalities(
#     transcript_segments=prof_segments,
#     text_ocr=text_out,
#     formula_ocr=formula_out
# )

# # Save output
# with open("final_lecture_notes.md", "w", encoding="utf-8") as f:
#     f.write(final_notes)

# print("\n📘 FINAL LECTURE NOTES GENERATED")
# print("📄 Saved as final_lecture_notes.md\n")
# print(final_notes)


print("\n✅ AKSHARVAANI PIPELINE COMPLETED SUCCESSFULLY")
