import torch
import numpy as np
from speechbrain.pretrained import SpectralMaskEnhancement

denoiser = SpectralMaskEnhancement.from_hparams(
    source="speechbrain/metricgan-plus-voicebank",
    savedir="sb_denoiser"
)

def apply_speechbrain_denoiser(audio):
    if isinstance(audio, np.ndarray):
        audio = torch.tensor(audio, dtype=torch.float32)

    if audio.ndim == 1:
        audio = audio.unsqueeze(0)

    lengths = torch.tensor([1.0])
    enhanced = denoiser.enhance_batch(audio, lengths)

    enhanced = np.array(
        enhanced.squeeze().detach().cpu().tolist(),
        dtype=np.float32
    )

    return enhanced / (np.max(np.abs(enhanced)) + 1e-8)
