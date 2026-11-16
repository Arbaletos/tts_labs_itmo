"""Copy-synthesis check: real wav -> mel -> HiFi-GAN -> wav. No acoustic model involved.

    python check_vocoder.py ../../data/RUSLAN/RUSLAN/000000_RUSLAN.wav
"""
import sys

import torch
from speechbrain.inference.vocoders import HIFIGAN

from TTS.config.shared_configs import BaseAudioConfig
from TTS.utils.audio import AudioProcessor

wav_path = sys.argv[1]

# These five differ from coqui's defaults and are what the vocoder was trained on.
audio_config = BaseAudioConfig(
    sample_rate=22050, fft_size=1024, win_length=1024, hop_length=256, num_mels=80,
    mel_fmin=0.0,
    mel_fmax=8000.0,          # coqui default: None
    spec_gain=1.0,            # coqui default: 20
    log_func="np.log",        # coqui default: np.log10
    signal_norm=False,        # coqui default: True
    do_trim_silence=False,    # coqui default: True
    preemphasis=0.0, ref_level_db=20, trim_db=60.0,
)

ap = AudioProcessor.init_from_config(audio_config)
wav = ap.load_wav(wav_path)
mel = ap.melspectrogram(wav)

# A natural RUSLAN mel spans about [-10.2, 0.5]. Far from that means the config is wrong.
print(f"mel {mel.shape}, range [{mel.min():.2f}, {mel.max():.2f}]")

hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-22050Hz",
                                savedir="pretrained_models/hifigan")
with torch.no_grad():
    vocoded = hifi_gan.decode_batch(torch.tensor(mel)).squeeze().cpu().numpy()

ap.save_wav(wav, "natural.wav", sr=22050)
ap.save_wav(vocoded, "copysynth_hifigan.wav", sr=22050)
ap.save_wav(ap.inv_melspectrogram(mel), "copysynth_griffinlim.wav", sr=22050)

# HiFi-GAN should sound almost like the original; Griffin-Lim is what lab 4 used.
print("wrote natural.wav, copysynth_hifigan.wav, copysynth_griffinlim.wav")
