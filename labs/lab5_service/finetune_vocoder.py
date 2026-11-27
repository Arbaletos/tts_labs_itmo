"""Fine-tune the SpeechBrain HiFi-GAN (LibriTTS 22 kHz) on RUSLAN - lab 5, extra task 2.

    python finetune_vocoder.py --name gt --steps 30000                # on natural mels
    python finetune_vocoder.py --name gta --mels ../../data/lab5_gta  # on lab 4 model mels
    python finetune_vocoder.py --name gt --resume                     # continue a run

The published checkpoint holds the **generator only**, so the discriminators (MPD + MSD, as in
the paper) start from scratch. A random discriminator gives the generator a meaningless
adversarial signal, so for the first `--d-warmup` steps only the discriminators train; then
both train with the paper's losses::

    L_D = LSGAN(real -> 1, fake -> 0)            over MPD + MSD
    L_G = LSGAN(fake -> 1) + 2 * L_FM + 45 * L1(mel(y_hat), mel(y))

with a smaller generator learning rate than the paper's 2e-4, since we adapt rather than
train. Batches are random 8192-sample (32-frame) segments of the training utterances (number
not divisible by 5 - the lab 4 split, so the lab 4 evaluation sentences stay unseen).

Two input variants:

* natural mels (default), computed on the GPU from the audio segment with exactly the
  vocoder's mel (checked against coqui's AudioProcessor at start-up) - adaptation to the
  speaker and the language;
* `--mels DIR` with `{id}.npy` mels of the same frame count as the recording - e.g. lab 4
  model outputs with teacher-forced durations (GTA). The vocoder then learns to produce the
  *real* audio from the *model's* smoothed mel, which is the actual mismatch at synthesis.

Every `--save-every` steps a SpeechBrain-loadable directory `step_<N>/` is written
(`hyperparams.yaml` + `generator.ckpt`), usable as `TTSPipeline(vocoder_source=...)`. Create
a file `STOP` in the run directory to stop cleanly after the next save.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import time
from pathlib import Path

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / 'data' / 'hf_cache'))
WAVS = ROOT / 'data' / 'RUSLAN' / 'RUSLAN'
METADATA = ROOT / 'data' / 'metadata_phones_gt_train.csv'
EVAL_SET = ROOT / 'labs' / 'lab4_acoustic' / 'data' / 'eval_sentences.csv'
SOURCE = 'speechbrain/tts-hifigan-libritts-22050Hz'
SR, HOP, N_FFT, N_MELS = 22050, 256, 1024, 80


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--name', required=True)
    p.add_argument('--mels', type=Path, default=None, help='directory of {id}.npy input mels (GTA)')
    p.add_argument('--init', default=SOURCE, help='generator to start from (hub id or directory)')
    p.add_argument('--steps', type=int, default=30000)
    p.add_argument('--d-warmup', type=int, default=2000, help='steps with the generator frozen')
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--segment', type=int, default=8192)
    p.add_argument('--lr-g', type=float, default=5e-5)
    p.add_argument('--lr-d', type=float, default=2e-4)
    p.add_argument('--save-every', type=int, default=2500)
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--init-d', type=Path, default=None,
                   help='train_state.pt to take already trained discriminators from (skips the warm-up need)')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


# ----------------------------------------------------------------------------------------
# mel exactly as the vocoder (and the lab 4 model) sees it
# ----------------------------------------------------------------------------------------
class Mel(torch.nn.Module):
    """|STFT| (center, reflect) -> slaney mel filterbank -> ln(max(x, 1e-5)).

    Matches coqui's AudioProcessor with the vocoder's config (`check_vocoder.py`);
    `fmax=None` gives the full-band mel used for the loss, as in the HiFi-GAN recipe.
    """

    def __init__(self, fmax: float | None = 8000.0):
        super().__init__()
        import librosa
        fb = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=0.0, fmax=fmax)
        self.register_buffer('fb', torch.tensor(fb, dtype=torch.float32))
        self.register_buffer('window', torch.hann_window(N_FFT))

    def forward(self, wav: torch.Tensor) -> torch.Tensor:              # [B, T] -> [B, 80, frames]
        spec = torch.stft(wav, N_FFT, HOP, N_FFT, self.window, center=True, pad_mode='reflect',
                          return_complex=True).abs()
        return torch.log(torch.clamp(self.fb @ spec, min=1e-5))


def check_mel_matches_coqui(mel: Mel) -> float:
    import sys
    sys.path.insert(0, str(HERE))
    from lab5_eval import audio_processor
    ap = audio_processor()
    wav = ap.load_wav(str(WAVS / '000001_RUSLAN.wav'))
    ref = ap.melspectrogram(wav)
    ours = mel(torch.tensor(wav, dtype=torch.float32)[None]).squeeze(0).numpy()
    return float(np.abs(ref - ours).max())


# ----------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------
def read_ids(path: Path) -> list[str]:
    with path.open(encoding='utf-8') as fh:
        return [line.split('|', 1)[0] for line in fh if line.strip()]


class Segments(Dataset):
    """Random segments of the training utterances; with `mel_dir`, the matching mel frames."""

    def __init__(self, ids: list[str], segment: int, mel_dir: Path | None, epoch_size: int):
        self.segment, self.mel_dir, self.epoch_size = segment, mel_dir, epoch_size
        self.frames = segment // HOP
        self.items = []
        for i in ids:
            if mel_dir is not None and not (mel_dir / f'{i}.npy').exists():
                continue
            n = sf.info(str(WAVS / f'{i}.wav')).frames
            if n > segment + HOP:
                self.items.append((i, n))

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, _):
        i, n = random.choice(self.items)
        if self.mel_dir is None:
            start = random.randrange(0, n - self.segment)
            wav, _ = sf.read(str(WAVS / f'{i}.wav'), start=start, frames=self.segment, dtype='float32')
            return torch.from_numpy(wav), torch.zeros(0)
        mel = np.load(self.mel_dir / f'{i}.npy', mmap_mode='r')                # [80, frames]
        f0 = random.randrange(0, min(mel.shape[1], n // HOP) - self.frames)
        wav, _ = sf.read(str(WAVS / f'{i}.wav'), start=f0 * HOP, frames=self.segment, dtype='float32')
        if len(wav) < self.segment:
            wav = np.pad(wav, (0, self.segment - len(wav)))
        return torch.from_numpy(wav), torch.from_numpy(np.array(mel[:, f0:f0 + self.frames], dtype=np.float32))


# ----------------------------------------------------------------------------------------
# losses
# ----------------------------------------------------------------------------------------
def d_loss(real_scores, fake_scores) -> torch.Tensor:
    return sum(torch.mean((1 - r) ** 2) + torch.mean(f ** 2) for r, f in zip(real_scores, fake_scores))


def g_adv_loss(fake_scores) -> torch.Tensor:
    return sum(torch.mean((1 - f) ** 2) for f in fake_scores)


def feature_loss(real_feats, fake_feats) -> torch.Tensor:
    return sum(F.l1_loss(f, r.detach()) for rs, fs in zip(real_feats, fake_feats) for r, f in zip(rs, fs))




# ----------------------------------------------------------------------------------------
def save_generator(generator, run_dir: Path, step: int, hparams_src: Path) -> Path:
    out = run_dir / f'step_{step}'
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(hparams_src, out / 'hyperparams.yaml')
    torch.save(generator.state_dict(), out / 'generator.ckpt')
    return out


def main() -> None:
    from speechbrain.inference.vocoders import HIFIGAN
    from speechbrain.lobes.models.HifiGAN import HifiganDiscriminator
    from speechbrain.utils.fetching import LocalStrategy

    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = 'cuda'
    run_dir = HERE / 'runs' / 'vocoder_ft' / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    vocoder = HIFIGAN.from_hparams(source=args.init,
                                   savedir=str(HERE / 'pretrained_models' / Path(args.init).name),
                                   run_opts={'device': 'cuda:0'}, local_strategy=LocalStrategy.COPY)
    hparams_src = HERE / 'pretrained_models' / Path(args.init).name / 'hyperparams.yaml'
    generator = vocoder.hparams.generator.to(device).train()    # still with weight norm: decode_batch not called
    generator.requires_grad_(True)            # Pretrained() freezes all parameters (freeze_params=True)
    discriminator = HifiganDiscriminator().to(device).train()
    mel_in, mel_loss = Mel(8000.0).to(device), Mel(None).to(device)
    print(f'mel vs coqui AudioProcessor: max |diff| = {check_mel_matches_coqui(Mel(8000.0)):.2e}')

    opt_g = torch.optim.AdamW(generator.parameters(), args.lr_g, betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(discriminator.parameters(), args.lr_d, betas=(0.8, 0.99))
    if args.init_d is not None:
        discriminator.load_state_dict(torch.load(args.init_d, map_location=device)['discriminator'])
        print(f'discriminators from {args.init_d}')
    step = 0
    state_path = run_dir / 'train_state.pt'
    if args.resume and state_path.exists():
        state = torch.load(state_path, map_location=device)
        generator.load_state_dict(state['generator'])
        discriminator.load_state_dict(state['discriminator'])
        opt_g.load_state_dict(state['opt_g'])
        opt_d.load_state_dict(state['opt_d'])
        step = state['step']
        print(f'resumed at step {step}')

    eval_ids = set(read_ids(EVAL_SET))
    ids = [i for i in read_ids(METADATA) if i not in eval_ids]
    data = Segments(ids, args.segment, args.mels, epoch_size=args.batch * 1000)
    print(f'{len(data.items)} training utterances, run dir {run_dir}')
    loader = DataLoader(data, args.batch, num_workers=args.workers, persistent_workers=args.workers > 0,
                        drop_last=True, pin_memory=True)

    log_path = run_dir / 'log.csv'
    new_log = not log_path.exists()
    log = log_path.open('a', newline='', encoding='utf-8')
    writer = csv.writer(log)
    if new_log:
        writer.writerow(['step', 'd_loss', 'g_adv', 'g_fm', 'g_mel', 'sec_per_step'])
    acc, t0 = np.zeros(4), time.time()

    while step < args.steps:
        for wav, mel in loader:
            wav = wav.to(device, non_blocking=True)
            with torch.no_grad():
                if args.mels is None:
                    mel = mel_in(wav)[:, :, :args.segment // HOP]
                else:
                    mel = mel.to(device, non_blocking=True)
            y = wav[:, None, :]
            y_hat = generator(mel)[:, :, :args.segment]

            # discriminator
            opt_d.zero_grad(set_to_none=True)
            real_scores, _ = discriminator(y)
            fake_scores, _ = discriminator(y_hat.detach())
            loss_d = d_loss(real_scores, fake_scores)
            loss_d.backward()
            opt_d.step()

            # generator (frozen during the discriminator warm-up)
            mel_l1 = F.l1_loss(mel_loss(y_hat.squeeze(1)), mel_loss(wav))
            if step >= args.d_warmup:
                opt_g.zero_grad(set_to_none=True)
                fake_scores, fake_feats = discriminator(y_hat)
                with torch.no_grad():
                    _, real_feats = discriminator(y)
                adv, fm = g_adv_loss(fake_scores), feature_loss(real_feats, fake_feats)
                loss_g = adv + 2.0 * fm + 45.0 * mel_l1
                loss_g.backward()
                if step == args.d_warmup:             # first generator update: make sure it really learns
                    probe = generator.conv_post.conv.bias.detach().clone()
                opt_g.step()
                if step == args.d_warmup:
                    assert not torch.equal(probe, generator.conv_post.conv.bias), 'generator is not updated'
                acc += [loss_d.item(), adv.item(), fm.item(), mel_l1.item()]
            else:
                acc += [loss_d.item(), np.nan, np.nan, mel_l1.item()]
            step += 1

            if step % args.log_every == 0:
                sec = (time.time() - t0) / args.log_every
                row = [step, *np.round(acc / args.log_every, 4), round(sec, 3)]
                writer.writerow(row)
                log.flush()
                print(f'step {step:6d}  D {row[1]:.3f}  G adv {row[2]:.3f} fm {row[3]:.3f} '
                      f'mel {row[4]:.4f}  {sec:.2f} s/step', flush=True)
                acc, t0 = np.zeros(4), time.time()
            stop = (run_dir / 'STOP').exists()
            if step % args.save_every == 0 or step >= args.steps or stop:
                save_generator(generator, run_dir, step, hparams_src)
                torch.save({'generator': generator.state_dict(), 'discriminator': discriminator.state_dict(),
                            'opt_g': opt_g.state_dict(), 'opt_d': opt_d.state_dict(), 'step': step},
                           state_path)
                print(f'saved step {step}', flush=True)
            if step >= args.steps or stop:
                break
        if (run_dir / 'STOP').exists():
            break
    log.close()


if __name__ == '__main__':
    main()
