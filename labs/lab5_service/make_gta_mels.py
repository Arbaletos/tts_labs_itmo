"""Ground-truth-aligned (GTA) mels of the lab 4 model for vocoder fine-tuning — lab 5, extra 2.

    python make_gta_mels.py --out ../../data/lab5_gta

For every training utterance the FastSpeech 2 model runs in its *training* forward pass:
the aligner fixes the durations from the real mel, and the pitch and energy predictors are
fed the real (cached, lab 4) F0 and energy. The decoder output then has exactly the frames of
the recording, so `(model mel, real audio)` is a training pair for the vocoder — and the mel
carries the model's own smoothing, which is what the vocoder meets at synthesis time.

The real F0 and energy make these mels somewhat better than at inference (where both are
predicted); the durations and the decoder's smoothing are the model's own.
"""
from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', type=Path, default=ROOT / 'labs' / 'lab4_acoustic' / 'runs' / 'pitch_energy')
    p.add_argument('--metadata', type=Path, default=ROOT / 'data' / 'metadata_phones_gt_train.csv')
    p.add_argument('--cache', type=Path, default=ROOT / 'data' / 'lab4_cache')
    p.add_argument('--out', type=Path, default=ROOT / 'data' / 'lab5_gta')
    p.add_argument('--limit', type=int, default=None)
    return p.parse_args()


def cache_name(utt_id: str) -> str:
    """coqui's `string2filename(audio_unique_name)`: urlsafe base64 of `ruslan#<id>`."""
    return base64.urlsafe_b64encode(f'ruslan#{utt_id}'.encode()).decode()


def main() -> None:
    from tts_pipeline import TTSPipeline
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pipe = TTSPipeline(args.run)            # loads the model, its tokenizer and AudioProcessor
    model, ap, device = pipe.model, pipe.ap, pipe.device

    stats = {k: np.load(args.cache / k / f'{"pitch" if k == "f0" else k}_stats.npy', allow_pickle=True).item()
             for k in ('f0', 'energy')}

    def normalized(kind: str, utt_id: str) -> np.ndarray:
        suffix = 'pitch' if kind == 'f0' else 'energy'
        x = np.load(args.cache / kind / f'{cache_name(utt_id)}_{suffix}.npy').astype(np.float32)
        zero = x == 0.0
        x = (x - stats[kind]['mean']) / stats[kind]['std']
        x[zero] = 0.0
        return x

    rows = [line.rstrip('\n').split('|') for line in args.metadata.open(encoding='utf-8') if line.strip()]
    rows = rows[:args.limit]
    done = skipped = 0
    diffs = []
    for utt_id, _, phones in rows:
        target = args.out / f'{utt_id}.npy'
        if target.exists():
            continue
        wav = ap.load_wav(str(ROOT / 'data' / 'RUSLAN' / 'RUSLAN' / f'{utt_id}.wav'))
        mel = ap.melspectrogram(wav).astype(np.float32)                       # [80, T]
        try:
            pitch, energy = normalized('f0', utt_id), normalized('energy', utt_id)
        except FileNotFoundError:
            skipped += 1
            continue
        t = min(mel.shape[1], len(pitch), len(energy))
        mel, pitch, energy = mel[:, :t], pitch[:t], energy[:t]
        ids = model.tokenizer.text_to_ids(phones)
        x = torch.tensor([ids], device=device)
        with torch.no_grad():
            out = model.forward(x, torch.tensor([len(ids)], device=device), torch.tensor([t], device=device),
                                y=torch.tensor(mel.T[None], device=device),
                                pitch=torch.tensor(pitch[None, None], device=device),
                                energy=torch.tensor(energy[None, None], device=device))
        gta = out['model_outputs'][0].cpu().numpy().T                          # [80, T]
        np.save(target, gta.astype(np.float16))
        diffs.append(float(np.abs(gta - mel).mean()))
        done += 1
        if done % 1000 == 0:
            print(f'{done} done, mean |GTA - natural| = {np.mean(diffs):.3f}', flush=True)
    print(f'{done} written, {skipped} skipped (no F0 cache); mean |GTA - natural| = '
          f'{np.mean(diffs) if diffs else float("nan"):.3f}')


if __name__ == '__main__':
    main()
