"""Synthesize the evaluation set and score it — lab 4.

Produces everything the report needs from one checkpoint::

    python evaluate.py --run runs/baseline

Outputs under `<run>/eval/`:

| directory / file | what it is |
|---|---|
| `natural/{id}.wav` | the real recording, resampled by the AudioProcessor |
| `copysynth/{id}.wav` | real mel -> Griffin-Lim. **The ceiling** — no model involved |
| `model/{id}.wav` | your mel -> Griffin-Lim |
| `mcd.csv` | per-utterance mel cepstral distortion |

Listen to the three in that order. The first gap is Griffin-Lim's own error, which no
acoustic model can beat; the second gap is your model's contribution. Attributing a
complaint to the wrong one of those is the most common mistake in this lab.

**Why the score is computed on mels rather than on audio.** Griffin-Lim reconstructs phase
badly, so MCD measured after vocoding is dominated by Griffin-Lim and says almost nothing
about the acoustic model — a *perfect* mel still scores poorly through it. Comparing mels
directly removes the vocoder from the measurement entirely.

Two reference numbers are printed beside your model's:

- **constant mel** — the mean mel frame of the training data, repeated. This is "predict
  nothing". A model that does not beat it has not learned.
- **copy-synthesis** — 0 dB by construction, since it is the natural mel compared with
  itself. Listed only as a reminder that the mel-domain score says nothing about the
  vocoder.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.fft import dct

N_MFCC = 13       # c1..c13; c0 is energy and is dropped
MCD_SCALE = (10.0 / np.log(10.0)) * np.sqrt(2.0)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True,
                   help="run directory holding config.json and best_model_*.pth")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="explicit checkpoint; default is the newest best_model_*.pth")
    p.add_argument("--eval-set", type=Path, default=Path("data/eval_sentences.csv"),
                   help="id|text|phones, the shipped evaluation sentences")
    p.add_argument("--wavs", type=Path, default=Path("../../data/RUSLAN/RUSLAN"))
    p.add_argument("--metadata", type=Path, default=Path("../../data/metadata_phones.csv"),
                   help="training metadata, used only for the constant-mel reference")
    p.add_argument("--limit", type=int, default=None,
                   help="synthesize only the first N sentences, for a quick check")
    return p.parse_args()


def mel_to_mfcc(mel: np.ndarray) -> np.ndarray:
    """Log-mel `[n_mels, T]` -> MFCC `[N_MFCC, T]`, dropping c0.

    A DCT along the mel axis is all an MFCC is. The mel must already be in the log domain,
    which it is here: `spec_gain=1` and `log_func="np.log"` mean the AudioProcessor and the
    model both produce natural-log mels.
    """
    coeffs = dct(mel, axis=0, type=2, norm="ortho")
    return coeffs[1:1 + N_MFCC]


def dtw_align(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Monotonic alignment between two MFCC sequences `[C, Ta]` and `[C, Tb]`.

    Needed because predicted durations differ from the real ones, so frame *i* of the
    synthesis is not frame *i* of the recording. Without this the score would mostly
    measure duration error.

    Returns:
        Index arrays `(ia, ib)` of equal length giving the matched frame pairs.
    """
    ta, tb = a.shape[1], b.shape[1]
    # squared euclidean distance between every pair of frames, via the dot-product identity
    cost = (np.sum(a ** 2, axis=0)[:, None]
            + np.sum(b ** 2, axis=0)[None, :]
            - 2.0 * (a.T @ b))
    np.maximum(cost, 0.0, out=cost)
    cost = np.sqrt(cost)

    acc = np.full((ta + 1, tb + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, ta + 1):
        c = cost[i - 1]
        prev, cur = acc[i - 1], acc[i]
        for j in range(1, tb + 1):
            cur[j] = c[j - 1] + min(prev[j], cur[j - 1], prev[j - 1])

    i, j = ta, tb
    ia, ib = [], []
    while i > 0 and j > 0:
        ia.append(i - 1)
        ib.append(j - 1)
        step = int(np.argmin([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]]))
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    return np.array(ia[::-1]), np.array(ib[::-1])


def mcd(reference_mel: np.ndarray, other_mel: np.ndarray) -> float:
    """Mel cepstral distortion in dB between two log-mels, DTW-aligned.

    Lower is better. Roughly: ~1-2 dB is vocoder-only error, 4-6 dB is a decent acoustic
    model, above 8 dB is poor. Treat it as a relative number — its agreement with what
    people actually hear is weak, which is why the listening ladder is the primary
    instrument and this is a supporting one.
    """
    ref = mel_to_mfcc(reference_mel)
    hyp = mel_to_mfcc(other_mel)
    ia, ib = dtw_align(ref, hyp)
    diff = ref[:, ia] - hyp[:, ib]
    return float(MCD_SCALE * np.mean(np.sqrt(np.sum(diff ** 2, axis=0))))


def mean_training_mel(ap, metadata: Path, wav_dir: Path, n: int = 200) -> np.ndarray:
    """The average mel frame over a sample of training audio: the 'predict nothing' model."""
    frames = []
    with metadata.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if len(frames) >= n:
                break
            cols = line.rstrip("\n").split("|")
            if len(cols) < 3:
                continue
            wav_path = wav_dir / f"{cols[0]}.wav"
            if not wav_path.is_file():
                continue
            frames.append(ap.melspectrogram(ap.load_wav(str(wav_path))).mean(axis=1))
    if not frames:
        msg = f"No training audio found via {metadata}"
        raise RuntimeError(msg)
    return np.mean(frames, axis=0)


def read_eval_set(path: Path, limit: int | None) -> list[tuple[str, str, str]]:
    """Read `id|text|phones` into tuples."""
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            cols = line.rstrip("\n").split("|")
            if len(cols) >= 3:
                rows.append((cols[0], cols[1], cols[2]))
    return rows[:limit] if limit else rows


def main() -> None:
    """Synthesize, vocode three ways, and score against the natural mel."""
    import torch
    from TTS.config import load_config
    from TTS.tts.models.forward_tts import ForwardTTS
    from TTS.tts.utils.text.tokenizer import TTSTokenizer
    from TTS.utils.audio import AudioProcessor
    from TTS.tts.configs.shared_configs import CharactersConfig

    args = parse_args()
    checkpoint = args.checkpoint
    if checkpoint is None:
        candidates = sorted(args.run.rglob("best_model*.pth"))
        if not candidates:
            msg = f"No best_model*.pth under {args.run}"
            raise SystemExit(msg)
        checkpoint = candidates[-1]
    config_path = next(args.run.rglob("config.json"))
    print(f"config    : {config_path}")
    print(f"checkpoint: {checkpoint}")

    config = load_config(str(config_path))

    character_cfg = json.loads(open(str(config_path)).read())['characters']
    character_cfg = CharactersConfig(**character_cfg)
    config.characters = character_cfg

    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)
    model = ForwardTTS(config, ap, tokenizer, speaker_manager=None)
    model.load_state_dict(torch.load(str(checkpoint), map_location="cpu")["model"])
    model.eval()

    out = args.run / "eval"
    for sub in ("natural", "copysynth", "model"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    rows = read_eval_set(args.eval_set, args.limit)
    print(f"evaluation sentences: {len(rows)}\n")

    constant_frame = mean_training_mel(ap, args.metadata, args.wavs)
    records = []
    ranges = []

    for utt_id, text, phones in rows:
        wav_path = args.wavs / f"{utt_id}.wav"
        if not wav_path.is_file():
            print(f"  skipping {utt_id}: no wav")
            continue

        natural_wav = ap.load_wav(str(wav_path))
        natural_mel = ap.melspectrogram(natural_wav)

        result = model.synthesize(phones, config, use_griffin_lim=True)
        model_mel = result["outputs"]["model_outputs"][0].detach().cpu().numpy().T

        ap.save_wav(natural_wav, str(out / "natural" / f"{utt_id}.wav"))
        ap.save_wav(ap.inv_melspectrogram(natural_mel), str(out / "copysynth" / f"{utt_id}.wav"))
        ap.save_wav(result["wav"], str(out / "model" / f"{utt_id}.wav"))

        constant_mel = np.repeat(constant_frame[:, None], natural_mel.shape[1], axis=1)
        records.append({
            "id": utt_id,
            "frames_natural": natural_mel.shape[1],
            "frames_model": model_mel.shape[1],
            "mcd_model": round(mcd(natural_mel, model_mel), 4),
            "mcd_constant": round(mcd(natural_mel, constant_mel), 4),
        })
        ranges.append((natural_mel.min(), natural_mel.max(),
                       model_mel.min(), model_mel.max()))
        print(f"  {utt_id}  frames {natural_mel.shape[1]:4d} -> {model_mel.shape[1]:4d}   "
              f"MCD {records[-1]['mcd_model']:6.2f} dB")

    with (out / "mcd.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, list(records[0]), delimiter="|", quoting=csv.QUOTE_NONE)
        writer.writeheader()
        writer.writerows(records)

    model_mcd = np.array([r["mcd_model"] for r in records])
    const_mcd = np.array([r["mcd_constant"] for r in records])
    frames_n = np.array([r["frames_natural"] for r in records])
    frames_m = np.array([r["frames_model"] for r in records])
    nat_lo, nat_hi, mod_lo, mod_hi = (np.array(x) for x in zip(*ranges))

    print(f"\n{'=' * 64}")
    print(f"MCD, your model   : {model_mcd.mean():6.2f} dB  (median {np.median(model_mcd):.2f})")
    print(f"MCD, constant mel : {const_mcd.mean():6.2f} dB")
    print(f"MCD, copy-synth   :   0.00 dB  (the natural mel against itself, by definition)")
    print(f"\nduration: model/natural frame ratio {np.mean(frames_m / frames_n):.3f} "
          f"(1.000 would be perfect on average)")
    print(f"\nmel range  natural: [{nat_lo.mean():7.2f}, {nat_hi.mean():6.2f}]")
    print(f"mel range  yours  : [{mod_lo.mean():7.2f}, {mod_hi.mean():6.2f}]")
    print(f"{'=' * 64}")
    print(f"\nwrote {out / 'mcd.csv'} and {len(records)} x 3 wav files under {out}")


if __name__ == "__main__":
    main()
