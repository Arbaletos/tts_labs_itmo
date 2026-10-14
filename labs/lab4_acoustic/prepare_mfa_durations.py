"""Build precomputed alignments from the MFA TextGrids — lab 4, ablation 2.

Writes one `.npy` per utterance plus a manifest, so training can take durations from the
forced alignment instead of learning them with the internal aligner::

    python prepare_mfa_durations.py --out ../../data/mfa_durations
    python train_fastspeech2.py --run-name mfa_dur --durations ../../data/mfa_durations/manifest.csv

**The format coqui expects**, traced through its source rather than guessed:

- each `.npy` is a one-hot matrix of shape `[T_mel, T_text]`; `dataset.py` loads it and the
  collate transposes it to `[T_text, T_mel]`, so the file is written mel-major
- the collate casts to float anyway, so **uint8 on disk is correct** and saves about 4x
- the manifest is `wav_path|npy_path`, one line per utterance, and goes to
  `datasets[0].meta_file_attn_mask`
- `base_tts.py` then converts the matrix to integer durations: argmax over the text axis
  for each frame, count the frames per token, force every token to at least one frame, and
  trim the longest durations until the total matches

That last step ends in a **hard assert**: `dur.sum() == mel_lengths[idx]`. If this script's
frame count disagrees with what the AudioProcessor produces for the same wav, training
crashes on the first batch containing that utterance. Which is why the mel length is taken
from the AudioProcessor itself here rather than computed from the sample count — and why
`do_trim_silence=False` is not optional.

Two things are checked per utterance and reported rather than silently skipped: that the
TextGrid's phone count matches the metadata's, and that every phone receives at least one
frame.
"""
import argparse
import os
import unicodedata
from pathlib import Path

import numpy as np
from praatio import textgrid
import tqdm

SIL = "<SIL>"
LENGTH_MARK = "ː"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata", type=Path, default=Path("../../data/metadata_phones.csv"),
                   help="id|text|phones from lab 3; defines which utterances and in what order")
    p.add_argument("--textgrids", type=Path, default=Path("../../data/RUSLAN_align_v2"))
    p.add_argument("--wavs", type=Path, default=Path("../../data/RUSLAN/RUSLAN"))
    p.add_argument("--out", type=Path, required=True, help="directory for the npy files and manifest")
    p.add_argument("--run", type=Path, default=None,
                   help="a run directory whose config.json supplies the audio settings; "
                        "by default the lab's canonical settings are used")
    return p.parse_args()


def read_metadata(path: Path) -> dict[str, list[str]]:
    """`id -> phone list`, NFC-normalized with the length mark collapsed."""
    out = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            cols = line.rstrip("\n").split("|")
            if len(cols) < 3:
                continue
            phones = [unicodedata.normalize("NFC", p).replace(LENGTH_MARK, "")
                      for p in cols[2].split()]
            out[cols[0]] = phones
    return out


def phone_intervals(tg_path: Path) -> list[tuple[float, float, str]]:
    """`(start, end, label)` of the phone tier, silence mapped to `<SIL>`."""
    tg = textgrid.openTextgrid(str(tg_path), True)
    if len(tg.tiers) < 2:
        msg = f"{tg_path.name}: expected 2 tiers, found {len(tg.tiers)}"
        raise ValueError(msg)
    out = []
    for entry in tg.tiers[1].entries:
        label = entry.label.strip()
        label = SIL if not label else unicodedata.normalize("NFC", label).replace(LENGTH_MARK, "")
        out.append((float(entry.start), float(entry.end), label))
    return out


def frame_owners(intervals: list[tuple[float, float, str]], n_frames: int,
                 hop_seconds: float) -> np.ndarray:
    """Which phone index owns each mel frame.

    A frame is assigned by its centre time, which is the same midpoint rule the lexicon
    used in lab 3 — boundaries written by two independent tiers can differ in the last
    decimal, and a midpoint never lands ambiguously.

    Frames past the last interval (the AudioProcessor can produce one or two more than the
    TextGrid covers, depending on padding) are given to the final phone.

    Args:
        intervals: Phone intervals in time order.
        n_frames: Mel frame count from the AudioProcessor. Authoritative.
        hop_seconds: `hop_length / sample_rate`.

    Returns:
        `int` array of length `n_frames` holding phone indices.
    """
    starts = np.array([s for s, _, _ in intervals])
    ends = np.array([e for _, e, _ in intervals])
    centres = (np.arange(n_frames) + 0.5) * hop_seconds

    owners = np.searchsorted(ends, centres, side="right")
    np.clip(owners, 0, len(intervals) - 1, out=owners)
    # a centre before the first interval's start still belongs to phone 0
    owners[centres < starts[0]] = 0
    return owners


def one_hot(owners: np.ndarray, n_text: int) -> np.ndarray:
    """`[T_mel, T_text]` uint8 one-hot, with every token guaranteed at least one frame.

    A zero-duration token would make coqui's duration conversion borrow a frame from the
    longest token, which is harmless but means the matrix no longer says what it looks
    like. Giving the token its own frame keeps the file honest; the nearest frame is taken
    from whichever neighbour has the most to spare.
    """
    matrix = np.zeros((len(owners), n_text), dtype=np.uint8)
    matrix[np.arange(len(owners)), owners] = 1

    counts = matrix.sum(axis=0)
    for token in np.flatnonzero(counts == 0):
        donor = int(np.argmax(counts))
        if counts[donor] < 2:
            break  # nothing to give; coqui's own fix-up will handle it
        frame = int(np.flatnonzero(matrix[:, donor])[-1])
        matrix[frame, donor] = 0
        matrix[frame, token] = 1
        counts[donor] -= 1
        counts[token] = 1
    return matrix


def main() -> None:
    """Write one alignment per utterance and the manifest that points at them."""
    from TTS.config import load_config
    from TTS.utils.audio import AudioProcessor

    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.run is not None:
        config = load_config(str(next(args.run.rglob("config.json"))))
        ap = AudioProcessor.init_from_config(config)
    else:
        from TTS.config.shared_configs import BaseAudioConfig
        ap = AudioProcessor.init_from_config(BaseAudioConfig(
            sample_rate=22050, fft_size=1024, win_length=1024, hop_length=256,
            num_mels=80, mel_fmin=0.0, mel_fmax=8000.0, spec_gain=1.0,
            log_func="np.log", signal_norm=False, preemphasis=0.0, ref_level_db=20,
            do_trim_silence=False, trim_db=60.0,
        ))
    hop_seconds = ap.hop_length / ap.sample_rate
    print(f"hop {ap.hop_length} at {ap.sample_rate} Hz = {hop_seconds * 1000:.2f} ms per frame")

    metadata = read_metadata(args.metadata)
    print(f"utterances in metadata: {len(metadata)}")

    manifest = []
    skipped_count = skipped_missing = tight = 0
    for utt_id, phones in tqdm.tqdm(sorted(metadata.items())):
        tg_path = args.textgrids / f"{utt_id}.TextGrid"
        wav_path = args.wavs / f"{utt_id}.wav"
        if not tg_path.is_file() or not wav_path.is_file():
            skipped_missing += 1
            continue

        intervals = phone_intervals(tg_path)
        if len(intervals) != len(phones):
            # the metadata and the alignment disagree; using it would misalign every
            # phone after the discrepancy, so skip rather than guess
            skipped_count += 1
            continue

        n_frames = ap.melspectrogram(ap.load_wav(str(wav_path))).shape[1]
        owners = frame_owners(intervals, n_frames, hop_seconds)
        matrix = one_hot(owners, len(phones))
        if (matrix.sum(axis=0) == 0).any():
            tight += 1

        npy_path = args.out / f"{utt_id}.npy"
        np.save(npy_path, matrix)
        manifest.append((str(wav_path), str(npy_path)))

    manifest_path = args.out / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as fh:
        for wav, npy in manifest:
            fh.write(f"{wav}|{npy}\n")

    print(f"\nwrote {len(manifest)} alignments to {args.out}")
    if skipped_missing:
        print(f"  skipped, no TextGrid or wav      : {skipped_missing}")
    if skipped_count:
        print(f"  skipped, phone count mismatch    : {skipped_count}")
        print("    metadata and alignment disagree -- regenerate the metadata from the")
        print("    same TextGrids, or these utterances will train on shifted durations")
    if tight:
        print(f"  utterances with a zero-frame phone: {tight}")
        print("    too many phones for the audio length; coqui will borrow frames itself")
    print(f"  manifest: {manifest_path}")
    print(f"\nNow: python train_fastspeech2.py --run-name mfa_dur --durations {manifest_path}")


if __name__ == "__main__":
    main()
