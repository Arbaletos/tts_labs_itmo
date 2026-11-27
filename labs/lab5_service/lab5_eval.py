"""Measurements for the lab 5 report: vocoder quality, intelligibility, speed.

Everything heavy writes its result under `runs/` (git-ignored) and is skipped when the
result is already there, so the report notebook can be re-executed cheaply::

    python lab5_eval.py copysynth      # natural mel -> HiFi-GAN / Griffin-Lim, 120 sentences
    python lab5_eval.py model_mels     # lab 4 model mels -> HiFi-GAN (vs Griffin-Lim of lab 4)
    python lab5_eval.py phrases        # the 10 + 5 test phrases through the whole pipeline
    python lab5_eval.py rtf            # RTF against input length, GPU and CPU
    python lab5_eval.py asr            # Whisper on everything synthesized so far

Vocoder quality is judged three ways, because no single number covers it:

* **mel distance** — re-analyse the vocoded audio with the same AudioProcessor and compare
  with the mel that went in (L1 in ln units, and MCD). It says how much of its input the
  vocoder fails to reproduce; it cannot be computed against the recording for model mels,
  whose timing differs;
* **F0 (pYIN)** — voiced share and agreement of the pitch contour with the recording:
  whether the vocoder produces a *tone*;
* **ASR** (Whisper large-v3-turbo, as in lab 4) — intelligibility, CER/WER.
"""
from __future__ import annotations

import csv
import importlib.util
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LAB4 = ROOT / 'labs' / 'lab4_acoustic'
os.environ.setdefault('HF_HOME', str(ROOT / 'data' / 'hf_cache'))   # ASCII path, inside data/
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '4')

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(HERE))
from tts_pipeline import trim_vocoder_padding, vocoder_savedir  # noqa: E402

RUNS = HERE / 'runs'
AUDIO = HERE / 'audio'
WAVS = ROOT / 'data' / 'RUSLAN' / 'RUSLAN'
EVAL_SET = LAB4 / 'data' / 'eval_sentences.csv'
SR = 22050
ASR_MODEL = 'openai/whisper-large-v3-turbo'
ASR_CSV = RUNS / 'asr.csv'
ASR_FIELDS = ['set', 'variant', 'id', 'ref', 'hyp', 'cer', 'wer']

# lab 4 evaluation directories whose saved mels (`mels/{id}_model.npy`) are vocoded again
MODEL_MELS = {
    'gt': LAB4 / 'runs' / 'gt' / 'eval',
    'mfa_dur': LAB4 / 'runs' / 'mfa_dur' / 'eval_30k',
    'pitch_energy': LAB4 / 'runs' / 'pitch_energy' / 'eval_30k',
}
# ... and the lab 4 ASR labels of the same audio through Griffin-Lim
LAB4_ASR_LABEL = {'gt': 'gt', 'mfa_dur': 'mfa_dur@30k', 'pitch_energy': 'pitch_energy@30k'}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lab4_eval = _load('lab4_evaluate', LAB4 / 'evaluate.py')      # mel_to_mfcc, dtw_align, mcd
# F0 runs in worker processes (spawn on Windows): they must be able to import the module by
# name, so this one goes through sys.path rather than a file spec.
sys.path.insert(0, str(LAB4))
import analysis as lab4_analysis  # noqa: E402  (f0, harmonic ripple)


# ----------------------------------------------------------------------------------------
# audio front: the vocoder's audio configuration (same as check_vocoder.py)
# ----------------------------------------------------------------------------------------
def audio_processor():
    from TTS.config.shared_configs import BaseAudioConfig
    from TTS.utils.audio import AudioProcessor
    cfg = BaseAudioConfig(sample_rate=SR, fft_size=1024, win_length=1024, hop_length=256,
                          num_mels=80, mel_fmin=0.0, mel_fmax=8000.0, spec_gain=1.0,
                          log_func='np.log', signal_norm=False, do_trim_silence=False,
                          preemphasis=0.0, ref_level_db=20, trim_db=60.0)
    return AudioProcessor.init_from_config(cfg)


def load_hifigan(source: str = 'speechbrain/tts-hifigan-libritts-22050Hz', device: str = 'cuda'):
    """SpeechBrain HiFi-GAN from the hub id or a local directory with hyperparams.yaml."""
    import torch
    from speechbrain.inference.vocoders import HIFIGAN
    from speechbrain.utils.fetching import LocalStrategy
    device = device if torch.cuda.is_available() else 'cpu'
    return HIFIGAN.from_hparams(source=source,
                                savedir=str(vocoder_savedir(source)),
                                run_opts={'device': 'cuda:0' if device == 'cuda' else device},
                                local_strategy=LocalStrategy.COPY)


def hifigan_fn(vocoder):
    import torch

    def run(mel: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            wav = vocoder.decode_batch(torch.tensor(mel, dtype=torch.float32)).squeeze().cpu().numpy()
        return trim_vocoder_padding(vocoder, wav)
    return run


def read_eval_set(path: Path = EVAL_SET) -> dict[str, str]:
    with path.open(encoding='utf-8') as fh:
        return dict(line.rstrip('\n').split('|')[:2] for line in fh if line.strip())


def write_wav(path: Path, wav: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.abs(wav).max()) if len(wav) else 0.0
    if peak > 1.0:                                   # Griffin-Lim can overshoot
        wav = wav / peak * 0.99
    sf.write(str(path), wav.astype(np.float32), SR, subtype='PCM_16')


# ----------------------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------------------
def mel_distance(ap, mel_in: np.ndarray, wav: np.ndarray) -> dict:
    """Re-analyse `wav` and compare with the mel it was made from (frame-aligned)."""
    mel_out = ap.melspectrogram(wav.astype(np.float32))
    n = min(mel_in.shape[1], mel_out.shape[1])
    a, b = mel_in[:, :n], mel_out[:, :n]
    mfa, mfb = lab4_eval.mel_to_mfcc(a), lab4_eval.mel_to_mfcc(b)
    mcd = lab4_eval.MCD_SCALE * np.mean(np.sqrt(np.sum((mfa - mfb) ** 2, axis=0)))
    return {'mel_l1': float(np.abs(a - b).mean()), 'mcd_frames': float(mcd),
            'ripple_out': lab4_analysis.harmonic_ripple(mel_out)}


def f0_table(paths: dict[str, Path], natural: dict[str, Path], mels: dict | None = None,
             workers: int = 6) -> pd.DataFrame:
    """pYIN on every wav; F0 agreement with the recording along a DTW path on the mels
    (identity path when `mels` is None, i.e. the audio has the recording's timing)."""
    keys = list(paths)
    f0 = lab4_analysis.f0_contours({('hyp', k): paths[k] for k in keys}
                                   | {('ref', k): natural[k] for k in keys}, workers=workers)
    rows = []
    for k in keys:
        a, b = f0[('ref', k)], f0[('hyp', k)]
        if mels is None:
            n = min(len(a), len(b))
            ia = ib = np.arange(n)
        else:
            ia, ib = lab4_eval.dtw_align(lab4_eval.mel_to_mfcc(mels[k][0]),
                                         lab4_eval.mel_to_mfcc(mels[k][1]))
        rows.append({'id': k, **lab4_analysis.f0_summary(b), **lab4_analysis.f0_agreement(a, b, ia, ib)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
# 1. copy-synthesis
# ----------------------------------------------------------------------------------------
def copysynth(out: Path = RUNS / 'copysynth', limit: int | None = None) -> pd.DataFrame:
    """Natural mel -> {HiFi-GAN, Griffin-Lim} for the 120 lab 4 evaluation sentences."""
    table = out / 'metrics.csv'
    if table.exists():
        return pd.read_csv(table)
    ap = audio_processor()
    vocoder = hifigan_fn(load_hifigan())
    ids = list(read_eval_set())[:limit]
    rows = []
    for i in ids:
        wav = ap.load_wav(str(WAVS / f'{i}.wav'))
        mel = ap.melspectrogram(wav)
        write_wav(out / 'natural' / f'{i}.wav', wav)
        for name, fn in (('hifigan', vocoder), ('griffinlim', ap.inv_melspectrogram)):
            start = time.perf_counter()
            y = fn(mel)
            seconds = time.perf_counter() - start
            write_wav(out / name / f'{i}.wav', y)
            rows.append({'id': i, 'vocoder': name, 'seconds': seconds,
                         'audio_s': len(y) / SR, **mel_distance(ap, mel, y)})
        rows.append({'id': i, 'vocoder': 'natural', 'audio_s': len(wav) / SR,
                     **mel_distance(ap, mel, wav)})
    df = pd.DataFrame(rows)
    nat = {i: out / 'natural' / f'{i}.wav' for i in ids}
    f0 = pd.concat([f0_table({i: out / v / f'{i}.wav' for i in ids}, nat).assign(vocoder=v)
                    for v in ('hifigan', 'griffinlim')])
    df = df.merge(f0, on=['id', 'vocoder'], how='left')
    df.to_csv(table, index=False)
    return df


def copysynth_vocoder(source: str, tag: str, out: Path = RUNS / 'copysynth') -> pd.DataFrame:
    """Copy-synthesis of the same 120 sentences with another (fine-tuned) HiFi-GAN."""
    table = out / f'metrics_{tag}.csv'
    if table.exists():
        return pd.read_csv(table)
    ap = audio_processor()
    vocoder = hifigan_fn(load_hifigan(source))
    ids = list(read_eval_set())
    rows = []
    for i in ids:
        mel = ap.melspectrogram(ap.load_wav(str(WAVS / f'{i}.wav')))
        y = vocoder(mel)
        write_wav(out / tag / f'{i}.wav', y)
        rows.append({'id': i, 'vocoder': tag, 'audio_s': len(y) / SR, **mel_distance(ap, mel, y)})
    nat = {i: out / 'natural' / f'{i}.wav' for i in ids}
    f0 = f0_table({i: out / tag / f'{i}.wav' for i in ids}, nat).assign(vocoder=tag)
    df = pd.DataFrame(rows).merge(f0, on=['id', 'vocoder'], how='left')
    df.to_csv(table, index=False)
    return df


def finetuned(tag: str, step: int) -> Path:
    """Directory of a fine-tuned generator: `runs/vocoder_ft/<tag>/step_<step>`."""
    return RUNS / 'vocoder_ft' / tag / f'step_{step}'


def evaluate_finetuned(tag: str, step: int) -> None:
    """Everything the report compares for one fine-tuned vocoder (cached like the rest)."""
    name = f'ft_{tag}_{step}'
    source = str(finetuned(tag, step))
    copysynth_vocoder(source, name)
    model_mels(vocoder_source=source, tag=name)
    from tts_pipeline import TTSPipeline
    phrases(TTSPipeline(vocoder_source=source), out=RUNS / f'phrases_{name}', tag=name)


# ----------------------------------------------------------------------------------------
# 2. lab 4 model mels -> HiFi-GAN
# ----------------------------------------------------------------------------------------
def model_mels(out: Path = RUNS / 'model_mels', vocoder_source: str | None = None,
               tag: str = 'hifigan') -> pd.DataFrame:
    """The mels lab 4 synthesized (and vocoded with Griffin-Lim), now through HiFi-GAN."""
    table = out / f'metrics_{tag}.csv'
    if table.exists():
        return pd.read_csv(table)
    ap = audio_processor()
    vocoder = hifigan_fn(load_hifigan(vocoder_source) if vocoder_source else load_hifigan())
    ids = list(read_eval_set())
    rows, frames = [], []
    for run, eval_dir in MODEL_MELS.items():
        mels = {}
        for i in ids:
            mel = np.load(eval_dir / 'mels' / f'{i}_model.npy')
            nat = np.load(eval_dir / 'mels' / f'{i}_natural.npy')
            y = vocoder(mel)
            write_wav(out / tag / run / f'{i}.wav', y)
            mels[i] = (nat, mel)
            rows.append({'id': i, 'run': run, 'vocoder': tag, 'audio_s': len(y) / SR,
                         'ripple_in': lab4_analysis.harmonic_ripple(mel), **mel_distance(ap, mel, y)})
        natural = {i: eval_dir / 'natural' / f'{i}.wav' for i in ids}
        for name, paths in ((tag, {i: out / tag / run / f'{i}.wav' for i in ids}),
                            ('griffinlim', {i: eval_dir / 'model' / f'{i}.wav' for i in ids})):
            if name == 'griffinlim' and (out / f'f0_griffinlim_{run}.csv').exists():
                continue
            f0 = f0_table(paths, natural, mels).assign(run=run, vocoder=name)
            if name == 'griffinlim':
                f0.to_csv(out / f'f0_griffinlim_{run}.csv', index=False)
            else:
                frames.append(f0)
    df = pd.DataFrame(rows).merge(pd.concat(frames), on=['id', 'run', 'vocoder'], how='left')
    df.to_csv(table, index=False)
    return df


def griffinlim_f0(out: Path = RUNS / 'model_mels') -> pd.DataFrame:
    return pd.concat([pd.read_csv(p) for p in sorted(out.glob('f0_griffinlim_*.csv'))])


# ----------------------------------------------------------------------------------------
# 3. the test phrases through the whole pipeline
# ----------------------------------------------------------------------------------------
def read_phrases(path: Path) -> list[tuple[str, str]]:
    with path.open(encoding='utf-8') as fh:
        next(fh)
        return [tuple(line.rstrip('\n').split('|', 1)) for line in fh if line.strip()]


def phrases(pipeline=None, out: Path = AUDIO, tag: str = 'pipeline') -> pd.DataFrame:
    """Synthesize `data/evaluation_text*.csv` into `audio/` and log every stage."""
    table = RUNS / f'phrases_{tag}.csv'
    if table.exists():
        return pd.read_csv(table)
    import json
    if pipeline is None:
        from tts_pipeline import TTSPipeline
        pipeline = TTSPipeline()
    pipeline('Прогрев.')
    rows = []
    for name, markup in (('evaluation_text.csv', False), ('evaluation_text_pause.csv', True)):
        for pid, text in read_phrases(HERE / 'data' / name):
            wav, t = pipeline(text, markup=markup)
            write_wav(out / f'{pid}.wav', wav)
            rows.append({'id': pid, 'text': text, 'chars': len(text),
                         'phones': sum(len(p) for p in t.log['phones']),
                         'pieces': len(t.log['phones']), 'audio_s': t.audio_seconds,
                         'total_s': t.total, 'rtf': t.rtf,
                         **{f'{k}_ms': v * 1000 for k, v in t.stages.items()},
                         'normalized': t.log['normalized'],
                         'sentences': json.dumps(t.log['sentences'], ensure_ascii=False),
                         'phone_pieces': json.dumps(t.log['phones'], ensure_ascii=False),
                         'warnings': json.dumps(t.log['warnings'], ensure_ascii=False)})
    df = pd.DataFrame(rows)
    RUNS.mkdir(exist_ok=True)
    df.to_csv(table, index=False)
    return df


# ----------------------------------------------------------------------------------------
# 4. RTF against length
# ----------------------------------------------------------------------------------------
def rtf_texts(n_per_bucket: int = 6, seed: int = 0) -> list[str]:
    """Inputs of growing length: single RUSLAN test sentences, then several glued together."""
    rng = np.random.default_rng(seed)
    texts = list(read_eval_set().values())
    out = []
    for k in (1, 1, 2, 4, 8, 16):
        for _ in range(n_per_bucket):
            out.append(' '.join(rng.choice(texts, k, replace=False)))
    return out


def rtf(device: str = 'cuda', repeats: int = 3, run_dir=None) -> pd.DataFrame:
    """Per-stage time and RTF for `rtf_texts()`, median over `repeats` (after a warm-up)."""
    table = RUNS / f'rtf_{device}.csv'
    if table.exists():
        return pd.read_csv(table)
    import torch
    from tts_pipeline import DEFAULT_RUN, TTSPipeline
    if device == 'cpu':
        torch.set_num_threads(os.cpu_count() or 4)
    pipeline = TTSPipeline(run_dir or DEFAULT_RUN, device=device)
    for _ in range(2):
        pipeline('Прогрев конвейера синтеза.')
    rows = []
    for n, text in enumerate(rtf_texts()):
        for r in range(repeats):
            wav, t = pipeline(text)
            rows.append({'n': n, 'repeat': r, 'chars': len(text),
                         'phones': sum(len(p) for p in t.log['phones']), 'pieces': len(t.log['phones']),
                         'audio_s': t.audio_seconds, 'total_s': t.total, 'rtf': t.rtf,
                         **{f'{k}_s': v for k, v in t.stages.items()}})
    df = pd.DataFrame(rows).groupby('n').median(numeric_only=True).reset_index().assign(device=device)
    RUNS.mkdir(exist_ok=True)
    df.to_csv(table, index=False)
    return df


# ----------------------------------------------------------------------------------------
# 5. ASR
# ----------------------------------------------------------------------------------------
def normalize_for_asr(text: str) -> str:
    text = text.lower().replace('ё', 'е')
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def edit_distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def asr_jobs() -> list[tuple[str, str, dict[str, Path], dict[str, str]]]:
    """(set, variant, {id: wav}, {id: reference text}) for everything that exists on disk."""
    jobs = []
    eval_texts = read_eval_set()
    for d in sorted(p for p in (RUNS / 'copysynth').glob('*/') if p.is_dir()):
        jobs.append(('copysynth', d.name, {i: d / f'{i}.wav' for i in eval_texts}, eval_texts))
    for tag_dir in sorted((RUNS / 'model_mels').glob('*/')):
        for run_dir in sorted(p for p in tag_dir.glob('*/') if p.is_dir()):
            jobs.append((f'model_mels/{run_dir.name}', tag_dir.name,
                         {i: run_dir / f'{i}.wav' for i in eval_texts}, eval_texts))
    # test phrases: the reference is the *normalized* text (numbers spelled out, Latin
    # transliterated) — the words the pipeline was asked to say
    table = RUNS / 'phrases_pipeline.csv'
    if table.exists():
        texts = dict(pd.read_csv(table)[['id', 'normalized']].values)
        if (AUDIO / 'eval_01.wav').exists():
            jobs.append(('phrases', 'pipeline', {i: AUDIO / f'{i}.wav' for i in texts}, texts))
        for sub in sorted(p for p in RUNS.glob('phrases_*/') if p.is_dir()):
            jobs.append(('phrases', sub.name.removeprefix('phrases_'),
                         {i: sub / f'{i}.wav' for i in texts}, texts))
    return jobs


def asr(batch_size: int = 8, references: dict[str, str] | None = None) -> pd.DataFrame:
    """Whisper on every job of :func:`asr_jobs` not yet in `runs/asr.csv`."""
    done = set()
    if ASR_CSV.exists():
        old = pd.read_csv(ASR_CSV, sep='|', keep_default_na=False)
        done = set(zip(old.set, old.variant, old.id))
    todo = [(s, v, {i: p for i, p in paths.items() if p.exists() and (s, v, i) not in done}, refs)
            for s, v, paths, refs in asr_jobs()]
    todo = [j for j in todo if j[2]]
    if todo:
        import torch
        from transformers import pipeline as hf_pipeline
        device = 0 if torch.cuda.is_available() else -1
        model = hf_pipeline('automatic-speech-recognition', model=ASR_MODEL, device=device,
                            torch_dtype=torch.float16 if device == 0 else torch.float32)
        new = not ASR_CSV.exists()
        RUNS.mkdir(exist_ok=True)
        with ASR_CSV.open('a', encoding='utf-8', newline='') as fh:
            writer = csv.DictWriter(fh, ASR_FIELDS, delimiter='|', quoting=csv.QUOTE_NONE, escapechar='\\')
            if new:
                writer.writeheader()
            for s, v, paths, refs in todo:
                ids = list(paths)
                audio = [{'raw': sf.read(str(paths[i]), dtype='float32')[0], 'sampling_rate': SR}
                         for i in ids]
                # > 30 s switches Whisper to long-form decoding, whose inputs cannot share a
                # batch with short ones: those go one by one
                kw = dict(return_timestamps=True, generate_kwargs={'language': 'russian', 'task': 'transcribe'})
                long = [len(a['raw']) > 30 * SR for a in audio]
                short_hyps = iter(model([a for a, l in zip(audio, long) if not l], batch_size=batch_size, **kw)
                                  if not all(long) else [])
                hyps = [model(a, **kw) if l else next(short_hyps) for a, l in zip(audio, long)]
                for i, h in zip(ids, hyps):
                    ref, hyp = normalize_for_asr(refs[i]), normalize_for_asr(h['text'])
                    writer.writerow({'set': s, 'variant': v, 'id': i, 'ref': ref,
                                     'hyp': hyp.replace('|', ' '),
                                     'cer': round(edit_distance(ref, hyp) / max(1, len(ref)), 4),
                                     'wer': round(edit_distance(ref.split(), hyp.split())
                                                  / max(1, len(ref.split())), 4)})
                fh.flush()
                print(f'{s:28s} {v:12s} n={len(ids)}')
    return pd.read_csv(ASR_CSV, sep='|', keep_default_na=False)


if __name__ == '__main__':
    sys.path.insert(0, str(HERE))
    step = sys.argv[1] if len(sys.argv) > 1 else ''
    if step == 'copysynth':
        print(copysynth().groupby('vocoder').mean(numeric_only=True).round(3).T)
    elif step == 'model_mels':
        print(model_mels().groupby('run').mean(numeric_only=True).round(3).T)
    elif step == 'phrases':
        print(phrases()[['id', 'audio_s', 'total_s', 'rtf']])
    elif step == 'rtf':
        for dev in sys.argv[2:] or ['cuda', 'cpu']:
            print(rtf(dev)[['chars', 'audio_s', 'total_s', 'rtf']])
    elif step == 'finetuned':                       # python lab5_eval.py finetuned gt 20000
        evaluate_finetuned(sys.argv[2], int(sys.argv[3]))
    elif step == 'asr':
        print(asr().groupby(['set', 'variant'])[['cer', 'wer']].mean().round(3))
    else:
        print(__doc__)
