"""Синтез evaluation_text.csv и evaluation_text_pause.csv, замер RTF."""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

# чтобы `from labs...` работал при запуске из любой директории
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from labs.lab5_service.tts_pipeline import TTSPipeline


def read_csv(path: Path):
    with open(path, encoding='utf-8') as fh:
        return list(csv.DictReader(fh, delimiter='|'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--lab2-dir', default=None)
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--out-dir', default='labs/lab5_service/audio')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--markup', action='store_true')
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    pipe = TTSPipeline(run_dir=args.run_dir, device=args.device, lab2_dir=args.lab2_dir)
    print(f'модели загружены за {time.perf_counter() - t0:.1f} с')

    fname = 'evaluation_text_pause.csv' if args.markup else 'evaluation_text.csv'
    rows = read_csv(Path(args.data_dir) / fname)

    summary = []
    for row in rows:
        try:
            wav, tim = pipe(row['text'], markup=args.markup)
            path = out / f"{row['id']}.wav"
            pipe.save(wav, path)
            summary.append({'id': row['id'], 'audio_s': round(tim.audio_seconds, 3),
                            'total_s': round(tim.total, 3), 'rtf': round(tim.rtf, 3),
                            'stages': {k: round(v, 4) for k, v in tim.stages.items()}})
            print(f"{row['id']:14s} {tim.audio_seconds:6.2f}s  RTF {tim.rtf:.3f}  -> {path}")
        except Exception as e:
            print(f"{row['id']:14s} FAIL: {e}")
            summary.append({'id': row['id'], 'error': str(e)})

    with open(out / ('rtf_pause.json' if args.markup else 'rtf.json'), 'w',
              encoding='utf-8') as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()