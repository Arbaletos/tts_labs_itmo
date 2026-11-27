"""TTS service — lab 5, extra task 3.

REST API (FastAPI + uvicorn) and a Gradio demo page on top of :class:`TTSPipeline`::

    python service.py                       # http://127.0.0.1:8000, demo at /demo
    python service.py --host 0.0.0.0        # reachable from the local network
    python service.py --share               # + a temporary public Gradio link

    curl -X POST http://127.0.0.1:8000/synthesize -H "Content-Type: application/json" \
         -d '{"text": "Привет, мир!"}' -o out.wav

The models are loaded once, when the process starts; a request only runs the pipeline. The
GPU model is not meant to be called from several threads at once, and FastAPI runs plain
`def` handlers in a thread pool, so synthesis is serialized with a lock.

The response is `audio/wav`; the measurements travel in headers (`X-RTF`,
`X-Synthesis-Seconds`, `X-Audio-Seconds`, `X-Stage-<name>-Ms`, `X-Warnings`) so a client sees
the real-time factor without a second request.
"""
import argparse
import io
import json
import threading
import urllib.parse
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from tts_pipeline import DEFAULT_RUN, SAMPLE_RATE, TTSPipeline

MAX_CHARS = 5000            # one request must not hold the GPU for minutes


class SynthesisRequest(BaseModel):
    text: str = Field(..., description='Russian text; <pause type="normal|none"> tags allowed')
    markup: bool = Field(True, description='parse <pause> tags')
    length_scale: float = Field(1.0, ge=0.5, le=2.0, description='>1 slower, <1 faster')


def wav_bytes(wav: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, wav, SAMPLE_RATE, format='WAV', subtype='PCM_16')
    return buffer.getvalue()


def create_app(pipeline: TTSPipeline) -> FastAPI:
    app = FastAPI(title='RUSLAN TTS (ITMO lab 5)')
    lock = threading.Lock()

    def synthesize(text: str, markup: bool, length_scale: float):
        if len(text) > MAX_CHARS:
            raise HTTPException(413, f'text longer than {MAX_CHARS} characters')
        with lock:
            return pipeline(text, markup=markup, length_scale=length_scale)

    @app.get('/health')
    def health() -> dict:
        return {'status': 'ok', 'device': pipeline.device,
                'checkpoint': str(pipeline.checkpoint.relative_to(pipeline.checkpoint.parents[3]))}

    @app.post('/synthesize', response_class=Response,
              responses={200: {'content': {'audio/wav': {}}}})
    def synthesize_endpoint(request: SynthesisRequest) -> Response:
        try:
            wav, timings = synthesize(request.text, request.markup, request.length_scale)
        except HTTPException:
            raise
        except Exception as error:                                   # never take the process down
            raise HTTPException(500, f'{type(error).__name__}: {error}') from error
        headers = {
            'X-RTF': f'{timings.rtf:.4f}' if timings.audio_seconds else 'inf',
            'X-Synthesis-Seconds': f'{timings.total:.4f}',
            'X-Audio-Seconds': f'{timings.audio_seconds:.3f}',
            # headers are latin-1: the warnings (Russian words) go percent-encoded JSON
            'X-Warnings': urllib.parse.quote(json.dumps(timings.log['warnings'], ensure_ascii=False)),
        }
        headers.update({f'X-Stage-{name}-Ms': f'{t * 1000:.1f}' for name, t in timings.stages.items()})
        return Response(wav_bytes(wav), media_type='audio/wav', headers=headers)

    @app.post('/frontend')
    def frontend_endpoint(request: SynthesisRequest) -> dict:
        """What the acoustic model would receive: normalized text, tokens, phones."""
        from tts_pipeline import Timings
        timings = Timings()
        pipeline.front_end(request.text, request.markup, timings)
        return {k: v for k, v in timings.log.items() if k != 'mels'}

    app.state.synthesize = synthesize
    return app


def mount_demo(app: FastAPI, synthesize) -> FastAPI:
    """A Gradio page at /demo: text box, tempo slider, player, timings."""
    import gradio as gr

    examples_file = Path(__file__).with_name('data') / 'evaluation_text.csv'
    examples = [line.split('|', 1)[1].strip()
                for line in examples_file.read_text(encoding='utf-8').splitlines()[1:]]
    examples += ['Казнить <pause type="normal"> нельзя, <pause type="none"> помиловать.']

    def run(text: str, length_scale: float):
        wav, timings = synthesize(text, True, length_scale)
        info = timings.report()
        if timings.log['warnings']:
            info += '\n\n' + '\n'.join(timings.log['warnings'])
        phones = '\n'.join(' '.join(p) for p in timings.log.get('phones', []))
        return (SAMPLE_RATE, wav), info, phones

    with gr.Blocks(title='RUSLAN TTS') as demo:
        gr.Markdown('## Синтез речи: лабы 1–5 (RUSLAN, FastSpeech 2 + HiFi-GAN)\n'
                    'Паузы можно задать тегами `<pause type="normal">` / `<pause type="none">`, '
                    'ударение — `+` перед гласной (`з+амок`).')
        text = gr.Textbox(label='Текст', lines=4, value=examples[0])
        speed = gr.Slider(0.7, 1.5, value=1.0, step=0.05, label='length_scale (>1 медленнее)')
        button = gr.Button('Синтезировать', variant='primary')
        audio = gr.Audio(label='Результат', type='numpy')
        timings = gr.Textbox(label='Время по этапам', lines=8)
        phones = gr.Textbox(label='Фонемы (по кускам)', lines=4)
        button.click(run, [text, speed], [audio, timings, phones])
        gr.Examples(examples, [text])
    return gr.mount_gradio_app(app, demo, path='/demo'), demo


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', type=Path, default=DEFAULT_RUN, help='lab 4 run directory')
    p.add_argument('--vocoder', default=None, help='HiFi-GAN hub id or local directory')
    p.add_argument('--device', default='cuda')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--no-demo', action='store_true', help='REST only, without Gradio')
    p.add_argument('--share', action='store_true',
                   help='also open a temporary public Gradio link (*.gradio.live)')
    args = p.parse_args()

    kwargs = {'vocoder_source': args.vocoder} if args.vocoder else {}
    pipeline = TTSPipeline(args.run, device=args.device, **kwargs)
    pipeline('Прогрев.')                                  # first call: CUDA/cuDNN initialization
    app = create_app(pipeline)
    if not args.no_demo:
        app, demo = mount_demo(app, app.state.synthesize)
        if args.share:
            _, _, public = demo.launch(share=True, prevent_thread_lock=True)
            print(f'public demo: {public}')

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
