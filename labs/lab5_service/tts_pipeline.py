"""The full text-to-speech pipeline — lab 5.

Assembles labs 1–4 plus a pretrained vocoder into one callable::

    pipeline = TTSPipeline(run_dir='../lab4_acoustic/runs/baseline')
    wav, timings = pipeline('Здравствуйте, ваш заказ готов.')

The back half — acoustic model and vocoder — is provided and works. The four hooks marked
`TODO` are where your own modules from labs 1–3 plug in.

Stage order::

    text
      → normalize()        lab 1
      → tokenize()         lab 2
      → insert_pauses()    lab 2, emits <SIL>
      → to_phones()        lab 3, <SIL> passes through
      → " ".join()
      → mel()              lab 4
      → vocode()           HiFi-GAN
    audio

Pause control markup is **optional** — extra task 1. :func:`strip_markup` is here for it,
but the required path never calls it.
"""
import json
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SIL = '<SIL>'
SAMPLE_RATE = 22050
VOCODER_SOURCE = 'speechbrain/tts-hifigan-libritts-22050Hz'

# Extra task 1 only. `type="normal"` forces a pause the predictor would not place,
# `type="none"` suppresses one it would.
MARKUP_RE = re.compile(r'<pause\s+type\s*=\s*"(normal|none)"\s*/?>')
FORCE, SUPPRESS = 'normal', 'none'


@dataclass
class Timings:
    """Wall clock per stage, and the real-time factor of the whole call.

    RTF = synthesis time / audio duration. Below 1.0 means faster than real time: at
    RTF 2.0 a ten-second phrase takes twenty seconds to produce.
    """
    stages: dict[str, float] = field(default_factory=dict)
    audio_seconds: float = 0.0

    @property
    def total(self) -> float:
        return sum(self.stages.values())

    @property
    def rtf(self) -> float:
        return self.total / self.audio_seconds if self.audio_seconds else float('inf')

    def report(self) -> str:
        lines = [f'  {name:14s} {t * 1000:8.1f} ms  {t / max(self.total, 1e-9):5.1%}'
                 for name, t in self.stages.items()]
        lines.append(f'  {"total":14s} {self.total * 1000:8.1f} ms for '
                     f'{self.audio_seconds:.2f} s of audio  ->  RTF {self.rtf:.3f}')
        return '\n'.join(lines)


def strip_markup(text: str) -> tuple[str, dict[int, str]]:
    """Pull pause-control tags out of the text. **Extra task 1 only.**

    Call this *before* the normalizer, not after: lab 1 lists `<` and `>` among the
    characters to reject, so a normalizer handed `<pause type="none">` would throw the
    instruction away. Real engines parse SSML before text normalization for the same
    reason.

    A tag becomes a **space**, not nothing. `одно<pause type="normal">потом` would
    otherwise collapse to `однопотом` — one word instead of two — and every pause index
    after it would be off by one.

    Args:
        text: Raw input, possibly containing `<pause type="normal">` or `type="none"`.

    Returns:
        The text with tags replaced by spaces, and `{token_index: 'normal' | 'none'}`
        where `token_index` is the **0-based index of the token the tag follows**. A tag
        before the first word has nothing to attach to and is dropped.
    """
    requests: dict[int, str] = {}
    out: list[str] = []
    cursor = 0
    for match in MARKUP_RE.finditer(text):
        out.append(text[cursor:match.start()])
        n_words = len(re.findall(r'\w+', ''.join(out)))
        if n_words:                                   # a tag before any word has no anchor
            requests[n_words - 1] = match.group(1)
        out.append(' ')
        cursor = match.end()
    out.append(text[cursor:])
    return ''.join(out), requests


def lookup_key(word: str) -> str:
    """The form to look a word up by: NFC, lowercase, no combining acute."""
    word = unicodedata.normalize('NFC', word).lower()
    return unicodedata.normalize('NFC',
                                 ''.join(c for c in unicodedata.normalize('NFD', word)
                                         if c != '́'))


class TTSPipeline:
    """Text in, waveform out.

    Models load once in `__init__`, never per call. That is not tidiness: loading
    FastSpeech2 and HiFi-GAN takes seconds, and doing it per request puts the RTF far above
    1 no matter how fast the model is.

    Args:
        run_dir: A lab 4 run directory holding `config.json` and `best_model_*.pth`.
        device: `'cuda'` or `'cpu'`. RTF differs by roughly an order of magnitude.
        vocoder_source: SpeechBrain HiFi-GAN to use; the default matches the audio config
            every lab has been using.
    """

    def __init__(self, run_dir: str | Path, device: str = 'cuda',
                 vocoder_source: str = VOCODER_SOURCE) -> None:
        import torch
        from speechbrain.inference.vocoders import HIFIGAN
        from TTS.config import load_config
        from TTS.tts.configs.shared_configs import CharactersConfig
        from TTS.tts.models.forward_tts import ForwardTTS
        from TTS.tts.utils.text.tokenizer import TTSTokenizer
        from TTS.utils.audio import AudioProcessor

        self.device = device
        run_dir = Path(run_dir)
        config_path = next(run_dir.rglob('config.json'))
        checkpoint = sorted(run_dir.rglob('best_model*.pth'))[-1]

        config = load_config(str(config_path))
        # `load_config` cannot deserialize a list phone inventory: coqpit decides the
        # `characters` field should be a str, fails, and silently substitutes None, leaving
        # the tokenizer on coqui's default Latin set with no error. Rebuild it from the raw
        # JSON — of the checkpoint, so the vocabulary stays tied to these weights rather
        # than to whatever common/phones_mfa.txt says today.
        with open(config_path, encoding='utf-8') as fh:
            config.characters = CharactersConfig(**json.load(fh)['characters'])

        self.ap = AudioProcessor.init_from_config(config)
        self.tokenizer, self.config = TTSTokenizer.init_from_config(config)
        self.model = ForwardTTS(self.config, self.ap, self.tokenizer, speaker_manager=None)
        self.model.load_state_dict(torch.load(str(checkpoint), map_location='cpu')['model'])
        self.model.eval().to(device)

        self.vocoder = HIFIGAN.from_hparams(
            source=vocoder_source,
            savedir=str(run_dir / 'pretrained_vocoder'),
            run_opts={'device': device},
        )
        self.phone_set = set(self.config.characters.characters)

    # ---------------------------------------------------------------- front end (yours)
    def normalize(self, text: str) -> str:
        """Normalize the input. **TODO: your lab 1 normalizer.**"""
        raise NotImplementedError('lab 1 normalizer')

    def tokenize(self, text: str) -> list[str]:
        """Split into tokens carrying their trailing punctuation. **TODO: your lab 2.**

        The pause predictor was trained on `label_raw`, so the tokens here must have the
        same shape: a word plus the punctuation up to the next word.
        """
        raise NotImplementedError('lab 2 tokenizer')

    def insert_pauses(self, tokens: list[str],
                      requests: dict[int, str] | None = None) -> list[str]:
        """Return the token sequence with `<SIL>` where a pause belongs. **TODO: lab 2.**

        `requests` is empty on the required path. For extra task 1 it carries the output of
        :func:`strip_markup`: `'normal'` forces a `<SIL>` after that token, `'none'`
        suppresses one.
        """
        raise NotImplementedError('lab 2 pause predictor')

    def to_phones(self, tokens: list[str]) -> list[str]:
        """Token sequence to flat phone sequence. **TODO: your lab 3.**

        `<SIL>` maps to itself. Every output phone must be in `common/phones_mfa.txt`;
        anything else becomes an unknown token in the acoustic model and is not reported.
        """
        raise NotImplementedError('lab 3 G2P')

    # ------------------------------------------------------------- back end (provided)
    def mel(self, phones: list[str]) -> np.ndarray:
        """Phone sequence to mel spectrogram `[n_mels, T]`.

        The phones go in space-separated because that is what
        `multicharacter_phoneme_cleaners` splits on.
        """
        import torch
        unknown = [p for p in phones if p not in self.phone_set]
        if unknown:
            msg = f'phones outside the inventory: {sorted(set(unknown))[:8]}'
            raise ValueError(msg)
        with torch.no_grad():
            out = self.model.synthesize(' '.join(phones), self.config, use_griffin_lim=False)
        return out['outputs']['model_outputs'][0].detach().cpu().numpy().T

    def vocode(self, mel: np.ndarray) -> np.ndarray:
        """Mel `[n_mels, T]` to waveform at 22 050 Hz."""
        import torch
        with torch.no_grad():
            wav = self.vocoder.decode_batch(torch.tensor(mel).to(self.device))
        return wav.squeeze().cpu().numpy()

    # ----------------------------------------------------------------------- the whole
    def __call__(self, text: str, markup: bool = False) -> tuple[np.ndarray, Timings]:
        """Synthesize one input, timing every stage.

        Empty or whitespace-only input returns an empty waveform rather than raising: a
        service that crashes on a blank request is not a service.

        Args:
            text: The input.
            markup: Parse `<pause type=...>` tags before normalizing. Off by default —
                extra task 1.
        """
        timings = Timings()

        def timed(name, fn, *args):
            start = time.perf_counter()
            result = fn(*args)
            timings.stages[name] = time.perf_counter() - start
            return result

        text = unicodedata.normalize('NFC', str(text or ''))
        requests: dict[int, str] = {}
        if markup:
            text, requests = timed('markup', strip_markup, text)
        if not text.strip():
            return np.zeros(0, dtype=np.float32), timings

        normalized = timed('normalize', self.normalize, text)
        tokens = timed('tokenize', self.tokenize, normalized)
        with_pauses = timed('pauses', self.insert_pauses, tokens, requests)
        phones = timed('g2p', self.to_phones, with_pauses)
        if not phones:
            return np.zeros(0, dtype=np.float32), timings

        mel = timed('acoustic', self.mel, phones)
        wav = timed('vocoder', self.vocode, mel)
        timings.audio_seconds = len(wav) / SAMPLE_RATE
        return wav, timings

    def save(self, wav: np.ndarray, path: str | Path) -> None:
        """Write a waveform with the AudioProcessor, so the sample rate matches the rest."""
        self.ap.save_wav(wav, str(path), sr=SAMPLE_RATE)
