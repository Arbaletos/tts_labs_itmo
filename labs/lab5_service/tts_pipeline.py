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
"""The full text-to-speech pipeline — lab 5."""
import json
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from labs.lab1_text.text_normalizer import TextNormalizer
from labs.lab2_pauses.pause_predictor import PausePredictor
from labs.lab3_g2p.g2p import G2P, build_lexicon, load_mfa_dict, load_tokens


SIL = '<SIL>'
SAMPLE_RATE = 22050
VOCODER_SOURCE = 'speechbrain/tts-hifigan-libritts-22050Hz'

MARKUP_RE = re.compile(r'<pause\s+type\s*=\s*"(normal|none)"\s*/?>')
FORCE, SUPPRESS = 'normal', 'none'


@dataclass
class Timings:
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
    requests: dict[int, str] = {}
    out: list[str] = []
    cursor = 0
    for match in MARKUP_RE.finditer(text):
        out.append(text[cursor:match.start()])
        n_words = len(re.findall(r'\w+', ''.join(out)))
        if n_words:
            requests[n_words - 1] = match.group(1)
        out.append(' ')
        cursor = match.end()
    out.append(text[cursor:])
    return ''.join(out), requests


def lookup_key(word: str) -> str:
    word = unicodedata.normalize('NFC', word).lower()
    return unicodedata.normalize('NFC',
                                 ''.join(c for c in unicodedata.normalize('NFD', word)
                                         if c != '́'))


class TTSPipeline:
    def __init__(self, run_dir, device='cuda', vocoder_source=VOCODER_SOURCE,
                 lab2_dir=None, tokens_phones_path=None, mfa_dict_path=None,
                 phones_path='common/phones_mfa.txt') -> None:
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
        checkpoint = run_dir / 'best_model.pth' 
        if not checkpoint.exists():
            checkpoint = sorted(run_dir.rglob('best_model*.pth'))[-1]

        config = load_config(str(config_path))
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

        # --- lab 1 ---
        self.normalizer = TextNormalizer()

        # --- lab 2 ---
        repo_root = Path(__file__).resolve().parents[2]
        lab2_dir = Path(lab2_dir) if lab2_dir else Path(__file__).resolve().parents[1] / 'lab2_pauses'
        self.pause_predictor = PausePredictor(model_dir=str(lab2_dir))

        # --- lab 3 ---
        tokens_phones_path = Path(tokens_phones_path or repo_root / 'labs' / 'lab3_g2p' / 'data' / 'tokens_phones.csv')
        mfa_dict_path = Path(mfa_dict_path or repo_root / 'labs' / 'lab3_g2p' / 'data' / 'mfa_dict.csv')
        tokens = load_tokens(str(tokens_phones_path))
        lexicon = build_lexicon(tokens[tokens.set == 'train'])
        mfa_dict = load_mfa_dict(str(mfa_dict_path))
        self.g2p = G2P(lexicon, mfa_dict)

        # --- инвентарь фонем ---
        phones_path = Path(phones_path)
        if not phones_path.is_absolute():
            phones_path = repo_root / phones_path
        with open(phones_path, encoding='utf-8') as fh:
            self.phone_set = {p.strip() for p in fh if p.strip()}

    # ---------------------------------------------------------------- front end
    def normalize(self, text: str) -> str:
        return self.normalizer.normalize(text)

    def tokenize(self, text: str) -> list[str]:
        """Слово + пунктуация до следующего слова — формат label_raw из lab 2."""
        tokens: list[str] = []
        for w in text.split():
            if tokens and all(not c.isalnum() for c in w):
                tokens[-1] += w
            else:
                tokens.append(w)
        return tokens

    def insert_pauses(self, tokens, requests=None):
        tokens_with_sil, _ = self.pause_predictor.predict_durations(tokens)
        tokens_with_sil = [str(t) for t in tokens_with_sil]

        if not requests:
            return tokens_with_sil

        base: list[str] = []
        sil_after: dict[int, bool] = {}
        src_idx = -1
        for t in tokens_with_sil:
            if t == SIL:
                if src_idx >= 0:
                    sil_after[src_idx] = True
            else:
                src_idx += 1
                base.append(t)

        for idx, kind in requests.items():
            if kind == FORCE:
                sil_after[idx] = True
            elif kind == SUPPRESS:
                sil_after[idx] = False

        out: list[str] = []
        for i, t in enumerate(base):
            out.append(t)
            if sil_after.get(i):
                out.append(SIL)
        return out

    def to_phones(self, tokens: list[str]) -> list[str]:
        # G2P не умеет пунктуацию: отрезаем хвостовую у не-<SIL> токенов
        cleaned = [t if t == SIL else re.sub(r'[^\w\-]+$', '', t) for t in tokens]
        per_token = self.g2p.transcribe(cleaned)
        if len(per_token) != len(tokens):
            raise ValueError(f'G2P вернул {len(per_token)} списков на {len(tokens)} токенов')
        phones = [p for word in per_token for p in word]
        phones = [p.replace('ː', '') for p in phones]
        unknown = [p for p in phones if p != SIL and p not in self.phone_set]
        if unknown:
            raise ValueError(f'фонемы вне инвентаря: {sorted(set(unknown))[:8]}')
        return phones

    # ------------------------------------------------------------- back end
    def mel(self, phones: list[str]) -> np.ndarray:
        import torch
        unknown = [p for p in phones if p not in self.phone_set]
        if unknown:
            msg = f'phones outside the inventory: {sorted(set(unknown))[:8]}'
            raise ValueError(msg)
        with torch.no_grad():
            out = self.model.synthesize(' '.join(phones), self.config, use_griffin_lim=False)
        return out['outputs']['model_outputs'][0].detach().cpu().numpy().T

    def vocode(self, mel: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            wav = self.vocoder.decode_batch(torch.tensor(mel).to(self.device))
        return wav.squeeze().cpu().numpy()

    SILENCE_SEC = 0.25

    @staticmethod
    def _chunk_tokens(tokens, max_chars=350):
        """Делит токены на фрагменты ≤ max_chars, предпочитая границы с пунктуацией."""
        chunks, cur, cur_len = [], [], 0
        for t in tokens:
            cur.append(t)
            cur_len += len(t) + 1
            end_punct = bool(re.search(r'[.!?…,:;—–-]$', t))
            if (end_punct and cur_len >= max_chars // 2) or cur_len >= max_chars:
                chunks.append(cur)
                cur, cur_len = [], 0
        if cur:
            chunks.append(cur)
        return chunks

    def __call__(self, text: str, markup: bool = False) -> tuple[np.ndarray, Timings]:
        timings = Timings()

        def timed(name, fn, *args):
            start = time.perf_counter()
            result = fn(*args)
            timings.stages[name] = timings.stages.get(name, 0.0) + time.perf_counter() - start
            return result

        text = unicodedata.normalize('NFC', str(text or ''))
        requests: dict[int, str] = {}
        if markup:
            text, requests = timed('markup', strip_markup, text)
        if not text.strip():
            return np.zeros(0, dtype=np.float32), timings

        normalized = timed('normalize', self.normalize, text)
        tokens = timed('tokenize', self.tokenize, normalized)
        chunks = self._chunk_tokens(tokens)
        if len(chunks) > 1:
            print(f'  длинный вход: {len(tokens)} токенов -> {len(chunks)} фрагментов')

        wavs: list[np.ndarray] = []
        for chunk in chunks:
            with_pauses = timed('pauses', self.insert_pauses, chunk, requests)
            phones = timed('g2p', self.to_phones, with_pauses)
            if not phones:
                continue
            mel = timed('acoustic', self.mel, phones)
            wavs.append(timed('vocoder', self.vocode, mel))
            wavs.append(np.zeros(int(self.SILENCE_SEC * SAMPLE_RATE), dtype=wavs[-1].dtype))

        if not wavs:
            return np.zeros(0, dtype=np.float32), timings
        wav = np.concatenate(wavs[:-1])  # без хвостовой тишины
        timings.audio_seconds = len(wav) / SAMPLE_RATE
        return wav, timings

    def save(self, wav: np.ndarray, path) -> None:
        self.ap.save_wav(wav, str(path), sr=SAMPLE_RATE)