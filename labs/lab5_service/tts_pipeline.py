"""The full text-to-speech pipeline — lab 5.

Assembles labs 1–4 plus a pretrained vocoder into one callable::

    pipeline = TTSPipeline(run_dir='../lab4_acoustic/runs/pitch_energy')
    wav, timings = pipeline('Здравствуйте, ваш заказ готов.')

Stage order::

    text
      → markup         extra task 1: <pause type=...> tags cut out *before* lab 1
      → normalize()    lab 1 (+ frontend_rules: symbols, Latin, abbreviations, `+` stress)
      → tokenize()     lab 2 tokenizer: a word plus its trailing punctuation
      → sentences      the token stream is cut into sentences
      → insert_pauses() lab 2 pause predictor, emits <SIL>; one final <SIL> per sentence
      → to_phones()    lab 3 G2P, <SIL> passes through
      → chunks         a sentence longer than the model ever saw is cut at its pauses
      → mel()          lab 4 FastSpeech 2
      → vocode()       HiFi-GAN
    audio

Every sentence (and every chunk of an over-long one) is synthesized separately and the
waveforms are concatenated. Two reasons: the lab 4 model was trained on utterances of at most
10 s (`max_audio_len = 220500`), and coqui's positional encoding is limited to 5000 frames
(58 s) — longer input is not degraded but rejected with an error.

`TTSPipeline(run_dir=None)` loads the text front end only (labs 1–3, no torch): enough to
inspect what the acoustic model would receive.
"""
import json
import re
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _lab in ('lab1_text', 'lab2_pauses', 'lab3_g2p'):
    if str(ROOT / 'labs' / _lab) not in sys.path:
        sys.path.insert(0, str(ROOT / 'labs' / _lab))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import frontend_rules  # noqa: E402

SIL = '<SIL>'
SAMPLE_RATE = 22050
VOCODER_SOURCE = 'speechbrain/tts-hifigan-libritts-22050Hz'
PHONES_FILE = ROOT / 'common' / 'phones_mfa.txt'
LEXICON_FILE = ROOT / 'labs' / 'lab3_g2p' / 'data' / 'v3' / 'lexicon.csv'
DEFAULT_RUN = ROOT / 'labs' / 'lab4_acoustic' / 'runs' / 'pitch_energy'

# Longest phone sequence sent to the acoustic model in one call. Utterances of the lab 4
# training set (<= 10 s) have at most ~150 phones; see the report, section 4.
MAX_PHONES = 150

# Extra task 1. `type="normal"` forces a pause the predictor would not place,
# `type="none"` suppresses one it would.
MARKUP_RE = re.compile(r'<pause\s+type\s*=\s*"(normal|none)"\s*/?>')
# Anything else that looks like a tag in markup mode: unknown type, unclosed, other elements.
BROKEN_TAG_RE = re.compile(r'</?\s*[A-Za-z][\w-]*(?:\s+[\w-]+\s*=\s*"[^"<>]*"?)*\s*/?>?')
FORCE, SUPPRESS = 'normal', 'none'
_SENTENCE_END = '.!?…'


@dataclass
class Timings:
    """Wall clock per stage, and the real-time factor of the whole call.

    RTF = synthesis time / audio duration. Below 1.0 means faster than real time: at
    RTF 2.0 a ten-second phrase takes twenty seconds to produce.

    `log` keeps what every stage produced (normalized text, tokens, phones, warnings): a
    pipeline is debugged by looking at the intermediate forms, not at the audio.
    """
    stages: dict[str, float] = field(default_factory=dict)
    audio_seconds: float = 0.0
    log: dict = field(default_factory=lambda: {'warnings': []})

    @property
    def total(self) -> float:
        return sum(self.stages.values())

    @property
    def rtf(self) -> float:
        return self.total / self.audio_seconds if self.audio_seconds else float('inf')

    def add(self, name: str, seconds: float) -> None:
        self.stages[name] = self.stages.get(name, 0.0) + seconds

    def warn(self, message: str) -> None:
        self.log['warnings'].append(message)

    def report(self) -> str:
        lines = [f'  {name:14s} {t * 1000:8.1f} ms  {t / max(self.total, 1e-9):5.1%}'
                 for name, t in self.stages.items()]
        lines.append(f'  {"total":14s} {self.total * 1000:8.1f} ms for '
                     f'{self.audio_seconds:.2f} s of audio  ->  RTF {self.rtf:.3f}')
        return '\n'.join(lines)


def strip_markup(text: str) -> tuple[str, dict[int, str]]:
    """Pull pause-control tags out of the text (the version shipped with the assignment).

    Returns the text with tags replaced by spaces, and `{token_index: 'normal' | 'none'}`
    where `token_index` is the 0-based index of the `\\w+` word the tag follows.

    **Not used by the pipeline**, kept for comparison (report, section 7): the index counts
    `\\w+` runs of the *raw* text, while the pause predictor sees lab 2 tokens of the
    *normalized* text. The two disagree whenever normalization changes the word count
    (`1995` -> four words) or a token holds several `\\w+` runs (`что-то`, `Шипе-Тотека`) —
    the pause then lands after the wrong word. :func:`split_markup` avoids indices
    altogether.
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


def split_markup(text: str) -> list[tuple[str, str | None]]:
    """Cut the text at pause tags: `[(segment, tag that follows it or None), ...]`.

    Each segment is normalized and tokenized on its own, and the tag is attached to the last
    token the segment produced — whatever normalization did to the words in between.
    """
    segments: list[tuple[str, str | None]] = []
    cursor = 0
    for match in MARKUP_RE.finditer(text):
        segments.append((text[cursor:match.start()], match.group(1)))
        cursor = match.end()
    segments.append((text[cursor:], None))
    return segments


def drop_broken_tags(segment: str) -> tuple[str, list[str]]:
    """Remove what looks like a tag but is not a valid pause tag (`<pause type="long">`,
    `<pause type="normal"` without `>`, `<b>`): otherwise lab 1 deletes only `< >` and the
    rest of the tag is *read aloud* ("пауз тайп нормал")."""
    found = BROKEN_TAG_RE.findall(segment)
    return (BROKEN_TAG_RE.sub(' ', segment), found) if found else (segment, [])


def vocoder_savedir(source: str | Path) -> Path:
    """Where SpeechBrain copies a vocoder: one folder per hub id or local checkpoint.

    A local fine-tuned checkpoint is `runs/vocoder_ft/<run>/step_<N>`: the folder name alone
    (`step_20000`) is the same for every run, so the run name is part of the key.
    """
    path = Path(source)
    name = f'{path.parent.name}_{path.name}' if path.is_dir() else path.name
    return HERE / 'pretrained_models' / name


def trim_vocoder_padding(vocoder, wav: np.ndarray, hop: int = 256) -> np.ndarray:
    """Cut what SpeechBrain's HiFi-GAN adds around the input.

    `HifiganGenerator.inference` pads the mel with `inference_padding` (5) replicated frames
    on *each* side before vocoding, so the output is `(T + 10) * 256` samples and shifted by
    1280 samples (58 ms) against the mel. Harmless to the ear, but it breaks any frame-wise
    comparison with the input and adds a stretch of the edge frame at both ends.
    """
    pad = int(getattr(vocoder.hparams.generator, 'inference_padding', 0) or 0) * hop
    return wav[pad:len(wav) - pad] if pad and len(wav) > 2 * pad else wav


def lookup_key(word: str) -> str:
    """The form to look a word up by: NFC, lowercase, no combining acute."""
    word = unicodedata.normalize('NFC', word).lower()
    return unicodedata.normalize('NFC',
                                 ''.join(c for c in unicodedata.normalize('NFD', word)
                                         if c != '́'))


def load_lexicon(path: str | Path = LEXICON_FILE) -> dict[str, Counter]:
    """`word|phones|count` (lab 3, `prepare_lexicon.py`) -> `word -> Counter({phones: count})`."""
    lexicon: dict[str, Counter] = {}
    with open(path, encoding='utf-8') as fh:
        next(fh)
        for line in fh:
            word, phones, count = line.rstrip('\n').split('|')
            lexicon.setdefault(unicodedata.normalize('NFC', word), Counter())[
                unicodedata.normalize('NFC', phones)] += int(count)
    return lexicon


class TTSPipeline:
    """Text in, waveform out.

    Models load once in `__init__`, never per call. That is not tidiness: loading
    FastSpeech2 and HiFi-GAN takes seconds, and doing it per request puts the RTF far above
    1 no matter how fast the model is.

    Args:
        run_dir: A lab 4 run directory holding `config.json` and `best_model_*.pth`;
            None loads only the text front end (labs 1–3).
        device: `'cuda'` or `'cpu'`; falls back to CPU when CUDA is unavailable.
        vocoder_source: SpeechBrain HiFi-GAN to use (a hub id or a local directory).
        checkpoint: Explicit checkpoint instead of the newest `best_model_*.pth`.
        max_phones: Longest phone sequence per acoustic-model call (see module docstring).
    """

    def __init__(self, run_dir: str | Path | None = DEFAULT_RUN, device: str = 'cuda',
                 vocoder_source: str = VOCODER_SOURCE, checkpoint: str | Path | None = None,
                 max_phones: int = MAX_PHONES) -> None:
        self.max_phones = max_phones
        self._load_front_end()
        self.model = self.vocoder = None
        self.device = 'cpu'
        if run_dir is None:
            return

        import torch
        from speechbrain.inference.vocoders import HIFIGAN
        from speechbrain.utils.fetching import LocalStrategy
        from TTS.config import load_config
        from TTS.tts.configs.shared_configs import CharactersConfig
        from TTS.tts.models.forward_tts import ForwardTTS
        from TTS.tts.utils.text.tokenizer import TTSTokenizer
        from TTS.utils.audio import AudioProcessor

        self.device = device if (device == 'cpu' or torch.cuda.is_available()) else 'cpu'
        run_dir = Path(run_dir)
        config_path = next(run_dir.rglob('config.json'))
        if checkpoint is None:
            checkpoint = sorted(run_dir.rglob('best_model_*.pth'),
                                key=lambda p: int(re.findall(r'\d+', p.stem)[-1]))[-1]
        self.checkpoint = Path(checkpoint)

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
        self.model.eval().to(self.device)

        self.vocoder = HIFIGAN.from_hparams(
            source=vocoder_source,
            savedir=str(vocoder_savedir(vocoder_source)),
            run_opts={'device': 'cuda:0' if self.device == 'cuda' else self.device},
            # symlinks from the HF cache need admin rights on Windows (WinError 1314)
            local_strategy=LocalStrategy.COPY,
        )
        self.phone_set = set(self.config.characters.characters)

    def _load_front_end(self) -> None:
        from g2p import G2P
        from pause_predictor import PausePredictor
        from text_normalizer import _ANGLICISMS, TextNormalizer

        self.normalizer = TextNormalizer()
        self.latin_exceptions = dict(_ANGLICISMS)
        self.pause_predictor = PausePredictor(train_if_missing=False)
        if self.pause_predictor.clf is None:
            raise FileNotFoundError('lab 2 model labs/lab2_pauses/models/pause_model.pkl is missing')
        # Lexicon of the whole v3 alignment: the dictionary the lab 4 model's phones came from.
        self.g2p = G2P(load_lexicon())
        self.phone_set = {unicodedata.normalize('NFC', p.strip())
                          for p in PHONES_FILE.read_text(encoding='utf-8').splitlines() if p.strip()}

    # ---------------------------------------------------------------- front end
    def is_word(self, key: str) -> bool:
        """Is `key` (lowercase) an ordinary dictionary word? Decides `МИР` vs `МГУ`."""
        return self.g2p.in_dictionary(lookup_key(key))

    def normalize(self, text: str) -> str:
        """Lab 1 normalizer in its free-text mode (`expand=True`: numbers, abbreviations).

        `frontend_rules.prepare` runs first: it handles what lab 1 would delete or garble
        (see that module) — symbols between words, `+` stress marks, Latin words, `ТВ`.
        """
        text = frontend_rules.prepare(text, self.is_word, self.latin_exceptions)
        return self.normalizer.normalize(text, expand=True)

    def tokenize(self, text: str) -> list[str]:
        """Lab 2 tokenizer: a word plus the punctuation up to the next word (`label_raw`)."""
        from pause_predictor import tokenize_text
        return tokenize_text(text)

    @staticmethod
    def split_sentences(tokens: list[str]) -> list[tuple[int, int]]:
        """`[start, stop)` token ranges of sentences.

        A sentence ends at a token whose trailing punctuation has `. ! ? …`, if the next token
        starts with a capital letter, a dash or a quote (`К. был` is not a boundary).
        """
        from pause_predictor import split_token
        bounds, start = [], 0
        for i, token in enumerate(tokens[:-1]):
            _, _, trail = split_token(token)
            if not any(c in trail for c in _SENTENCE_END):
                continue
            lead, core, _ = split_token(tokens[i + 1])
            if core[:1].isupper() or any(c in lead for c in '—-"\'«'):
                bounds.append((start, i + 1))
                start = i + 1
        if tokens:
            bounds.append((start, len(tokens)))
        return bounds

    def insert_pauses(self, tokens: list[str],
                      requests: dict[int, str] | None = None) -> list[str]:
        """Token sequence of one sentence with `<SIL>` where a pause belongs.

        `requests` (extra task 1) overrides the predictor: `'normal'` forces a `<SIL>` after
        that token, `'none'` suppresses one. A `<SIL>` always closes the sentence: every
        utterance of the lab 4 training data ends with one.
        """
        if not tokens:
            return []
        is_pause, _ = self.pause_predictor.predict(list(tokens))
        is_pause = list(is_pause)
        for index, kind in (requests or {}).items():
            if 0 <= index < len(tokens):
                is_pause[index] = int(kind == FORCE)
        out: list[str] = []
        for token, pause in zip(tokens[:-1], is_pause[:-1]):
            out.append(token)
            if pause:
                out.append(SIL)
        out += [tokens[-1], SIL]
        return out

    def transcribe(self, tokens: list[str]) -> list[list[str]]:
        """Lab 3 G2P, one phone list per token (`<SIL>` -> `[<SIL>]`)."""
        self.g2p.warm([t for t in tokens if t != SIL])
        return self.g2p.transcribe(tokens)

    def to_phones(self, tokens: list[str], timings: Timings | None = None) -> list[str]:
        """Token sequence to flat phone sequence; `<SIL>` passes through.

        Phones outside the model's inventory are dropped and reported, as are words that got
        no phones at all: both would otherwise vanish from the audio without a trace.
        """
        return [p for word in self._word_phones(tokens, timings) for p in word]

    def _word_phones(self, tokens: list[str], timings: Timings | None) -> list[list[str]]:
        words = []
        for token, phones in zip(tokens, self.transcribe(tokens)):
            unknown = [p for p in phones if p not in self.phone_set]
            if unknown and timings is not None:
                timings.warn(f'{token!r}: phones outside the inventory dropped: {unknown}')
            phones = [p for p in phones if p in self.phone_set]
            if not phones and timings is not None:
                timings.warn(f'{token!r}: no phones, the word is skipped')
            if phones == [SIL] and words and words[-1] == [SIL]:
                continue                                           # never two <SIL> in a row
            words.append(phones)
        return words

    def chunk(self, words: list[list[str]]) -> list[list[str]]:
        """Cut a sentence (phones per token) into pieces of at most `max_phones` phones.

        Cuts go at `<SIL>` — where the speaker pauses anyway — packing as many pause-to-pause
        groups into a piece as fit. A group longer than the limit on its own is cut between
        words. Every piece ends with `<SIL>`, like the training utterances.
        """
        groups, current = [], []
        for phones in words:
            if phones == [SIL]:
                if current:
                    groups.append(current)
                current = []
            elif phones:
                current.append(phones)
        if current:
            groups.append(current)

        pieces: list[list[str]] = []
        piece: list[str] = []
        for group in groups:
            flat = [p for w in group for p in w]
            if piece and len(piece) + 1 + len(flat) > self.max_phones:
                pieces.append(piece + [SIL])
                piece = []
            if len(flat) <= self.max_phones:
                piece += ([SIL] if piece else []) + flat
                continue
            for word in group:                                     # one group too long
                if piece and len(piece) + len(word) > self.max_phones:
                    pieces.append(piece + [SIL])
                    piece = []
                piece += word
        if piece:
            pieces.append(piece + [SIL])
        return pieces

    def front_end(self, text: str, markup: bool = False,
                  timings: Timings | None = None) -> list[list[str]]:
        """Text -> list of phone sequences, one per acoustic-model call."""
        timings = timings or Timings()

        def timed(name, fn, *args):
            start = time.perf_counter()
            result = fn(*args)
            timings.add(name, time.perf_counter() - start)
            return result

        text = unicodedata.normalize('NFC', str(text or ''))
        if markup:
            segments = timed('markup', split_markup, text)
        else:
            segments = [(text, None)]

        tokens: list[str] = []
        requests: dict[int, str] = {}
        normalized = []
        for segment, tag in segments:
            if markup:
                segment, broken = drop_broken_tags(segment)
                if broken:
                    timings.warn(f'unrecognized markup removed: {broken}')
            dropped = frontend_rules.dropped_characters(segment)
            if dropped:
                timings.warn(f'unsupported characters not pronounced: {"".join(dropped)}')
            norm = timed('normalize', self.normalize, segment)
            normalized.append(norm)
            tokens += timed('tokenize', self.tokenize, norm)
            if tag and tokens:                         # a tag before any word has no anchor
                requests[len(tokens) - 1] = tag
        timings.log.update(normalized=' '.join(n for n in normalized if n), tokens=tokens,
                           requests=requests)

        sequences, log_tokens, log_phones = [], [], []
        for start, stop in self.split_sentences(tokens):
            local = {i - start: k for i, k in requests.items() if start <= i < stop}
            with_pauses = timed('pauses', self.insert_pauses, tokens[start:stop], local)
            words = timed('g2p', self._word_phones, with_pauses, timings)
            log_tokens.append(with_pauses)
            pieces = self.chunk(words)
            log_phones += pieces
            sequences += pieces
        timings.log.update(sentences=log_tokens, phones=log_phones)
        if len(sequences) > len(log_tokens):
            timings.warn(f'{len(log_tokens)} sentence(s) synthesized in {len(sequences)} '
                         f'pieces of at most {self.max_phones} phones')
        if not sequences:
            timings.warn('nothing to synthesize after normalization')
        return sequences

    # ------------------------------------------------------------- back end
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
        """Mel `[n_mels, T]` to waveform at 22 050 Hz, exactly `T * 256` samples."""
        import torch
        with torch.no_grad():
            wav = self.vocoder.decode_batch(torch.tensor(mel).to(self.device))
        return trim_vocoder_padding(self.vocoder, wav.squeeze().cpu().numpy())

    def _sync(self) -> None:
        if self.device != 'cpu':
            import torch
            torch.cuda.synchronize()

    # ----------------------------------------------------------------------- the whole
    def __call__(self, text: str, markup: bool = False, length_scale: float = 1.0,
                 keep_mels: bool = False) -> tuple[np.ndarray, Timings]:
        """Synthesize one input, timing every stage.

        Input that leaves nothing to say (empty, only symbols, an unsupported script)
        returns an empty waveform and a warning in `timings.log` rather than raising: a
        service that crashes on a blank request is not a service.

        Args:
            text: The input, any length: it is split into sentences.
            markup: Parse `<pause type=...>` tags (extra task 1).
            length_scale: Tempo: > 1 slower, < 1 faster (scales predicted durations).
            keep_mels: Store the mel of every piece in `timings.log['mels']`.
        """
        if self.model is None:
            raise RuntimeError('front end only: create TTSPipeline with a run_dir to synthesize')
        timings = Timings()
        sequences = self.front_end(text, markup, timings)
        self.model.length_scale = float(length_scale)
        wavs, mels = [], []
        for phones in sequences:
            start = time.perf_counter()
            mel = self.mel(phones)
            self._sync()
            timings.add('acoustic', time.perf_counter() - start)
            start = time.perf_counter()
            wavs.append(self.vocode(mel))
            self._sync()
            timings.add('vocoder', time.perf_counter() - start)
            if keep_mels:
                mels.append(mel)
        if keep_mels:
            timings.log['mels'] = mels
        wav = np.concatenate(wavs).astype(np.float32) if wavs else np.zeros(0, np.float32)
        timings.audio_seconds = len(wav) / SAMPLE_RATE
        return wav, timings

    def save(self, wav: np.ndarray, path: str | Path) -> None:
        """Write a waveform with the AudioProcessor, so the sample rate matches the rest."""
        self.ap.save_wav(wav, str(path), sr=SAMPLE_RATE)
