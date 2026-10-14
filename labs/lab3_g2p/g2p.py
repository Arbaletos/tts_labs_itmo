"""Grapheme-to-phoneme front end — skeleton for lab 3.

Run as a script to score the module on the prepared data::

    python g2p.py

Two phone error rates are reported on the split carried in the data's own `set` column: PER over
whole `test` utterances — the front end as lab 5 will run it — and PER over `test` tokens
absent from the training lexicon, which is the out-of-vocabulary model alone.
"""
import csv
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import tqdm


TYPOGRAPHIC_HYPHENS = '\u2010\u2011\u2012\u2013\u2014\u2212'
TOKENS_PHONES_DATA = 'data/tokens_phones.csv'
MFA_DICT_PATH = 'data/mfa_dict.csv'
SIL = '<SIL>'


GRAPHEME_TO_PHONEME = {
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'ɡ', 'д': 'd̪',
    'е': 'e', 'ё': 'o', 'ж': 'ʐ', 'з': 'z̪', 'и': 'i',
    'й': 'j', 'к': 'k', 'л': 'ɫ', 'м': 'm', 'н': 'n̪',
    'о': 'o', 'п': 'p', 'р': 'r', 'с': 's̪', 'т': 't̪',
    'у': 'u', 'ф': 'f', 'х': 'x', 'ц': 't̪s̪', 'ч': 'tɕ',
    'ш': 'ʂ', 'щ': 'ɕ', 'ъ': '', 'ы': 'ɨ', 'ь': '',
    'э': 'ɛ', 'ю': 'j u', 'я': 'j a',
}


SOFT_PAIRS = {
    'б': 'bʲ',
    'в': 'vʲ',
    'г': 'ɟ',
    'д': 'dʲ',
    'з': 'zʲ',
    'к': 'c',
    'л': 'ʎ',
    'м': 'mʲ',
    'н': 'ɲ',
    'п': 'pʲ',
    'р': 'rʲ',
    'с': 'sʲ',
    'т': 'tʲ',
    'ф': 'fʲ',
    'х': 'ç',
}


REDUCTION_FIRST = {
    'а': 'ɐ', 'о': 'ɐ', 'е': 'ɪ', 'и': 'ɪ',
    'у': 'ʊ', 'ы': 'ɨ', 'э': 'ɪ',
}
REDUCTION_LAST = {
    'а': 'ə', 'о': 'ə', 'е': 'ɪ', 'и': 'ɪ',
    'у': 'ʊ', 'ы': 'ɨ', 'э': 'ɪ',
}


DEVOICING_FINAL = {
    'б': 'p', 'в': 'f', 'г': 'k', 'д': 't̪',
    'ж': 'ʂ', 'з': 's̪',
}

DEVOICING_BEFORE = {
    'б': 'p', 'в': 'f', 'г': 'k', 'д': 't̪',
    'ж': 'ʂ', 'з': 's̪',
}

VOICELESS = set('кпстфхцчшщ')


SPECIAL_COMBOS = {
    'дж': ['dʐ'],
    'тч': ['tɕ'],
    'дч': ['tɕ'],
    'сч': ['ɕ'],
    'зч': ['ɕ'],
    'сж': ['ʐ'],
    'зж': ['ʐ'],
    'вств': ['s̪', 't̪', 'v'],
    'стн': ['s̪', 'n̪'],
    'здн': ['z̪', 'n̪'],
    'лнц': ['n̪', 't̪s̪'],
    'рдц': ['r', 't̪s̪'],
}


VOWEL_CHARS = set('аеёиоюяуыэ')
SOFTENING_VOWELS = set('яюеиёь')
HUSHING = set('шжц')


def normalize_token(token: str) -> str:
    """Канонизация токена: типографские дефисы → ASCII, нижний регистр."""
    for h in TYPOGRAPHIC_HYPHENS:
        token = token.replace(h, '-')
    return token.lower().strip()


def collapse_doubles(word: str) -> str:
    """Схлопывает двойные согласные: 'нн' → 'н', 'сс' → 'с'."""
    result = []
    prev = None
    for ch in word:
        if ch == prev and ch not in VOWEL_CHARS:
            continue
        result.append(ch)
        prev = ch
    return ''.join(result)


def load_mfa_dict(path: str = MFA_DICT_PATH) -> dict[str, str]:
    """Читает MFA-словарь (берём только rank=0 — самое вероятное произношение).

    Returns:
        {word: phones_string} — для каждого слова его основное произношение.
    """
    mfa = {}
    with open(path, encoding='utf-8') as fh:
        reader = csv.DictReader(fh, delimiter='|', quoting=csv.QUOTE_NONE)
        for row in reader:
            if row['rank'] == '0':
                mfa[row['word']] = row['phones']
    return mfa


class G2P:
    """Turns a sentence into a phone sequence.

    Input is one sentence as the token sequence lab 2 produces: words without punctuation,
    with ``"<SIL>"`` wherever the pause predictor placed a pause::

        ["я", "вышел", "из", "дома", <SIL>, "когда", "стемнело"]

    Punctuation is gone by this point on purpose — its only role in the pipeline is to
    predict pauses, which lab 2 has already done.

    Args:
        lexicon: `word -> Counter({phones: count})` as written by `prepare_lexicon.py`.
            Pronunciations are space-separated NFC strings. Keys are lowercase and carry no
            stress mark, so look words up accordingly.
        mfa_dict: `word -> phones_string` — внешний словарь MFA для fallback.
    """

    def __init__(self, lexicon: dict[str, Counter] | None = None,
                 mfa_dict: dict[str, str] | None = None) -> None:
        self.lexicon = lexicon or {}
        self.mfa_dict = mfa_dict or {}

    def _lookup(self, word: str) -> list[str] | None:
        """Ищет слово в lexicon, возвращает самое частое произношение."""
        counter = self.lexicon.get(word)
        if not counter:
            return None
        best_phones, _ = counter.most_common(1)[0]
        return best_phones.split()

    def _oov(self, word: str) -> list[str]:
        """OOV: главный метод. Проверяет специальные окончания."""
        if word.endswith('ться'):
            return self._oov_base(word[:-4]) + ['t̪', 's̪', 'ə']
        if word.endswith('тся'):
            return self._oov_base(word[:-3]) + ['t̪', 's̪', 'ə']
        if word.endswith('ее'):
            return self._oov_base(word[:-2]) + ['e', 'j', 'e']
        if word.endswith('ие'):
            return self._oov_base(word[:-2]) + ['ɪ', 'j', 'e']
        if word.endswith('ия'):
            return self._oov_base(word[:-2]) + ['ɪ', 'j', 'ə']
        if word.endswith('ого') or word.endswith('его'):
            return self._oov_ogo(word)
        return self._oov_base(word)

    def _oov_ogo(self, word: str) -> list[str]:
        """OOV для слов на '-ого' / '-его' (прилагательные, местоимения)."""
        base = word[:-3]
        return self._oov_base(base) + ['ə', 'v', 'ə']

    def _guess_stress(self, word: str, vowel_positions: list[int]) -> int:
        """Эвристика определения ударного слога (индекс гласной).

        Правила:
        1. Если есть 'ё' — ударение на ней (всегда ударная).
        2. Иначе — проверяем типичные окончания.
        3. Иначе — предпоследний слог (для слов >= 3 слогов), последний для 2.
        """
        n_vowels = len(vowel_positions)
        if n_vowels == 0:
            return -1
        if n_vowels == 1:
            return 0

        if 'ё' in word:
            yo_pos = word.index('ё')
            for idx, vpos in enumerate(vowel_positions):
                if vpos == yo_pos:
                    return idx

        endings = {
            'ение': 2, 'ание': 2, 'ация': 2, 'иция': 2, 'уция': 2,
            'ость': 2, 'есть': 2, 'изм': 2, 'ист': 2,
            'ический': 3, 'ическая': 3, 'ическое': 3,
            'овать': 2, 'евать': 2, 'ировать': 3,
            'ать': 2, 'ять': 2, 'еть': 2, 'ить': 2, 'ыть': 2, 'уть': 2,
            'ка': 2, 'ец': 2, 'ик': 2, 'ок': 2, 'ек': 2,
            'ая': 2, 'ое': 2, 'ые': 2, 'ый': 2, 'ой': 2, 'ей': 2,
        }
        for ending, from_end in endings.items():
            if word.endswith(ending) and n_vowels >= from_end:
                return n_vowels - from_end

        if n_vowels == 2:
            return 1
        return n_vowels - 2

    def _oov_base(self, word: str) -> list[str]:
        """Основная логика OOV: палатализация, редукция, оглушение."""
        word = collapse_doubles(word)

        vowel_positions = [i for i, ch in enumerate(word) if ch in VOWEL_CHARS]
        stressed_idx = self._guess_stress(word, vowel_positions)

        result = []
        i = 0
        vowel_counter = 0
        while i < len(word):
            ch = word[i]
            next_ch = word[i + 1] if i + 1 < len(word) else ''
            prev_ch = word[i - 1] if i > 0 else ''

            # Специальные сочетания
            matched = False
            for length in (4, 3, 2):
                if i + length <= len(word):
                    combo = word[i:i + length]
                    if combo in SPECIAL_COMBOS:
                        result.extend(SPECIAL_COMBOS[combo])
                        i += length
                        matched = True
                        break
            if matched:
                continue

            # Йотированные гласные
            if ch in 'яеюё':
                is_stressed = (vowel_counter == stressed_idx)
                is_before_stressed = (vowel_counter == stressed_idx - 1)

                if prev_ch and prev_ch not in VOWEL_CHARS and prev_ch not in 'ьъ':
                    if is_stressed:
                        v_map = {'я': 'a', 'е': 'e', 'ю': 'u', 'ё': 'o'}
                        result.append(v_map[ch])
                    elif is_before_stressed:
                        r_map = {'я': 'ɪ', 'е': 'ɪ', 'ю': 'ʊ', 'ё': 'ɐ'}
                        result.append(r_map[ch])
                    else:
                        r_map = {'я': 'ɪ', 'е': 'ɪ', 'ю': 'ʊ', 'ё': 'ə'}
                        result.append(r_map[ch])
                else:
                    if is_stressed:
                        j_map = {'я': 'j a', 'е': 'j e', 'ю': 'j u', 'ё': 'j o'}
                        result.extend(j_map[ch].split())
                    else:
                        j_map = {'я': 'j ɪ', 'е': 'j ɪ', 'ю': 'j ʊ', 'ё': 'j ə'}
                        result.extend(j_map[ch].split())
                vowel_counter += 1
                i += 1
                continue

            # Гласные с редукцией
            if ch in VOWEL_CHARS:
                if ch in 'еи' and prev_ch in HUSHING:
                    result.append('ɨ')
                    vowel_counter += 1
                    i += 1
                    continue
                if ch == 'о' and prev_ch in HUSHING:
                    result.append('o')
                    vowel_counter += 1
                    i += 1
                    continue

                if vowel_counter == stressed_idx:
                    phones = GRAPHEME_TO_PHONEME.get(ch, '')
                    if phones:
                        result.extend(phones.split())
                elif vowel_counter == stressed_idx - 1:
                    if ch in REDUCTION_FIRST:
                        result.append(REDUCTION_FIRST[ch])
                    else:
                        phones = GRAPHEME_TO_PHONEME.get(ch, '')
                        if phones:
                            result.extend(phones.split())
                else:
                    if ch in REDUCTION_LAST:
                        result.append(REDUCTION_LAST[ch])
                    else:
                        phones = GRAPHEME_TO_PHONEME.get(ch, '')
                        if phones:
                            result.extend(phones.split())
                vowel_counter += 1
                i += 1
                continue

            # Согласные
            if ch in SOFT_PAIRS or ch in GRAPHEME_TO_PHONEME:
                if ch in SOFT_PAIRS and next_ch in SOFTENING_VOWELS:
                    base_phone = SOFT_PAIRS[ch]
                else:
                    base_phone = GRAPHEME_TO_PHONEME.get(ch, '')

                is_final = (i == len(word) - 1) or (next_ch == 'ь' and i + 2 == len(word))
                if is_final and ch in DEVOICING_FINAL:
                    base_phone = DEVOICING_FINAL[ch]
                elif next_ch in VOICELESS and ch in DEVOICING_BEFORE:
                    base_phone = DEVOICING_BEFORE[ch]

                if base_phone:
                    result.extend(base_phone.split())
                i += 1
                continue

            phones = GRAPHEME_TO_PHONEME.get(ch, '')
            if phones:
                result.extend(phones.split())
            i += 1

        return result

    def transcribe(self, tokens: list[str] | np.ndarray) -> list[list[str]]:
        """Transcribe each token of one sentence.

        Args:
            tokens: Tokens of one sentence in order, possibly including ``"<SIL>"``.

        Returns:
            One phone list per input token, so that `len(result) == len(tokens)`.
            ``"<SIL>"`` maps to ``["<SIL>"]``.
        """
        result = []
        for token in tokens:
            if token == SIL:
                result.append([SIL])
                continue
            word = normalize_token(str(token))
            # 1. Основной словарь (train)
            phones = self._lookup(word)
            # 2. MFA-словарь (fallback)
            if phones is None and word in self.mfa_dict:
                phones = self.mfa_dict[word].split()
            # 3. OOV-эвристики
            if phones is None:
                phones = self._oov(word)
            result.append(phones)
        return result


def phone_error_rate(reference: list[list[str]], hypothesis: list[list[str]]) -> float:
    """Corpus-level PER: summed edit distance over summed reference length."""
    total_distance = 0
    total_length = 0
    for ref, hyp in zip(reference, hypothesis):
        total_length += len(ref)
        if not hyp:
            total_distance += len(ref)
            continue
        previous = list(range(len(hyp) + 1))
        for i, ref_phone in enumerate(ref, start=1):
            current = [i] + [0] * len(hyp)
            for j, hyp_phone in enumerate(hyp, start=1):
                current[j] = min(previous[j] + 1,
                                 current[j - 1] + 1,
                                 previous[j - 1] + (ref_phone != hyp_phone))
            previous = current
        total_distance += previous[-1]
    return total_distance / total_length if total_length else float('nan')


def word_accuracy(reference: list[list[str]], hypothesis: list[list[str]]) -> float:
    """Fraction of words whose whole phone sequence is exactly right."""
    if not reference:
        return float('nan')
    return float(np.mean([ref == hyp for ref, hyp in zip(reference, hypothesis)]))


def load_tokens(path: str = TOKENS_PHONES_DATA) -> pd.DataFrame:
    """Read the training table and drop tokens with no reference transcription."""
    tokens = pd.read_csv(path, sep='|', quoting=csv.QUOTE_NONE)
    tokens['token'] = tokens.token.astype(str).map(lambda s: unicodedata.normalize('NFC', s))
    tokens['phones'] = tokens.phones.fillna('').astype(str)

    unpronounced = (tokens.phones == '') | tokens.phones.str.split().map(lambda p: 'spn' in p)
    print(f'dropping {unpronounced.sum()} tokens without a transcription '
          f'({unpronounced.mean():.2%})')
    return tokens[~unpronounced].reset_index(drop=True)


def build_lexicon(tokens: pd.DataFrame) -> dict[str, Counter]:
    """Count pronunciations per token type over the given rows."""
    lexicon: dict[str, Counter] = defaultdict(Counter)
    for token, phones in tokens[['token', 'phones']].values:
        lexicon[token][phones] += 1
    return lexicon


def calc_metrics(g2p: G2P, tokens: pd.DataFrame, lexicon: dict[str, Counter]) -> None:
    """Print running-text PER, and PER and word accuracy on OOV tokens."""
    sentence_ref: list[list[str]] = []
    sentence_hyp: list[list[str]] = []
    oov_ref: list[list[str]] = []
    oov_hyp: list[list[str]] = []
    in_vocab_ref: list[list[str]] = []
    in_vocab_hyp: list[list[str]] = []

    for _, sentence in tqdm.tqdm(tokens.groupby('id', sort=False)):
        sentence = sentence.sort_values('token_id')
        words = sentence.token.values
        predicted = g2p.transcribe(words)
        if len(predicted) != len(words):
            raise ValueError(f'transcribe returned {len(predicted)} outputs '
                             f'for {len(words)} tokens')
        reference = [p.split() for p in sentence.phones.values]

        sentence_ref.append([p for word in reference for p in word if p != SIL])
        sentence_hyp.append([p for word in predicted for p in word if p != SIL])
        for token, ref, hyp in zip(words, reference, predicted):
            if token in lexicon:
                in_vocab_ref.append(ref)
                in_vocab_hyp.append(hyp)
            else:
                oov_ref.append(ref)
                oov_hyp.append(hyp)

    n_words = len(in_vocab_ref) + len(oov_ref)
    print(f'running text  PER {phone_error_rate(sentence_ref, sentence_hyp):.4f}')
    print(f'in-vocabulary PER {phone_error_rate(in_vocab_ref, in_vocab_hyp):.4f}  '
          f'({len(in_vocab_ref)} words, {len(in_vocab_ref) / n_words:.1%})')
    print(f'OOV           PER {phone_error_rate(oov_ref, oov_hyp):.4f}  '
          f'word accuracy {word_accuracy(oov_ref, oov_hyp):.4f}  '
          f'({len(oov_ref)} words, {len(oov_ref) / n_words:.1%})')


def test_g2p() -> None:
    """Build the lexicon from `train` and score the module on both folds."""
    tokens = load_tokens()
    train = tokens[tokens.set == 'train']
    test = tokens[tokens.set == 'test']
    print(f'train {len(train)} tokens / test {len(test)} tokens')

    lexicon = build_lexicon(train)
    print(f'lexicon: {len(lexicon)} types, '
          f'{sum(1 for c in lexicon.values() if len(c) > 1)} with several pronunciations')

    mfa_dict = load_mfa_dict(MFA_DICT_PATH)
    print(f'MFA dict: {len(mfa_dict)} words')

    g2p = G2P(lexicon, mfa_dict)

    print('\nCalculate metrics, training fold:')
    calc_metrics(g2p, train, lexicon)

    print('\nCalculate metrics, testing fold:')
    calc_metrics(g2p, test, lexicon)


if __name__ == '__main__':
    test_g2p()