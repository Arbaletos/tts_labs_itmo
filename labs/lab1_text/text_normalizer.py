"""Russian text normalizer for lab 1.

Brings corpus text into a form usable for training a speech synthesizer.

    "!.."           -> "!"
    "«цитата»"      -> '"цитата"'
    "текст * мусор" -> "текст мусор"
    "де‑факто"      -> "де-факто"      # U+2011 -> ordinary hyphen

**Word-changing edits.** The alignment for that utterance becomes invalid and the
row must be dropped from the training set — but the logic itself is still needed
for lab 5, where arbitrary user input arrives with no alignment at all::

    "в 1995 г."     -> "в тысяча девятьсот девяносто пятом году"
    "прим. автора"  -> "примечание автора"

Because of that, ``normalize()`` only ever performs *safe*, word-preserving edits
(punctuation, quotes, dashes, stray symbols, whitespace, NFC). Word-changing edits
(number-to-words, abbreviation expansion, anglicism transliteration) live behind
``expand=True`` and are meant for free-form text (lab 5), not for the corpus
preprocessing pipeline: a RUSLAN row that needs a word-changing edit should instead
be dropped by :class:`~text_filter.TextFilter`, which sees the un-expanded text and
still finds the digit/abbreviation/foreign word that gives it away.

Example:
    >>> normalizer = TextNormalizer()
    >>> normalizer.normalize("Расстреливать надо таких писателей!..")
    'Расстреливать надо таких писателей!'
"""

import re
import unicodedata

# Non-breaking / figure / other hyphen look-alikes used as a word-joiner -> ASCII "-".
_HYPHEN_LIKE = "‐‑‒―"
# En dash and horizontal bar used as a stand-alone "тире" -> canonical em dash.
_DASH_LIKE = "–—"

_INVISIBLE = "​‌‍⁠﻿­"

_DOUBLE_QUOTES = "«»„“”‟"
_SINGLE_QUOTES = "‘’ʼ`‹›"

# Anything outside this set is noise (markup, emoji, stray technical symbols) and
# is simply dropped. ́ is the combining acute accent used for stress marks.
_ALLOWED_CHARS = re.compile(
    r"[^а-яА-ЯёЁ́A-Za-z0-9\s.,!?;:\"'\-—…]"
)

_TRANSLATE_HYPHEN = {ord(c): "-" for c in _HYPHEN_LIKE}
_TRANSLATE_DASH = {ord(c): "—" for c in _DASH_LIKE}
_TRANSLATE_QUOTES = {ord(c): '"' for c in _DOUBLE_QUOTES}
_TRANSLATE_QUOTES.update({ord(c): "'" for c in _SINGLE_QUOTES})
_TRANSLATE_INVISIBLE = {ord(c): None for c in _INVISIBLE}


# ---------------------------------------------------------------------------
# Abbreviation / anglicism dictionaries for the optional word-changing pass.
# ---------------------------------------------------------------------------

_ABBREVIATIONS = {
    "ул.": "улица",
    "д.": "дом",
    "просп.": "проспект",
    "пр-т.": "проспект",
    "р-н.": "район",
    "обл.": "область",
    "т.д.": "так далее",
    "т.п.": "тому подобное",
    "прим.": "примечание",
    "тов.": "товарищ",
    "им.": "имени",
    "и.о.": "исполняющий обязанности",
    "г-н": "господин",
    "г-жа": "госпожа",
    "г-жи": "госпожи",
}

_ANGLICISMS = {
    "wi-fi": "вайфай",
    "wifi": "вайфай",
    "iphone": "айфон",
    "android": "андроид",
    "huawei": "хуавей",
    "samsung": "самсунг",
    "macbook": "макбук",
    "usb": "юсби",
    "nfc": "нэфси",
    "email": "имейл",
    "e-mail": "имейл",
}

_ONES = ["", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
_TEENS = [
    "десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать",
    "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать",
]
_TENS = [
    "", "", "двадцать", "тридцать", "сорок", "пятьдесят",
    "шестьдесят", "семьдесят", "восемьдесят", "девяносто",
]
_HUNDREDS = [
    "", "сто", "двести", "триста", "четыреста", "пятьсот",
    "шестьсот", "семьсот", "восемьсот", "девятьсот",
]
_ONES_FEM = dict(enumerate(_ONES))
_ONES_FEM[1] = "одна"
_ONES_FEM[2] = "две"
# (scale word, feminine ones) triples, largest first.
_SCALES = [
    (10 ** 9, ("миллиард", "миллиарда", "миллиардов"), False),
    (10 ** 6, ("миллион", "миллиона", "миллионов"), False),
    (10 ** 3, ("тысяча", "тысячи", "тысяч"), True),
]


def _plural_scale(n: int, forms: tuple) -> str:
    """Pick the right declension of a scale word (тысяча/тысячи/тысяч) for n."""
    n = n % 100
    if 11 <= n <= 19:
        return forms[2]
    n = n % 10
    if n == 1:
        return forms[0]
    if 2 <= n <= 4:
        return forms[1]
    return forms[2]


def _cardinal_under_1000(n: int, feminine: bool = False) -> str:
    parts = []
    if n >= 100:
        parts.append(_HUNDREDS[n // 100])
        n %= 100
    if n >= 20:
        parts.append(_TENS[n // 10])
        n %= 10
    elif n >= 10:
        parts.append(_TEENS[n - 10])
        n = 0
    if n > 0:
        parts.append((_ONES_FEM if feminine else {i: w for i, w in enumerate(_ONES)})[n])
    return " ".join(parts)


def cardinal(n: int) -> str:
    """Spell out an integer in (masculine, nominative case) Russian words.

    Example:
        >>> cardinal(1995)
        'тысяча девятьсот девяносто пять'
    """
    if n == 0:
        return "ноль"
    sign = "минус " if n < 0 else ""
    n = abs(n)
    chunks = []
    for scale, forms, feminine in _SCALES:
        if n >= scale:
            count = n // scale
            n %= scale
            # "тысяча девятьсот..." not "одна тысяча девятьсот...": the
            # multiplier is dropped for a bare "one thousand", same as "a
            # thousand" in English. Kept for "один миллион"/"одна тысяча X".
            count_word = "" if (count == 1 and scale == 1000) else _cardinal_under_1000(count, feminine)
            chunks.append(f"{count_word} {_plural_scale(count, forms)}".strip())
    if n > 0 or not chunks:
        chunks.append(_cardinal_under_1000(n))
    return sign + " ".join(chunks)


# Genitive-masculine/neuter ordinal endings, enough for dates ("восьмого марта",
# "двадцать первого века"). Nominative-only cardinals feed this table, so
# coverage is limited to whole tens/hundreds and 1-9 — sufficient for years,
# days-of-month and centuries, not a full declension engine.
_ORD_ONES_GEN = [
    "", "первого", "второго", "третьего", "четвёртого", "пятого",
    "шестого", "седьмого", "восьмого", "девятого",
]
_ORD_TEENS_GEN = [
    "десятого", "одиннадцатого", "двенадцатого", "тринадцатого", "четырнадцатого",
    "пятнадцатого", "шестнадцатого", "семнадцатого", "восемнадцатого", "девятнадцатого",
]
_ORD_TENS_GEN = [
    "", "", "двадцатого", "тридцатого", "сорокового", "пятидесятого",
    "шестидесятого", "семидесятого", "восьмидесятого", "девяностого",
]
_ORD_HUNDREDS_GEN = [
    "", "сотого", "двухсотого", "трёхсотого", "четырёхсотого", "пятисотого",
    "шестисотого", "семисотого", "восьмисотого", "девятисотого",
]


def ordinal_genitive(n: int) -> str:
    """Spell out an ordinal in genitive case, e.g. for "восьмого марта".

    Only the last two-or-three digits carry an ordinal ending; leading
    thousands/hundreds are read as plain cardinals, matching how dates and
    century numbers are actually read aloud.

    Example:
        >>> ordinal_genitive(21)
        'двадцать первого'
    """
    if n <= 0:
        return str(n)
    hundreds, rest = divmod(n, 100)
    prefix = _cardinal_under_1000(hundreds * 100) if hundreds and rest else ""
    if rest == 0:
        return _ORD_HUNDREDS_GEN[hundreds]
    if rest < 10:
        tail = _ORD_ONES_GEN[rest]
    elif rest < 20:
        tail = _ORD_TEENS_GEN[rest - 10]
    else:
        tens, ones = divmod(rest, 10)
        tail = _ORD_TENS_GEN[tens] if ones == 0 else f"{_TENS[tens]} {_ORD_ONES_GEN[ones]}"
    return f"{prefix} {tail}".strip()


class TextNormalizer:
    """Normalizes text in Russian.

    ``normalize()`` performs only safe, word-preserving edits by default, so it
    is safe to run on RUSLAN metadata before filtering. Pass ``expand=True`` to
    additionally spell out numbers, expand common abbreviations and transliterate
    anglicisms — useful for free-form input (lab 5), not for the corpus.

    Example:
        >>> textfilter = TextNormalizer()
        >>> textfilter.normalize("Расстреливать надо таких писателей!.")
        'Расстреливать надо таких писателей!'
    """

    def __init__(self):
        abbr_pattern = "|".join(re.escape(a) for a in sorted(_ABBREVIATIONS, key=len, reverse=True))
        self._re_abbr = re.compile(rf"(?<![а-яё]){abbr_pattern}(?![а-яё])", re.IGNORECASE)

        angl_pattern = "|".join(re.escape(a) for a in sorted(_ANGLICISMS, key=len, reverse=True))
        self._re_angl = re.compile(rf"\b(?:{angl_pattern})\b", re.IGNORECASE)

        self._re_number = re.compile(r"-?\d+")
        # "21-го", "восьмого-му" style genitive/dative ordinals (dates, centuries)
        # get a grammatically correct reading. Other ordinal suffixes ("-й",
        # "-я", "-е", ...) are a full declension problem outside this bonus
        # feature's scope: they are stripped and the bare number is read as a
        # cardinal instead of being silently left dangling.
        self._re_ordinal = re.compile(r"(-?\d+)-(?:го|му)\b")
        self._re_ordinal_suffix_strip = re.compile(r"(-?\d+)-(?:[ыои]?[хй]|ое|ая|ые|ых|е|я)\b")

        self._re_repeat_bangq = re.compile(r"!{2,}|\?{2,}")
        self._re_bang_dots = re.compile(r"([!?])\.{1,2}(?!\.)")
        self._re_two_dots = re.compile(r"(?<!\.)\.\.(?!\.)")
        self._re_many_dots = re.compile(r"\.{3,}")
        self._re_space_before_punct = re.compile(r"[ \t]+([,.!?;:])")
        self._re_missing_space_after = re.compile(r"([,.!?;:])(?=[А-Яа-яЁё])")
        self._re_whitespace = re.compile(r"[ \t]+")

        # Abbreviations / dialogue markers a naive sentence-splitter must not
        # treat as a sentence boundary just because they end in a dot.
        self._non_boundary_abbr = {
            "ул", "д", "г", "просп", "пр-т", "т.д", "т.п", "прим", "тов",
            "им", "и.о", "гр", "см", "кг", "им", "тыс", "млн", "млрд",
        }
        self._re_sentence_end = re.compile(r"(?<=[.!?…])\s+(?=[А-ЯЁ—\"«])")

    # -- safe, word-preserving normalization --------------------------------

    def normalize(self, text: str, expand: bool = False) -> str:
        """Normalize a single line.

        Args:
            text: Raw utterance text, exactly as stored in the corpus metadata.
            expand: If ``True``, additionally perform word-changing edits
                (numbers, abbreviations, anglicisms). These invalidate the
                audio alignment and must never be used for RUSLAN preprocessing.

        Returns:
            The normalized text. Returning the input unchanged is valid and
            common — most lines need nothing done to them.
        """
        text = unicodedata.normalize("NFC", text)
        text = text.translate(_TRANSLATE_INVISIBLE)
        text = text.translate(_TRANSLATE_HYPHEN)
        text = text.translate(_TRANSLATE_DASH)
        text = text.translate(_TRANSLATE_QUOTES)

        if expand:
            text = self._expand_words(text)

        text = _ALLOWED_CHARS.sub("", text)

        text = self._re_bang_dots.sub(r"\1", text)
        text = self._re_repeat_bangq.sub(lambda m: m.group()[0], text)
        text = self._re_many_dots.sub("…", text)
        text = self._re_two_dots.sub(".", text)

        text = self._re_space_before_punct.sub(r"\1", text)
        text = self._re_missing_space_after.sub(r"\1 ", text)
        text = self._re_whitespace.sub(" ", text)

        text = unicodedata.normalize("NFC", text.strip())
        return text

    # -- word-changing normalization (bonus: extended normalizer) -----------

    def _expand_words(self, text: str) -> str:
        text = self._re_ordinal.sub(lambda m: ordinal_genitive(int(m.group(1))), text)
        text = self._re_ordinal_suffix_strip.sub(r"\1", text)
        text = self._re_number.sub(lambda m: cardinal(int(m.group())), text)
        text = self._re_abbr.sub(lambda m: _ABBREVIATIONS[m.group().lower()], text)
        text = self._re_angl.sub(lambda m: _ANGLICISMS[m.group().lower()], text)
        return text

    # -- bonus: sentence boundary detection ----------------------------------

    def split_sentences(self, text: str) -> list:
        """Split a block of text into sentences.

        Meant for free-form input (e.g. a full paragraph fed to lab 5), where
        the synthesizer needs sentence-sized chunks. Splits on ``.``, ``!``,
        ``?`` or ``…`` followed by whitespace and a capital letter/dash/quote,
        but not after a token that is a known abbreviation (so "ул. Ленина"
        does not get cut in half) or after a single capital-letter initial
        ("В. Иванов").

        Example:
            >>> TextNormalizer().split_sentences("Я вышел. Ты остался.")
            ['Я вышел.', 'Ты остался.']
        """
        text = self.normalize(text)
        candidates = list(self._re_sentence_end.finditer(text))
        sentences = []
        start = 0
        for m in candidates:
            preceding = text[start:m.start()]
            last_word = re.findall(r"[А-Яа-яЁё.]+$", preceding.rstrip("!?…."))
            token = last_word[0].lower() if last_word else ""
            if token in self._non_boundary_abbr:
                continue
            if re.fullmatch(r"[а-яё]", token):
                continue
            sentences.append(text[start:m.start()].strip())
            start = m.start()
        tail = text[start:].strip()
        if tail:
            sentences.append(tail)
        return sentences
