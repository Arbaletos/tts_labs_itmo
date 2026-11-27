"""Text preparation rules the lab 1-3 modules do not cover — lab 5.

Labs 1-3 were built for RUSLAN: Cyrillic literary text with ordinary punctuation. Free input
breaks that assumption in a few predictable ways, and each one fails *silently* further down:

`икону+наши`, `ТВ/спазм`, `=кровь` | lab 1 deletes `+ / =` without a space, two words glue into one | :func:`symbols_to_spaces` 
`з+амок` (lab 3 stress control) | lab 1 deletes `+`, the stress request is lost | :func:`plus_to_acute` 
`люди -- передохнут` | `--` is not a dash for lab 1/2, the pause feature `dash` stays 0 | :func:`ascii_dashes` 
`fabric`, `details` | lab 3 keeps only Cyrillic letters, the word gets no phones at all | :func:`transliterate_latin` 
`ТВ`, `МГУ`, `К.` | read as a (vowelless) word | :func:`spell_abbreviations` |
`info@site.ru`, `https://…` | `@ /` deleted, `site.ru` cut into a sentence end + `ru` | :func:`speak_urls` 
`Αθήνα`, `東京`, emoji | deleted without a trace | reported by :func:`dropped_characters` 

Everything here works on raw text *before* the lab 1 normalizer and only ever replaces
characters or whole words, so the order of words is kept.
"""
import re
import unicodedata
from typing import Callable

ACUTE = '́'
CYR_VOWELS = 'аеёиоуыэюяАЕЁИОУЫЭЮЯ'

# Characters the lab 1 normalizer keeps or translates itself (see text_normalizer.py).
_KEPT = re.compile(r"[а-яА-ЯёЁA-Za-z0-9\s.,!?;:\"'\-—…" + ACUTE + r"]")
_TRANSLATED = set('‐‑‒―–—«»„“”‟‘’ʼ`‹›') | set('​‌‍⁠﻿­')


def plus_to_acute(text: str) -> str:
    """`з+амок` -> `за́мок`: a `+` right before a Cyrillic vowel inside a word is a stress mark.

    Lab 3 accepts both `+` and the combining acute, but lab 1 keeps only the acute, so the `+`
    form has to be converted before normalization. Any other `+` is left for
    :func:`symbols_to_spaces`.
    """
    return re.sub(rf'(?<=[а-яёА-ЯЁ])\+([{CYR_VOWELS}])|(?<![\w+])\+([{CYR_VOWELS}])(?=[а-яё])',
                  lambda m: (m.group(1) or m.group(2)) + ACUTE, text)


def symbols_to_spaces(text: str) -> str:
    """Replace every character lab 1 would *delete* with a space.

    Lab 1 drops markup, emoji and technical symbols in place: `ТВ/спазм` becomes `ТВспазм`.
    A space keeps the words apart; lab 1 collapses repeated spaces afterwards.
    """
    return ''.join(c if _KEPT.match(c) or c in _TRANSLATED else ' ' for c in text)


def ascii_dashes(text: str) -> str:
    """` -- ` and ` - ` between words -> ` — ` (a dash), as the typewriter convention means."""
    return re.sub(r'(?<=\S)\s+-{1,2}\s+(?=\S)', ' — ', text)


def dropped_characters(text: str) -> list[str]:
    """Letters of other scripts and pictographs (emoji) that :func:`symbols_to_spaces` removes.

    Punctuation and technical symbols (`/ = + *`) are not reported: they carry no words. A
    Greek or Chinese word, an emoji does — the user should know it was not pronounced.
    """
    out = []
    for c in dict.fromkeys(text):
        if _KEPT.match(c) or c in _TRANSLATED or c == '+':
            continue
        if unicodedata.category(c)[0] in 'LN' or unicodedata.category(c) == 'So':
            out.append(c)
    return out


# ---------------------------------------------------------------------------- URLs, e-mail
_URL_RE = re.compile(r'(?:https?://|www\.)[^\s<>"]+|[\w.+-]+@[\w-]+(?:\.[\w-]+)+', re.IGNORECASE)
_URL_WORDS = {'www': 'вэ-вэ-вэ', 'ru': 'ру', 'com': 'ком', 'org': 'орг', 'net': 'нэт',
              'io': 'ай-о', 'рф': 'эр-эф', 'info': 'инфо', 'html': 'эйч-ти-эм-эль'}
_URL_PARTS = {'.': ' точка ', '@': ' собака ', '/': ' слэш ', '-': ' дефис ', '_': ' подчёркивание ',
              '?': ' вопрос ', '=': ' равно ', '&': ' и ', '+': ' плюс '}


def speak_urls(text: str) -> str:
    """`info@site.ru` -> `info собака site точка ru`: separators read aloud, protocol dropped.

    Without this, lab 1 deletes `@` and `/`, and `site.ru` becomes a sentence end followed by
    a word "ru". The Latin parts are transliterated afterwards like any other Latin word.
    """
    def repl(m: re.Match) -> str:
        url = re.sub(r'^https?://', '', m.group(0), flags=re.IGNORECASE).rstrip('/.,')
        spoken = ''.join(_URL_PARTS.get(c, c) for c in url)
        return ' '.join(_URL_WORDS.get(w.lower(), w) for w in spoken.split())
    return _URL_RE.sub(repl, text)


# ---------------------------------------------------------------------------- Latin script
# Ordered: longer patterns first. A rough English-like reading, enough to make a foreign
# word pronounceable rather than silent; a real system would use a dictionary of loanwords.
_LATIN_RULES = [
    ('tion', 'шн'), ('sion', 'жн'), ('sch', 'ш'), ('tch', 'ч'), ('igh', 'ай'),
    ('sh', 'ш'), ('ch', 'ч'), ('th', 'т'), ('ph', 'ф'), ('ck', 'к'), ('qu', 'кв'), ('wh', 'у'),
    ('ee', 'и'), ('ea', 'и'), ('oo', 'у'), ('ai', 'эй'), ('ay', 'эй'), ('ey', 'эй'), ('ou', 'ау'),
    ('ow', 'оу'), ('oy', 'ой'), ('oi', 'ой'), ('au', 'о'), ('aw', 'о'), ('ew', 'ью'),
    ('ce', 'се'), ('ci', 'си'), ('cy', 'си'), ('ge', 'дже'), ('gi', 'джи'),
    ('a', 'а'), ('b', 'б'), ('c', 'к'), ('d', 'д'), ('e', 'е'), ('f', 'ф'), ('g', 'г'),
    ('h', 'х'), ('i', 'и'), ('j', 'дж'), ('k', 'к'), ('l', 'л'), ('m', 'м'), ('n', 'н'),
    ('o', 'о'), ('p', 'п'), ('q', 'к'), ('r', 'р'), ('s', 'с'), ('t', 'т'), ('u', 'а'),
    ('v', 'в'), ('w', 'в'), ('x', 'кс'), ('y', 'и'), ('z', 'з'),
]


def latin_word_to_cyrillic(word: str) -> str:
    """`details` -> `дитэйлс`-like reading of one Latin word (lowercase output)."""
    w = word.lower()
    if len(w) > 3 and w.endswith('e') and w[-2] not in 'aeiouy':   # silent final e: "make"
        w = w[:-1]
    if w.startswith('y') and len(w) > 1:
        w = 'й' + w[1:]
    out, i = [], 0
    while i < len(w):
        for pat, rep in _LATIN_RULES:
            if w.startswith(pat, i):
                out.append(rep)
                i += len(pat)
                break
        else:
            out.append(w[i])
            i += 1
    return ''.join(out)


def transliterate_latin(text: str, exceptions: dict[str, str] | None = None) -> str:
    """Replace every Latin word by a Cyrillic reading; `exceptions` (lowercase) wins over rules."""
    exceptions = exceptions or {}

    def repl(m: re.Match) -> str:
        word = m.group(0)
        if word.lower() in exceptions:                       # "wi-fi" -> "вайфай" as a whole
            return exceptions[word.lower()]
        return '-'.join(exceptions.get(p.lower()) or latin_word_to_cyrillic(p)
                        for p in word.split('-'))
    return re.sub(r"[A-Za-z]+(?:['-][A-Za-z]+)*", repl, text)


# ---------------------------------------------------------------------------- abbreviations
LETTER_NAMES = {
    'а': 'а', 'б': 'бэ', 'в': 'вэ', 'г': 'гэ', 'д': 'дэ', 'е': 'е', 'ё': 'ё', 'ж': 'жэ',
    'з': 'зэ', 'и': 'и', 'й': 'й', 'к': 'ка', 'л': 'эль', 'м': 'эм', 'н': 'эн', 'о': 'о',
    'п': 'пэ', 'р': 'эр', 'с': 'эс', 'т': 'тэ', 'у': 'у', 'ф': 'эф', 'х': 'ха', 'ц': 'цэ',
    'ч': 'че', 'ш': 'ша', 'щ': 'ща', 'ъ': 'твёрдый знак', 'ы': 'ы', 'ь': 'мягкий знак',
    'э': 'э', 'ю': 'ю', 'я': 'я',
}


def spell(word: str) -> str:
    """`ТВ` -> `тэ-вэ`: letter names joined by hyphens, so lab 2 keeps them one token."""
    return '-'.join(LETTER_NAMES.get(c, c) for c in word.lower())


def spell_abbreviations(text: str, is_word: Callable[[str], bool] = lambda w: False) -> str:
    """Spell out all-caps abbreviations and single-letter initials.

    * an all-caps Cyrillic word of 2-5 letters is spelled unless it has a vowel and
      `is_word(lowercase)` says it is an ordinary word written in capitals (`МИР`, `ВСЕ`);
    * a capital consonant followed by a dot and a lowercase word (`городе К. был`) is an
      initial: it is spelled and the dot, which is not a sentence end, is removed.
    """
    def caps(m: re.Match) -> str:
        word = m.group(0)
        # No vowel means it cannot be read as a word, whatever a dictionary says: the MFA
        # dictionary has `тв` as `t̪ f`.
        readable = any(c in CYR_VOWELS for c in word)
        return word if readable and is_word(word.lower()) else spell(word)
    text = re.sub(r'(?<![\w-])[А-ЯЁ]{2,5}(?![\w-])', caps, text)

    def initial(m: re.Match) -> str:
        return spell(m.group(1)) + m.group(2)
    return re.sub(r'(?<![\w-])([БВГДЖЗКЛМНПРСТФХЦЧШЩ])\.(\s+[а-яё])', initial, text)


def prepare(text: str, is_word: Callable[[str], bool] = lambda w: False,
            latin_exceptions: dict[str, str] | None = None) -> str:
    """All rules above, in the order they must run, on NFC text (before lab 1)."""
    text = unicodedata.normalize('NFC', text)
    text = speak_urls(text)
    text = plus_to_acute(text)
    text = ascii_dashes(text)
    text = symbols_to_spaces(text)
    text = transliterate_latin(text, latin_exceptions)
    text = spell_abbreviations(text, is_word)
    return text
