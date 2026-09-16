"""Normalized / non-normalized classifier for lab 1.

The filter is deliberately rule-based rather than a trained model: with the
categories described in the lab README (digits, foreign script, technical
symbols, unit signs, abbreviations, acronyms, stray punctuation, interjections,
emoji) the classes are separated by fairly crisp surface patterns, and 300 dev
examples are not enough to train a model that generalizes better than the rules
below do — see the lab report for the comparison.

Run as a script to score yourself on the development set::

    python text_filter.py
"""

import csv
import re
import unicodedata

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

DEV_SET_PATH = "data/dev_sentences.csv"

# Abbreviations that end in a dot: "ул.", "т.д." — always followed directly by
# the dot, never a stand-alone word on their own.
_ABBR_DOT = [
    "ул", "д", "просп", "пр-кт", "р-н", "обл", "тов", "им", "прим",
    "стр", "кв", "корп", "т.д", "т.п", "и.о", "co", "г",
]
# Abbreviations that use a hyphen instead: "г-н", "г-жа".
_ABBR_HYPHEN = ["г-н", "г-жа", "г-жи", "г-на", "г-ну", "г-жой", "г-же"]

# Acronyms that read letter-by-letter but are not spelled in all caps, so the
# generic ALL-CAPS-run check below would miss them.
_ACRONYM_WORDS = {"минюст"}

# "ммм", "хаха", "ыых" and the like: phonetic mimicry with no fixed
# letter-by-letter pronunciation (README's own examples), as opposed to
# ordinary interjections like "ну"/"ах" that are read exactly as written.
_INTERJECTIONS = {
    "ммм", "мм", "гмм", "гм", "хм", "хмм", "мда", "ыы", "ыых", "хехе", "хаха",
    "ахаха", "угу", "ага", "эээ", "ааа", "оо", "тсс", "фух", "уф",
}


class TextFilter:
    """Decides whether an utterance is usable as a training example.

    Example:
        >>> textfilter = TextFilter()
        >>> textfilter.filter("Я вышел из дома.")
        1
        >>> textfilter.filter("Александрову Г. П.")
        0
    """

    def __init__(self):
        """Compile every rule once so ``filter`` stays cheap per utterance."""
        abbr_dot = "|".join(re.escape(a) for a in _ABBR_DOT)
        self._re_abbr_dot = re.compile(rf"(?<![а-яё])(?:{abbr_dot})\.", re.IGNORECASE)

        abbr_hyphen = "|".join(re.escape(a) for a in _ABBR_HYPHEN)
        self._re_abbr_hyphen = re.compile(rf"\b(?:{abbr_hyphen})\b", re.IGNORECASE)

        self._re_initial = re.compile(r"\b[А-ЯЁ]\.")
        self._re_acronym = re.compile(r"\b[А-ЯЁ]{2,}\b")

        interj = "|".join(re.escape(w) for w in _INTERJECTIONS)
        self._re_interjection = re.compile(rf"(?:^|[.!?]\s+)(?:{interj})\b", re.IGNORECASE)

        self._re_digit = re.compile(r"\d")
        self._re_latin = re.compile(r"[A-Za-z]")
        # Technical symbols, unit/currency signs (%, °, $, ₽, ...), emoji and
        # markup characters — none of these have a fixed, context-free reading.
        self._re_forbidden_symbol = re.compile(
            r"[%°$€₽£@&<>*/+=~^_{}\[\]\\|#"
            r"\U0001F300-\U0001FAFF☀-➿]"
        )
        self._re_bad_quotes = re.compile(r"[„“”‘’ʼ`]")
        self._re_invisible = re.compile(r"[​‌‍⁠﻿­]")

        self._re_repeat_bang_q = re.compile(r"!{2,}|\?{2,}")
        self._re_two_dots = re.compile(r"(?<!\.)\.\.(?!\.)")
        self._re_bang_dots = re.compile(r"(?<!\?)!\.+")
        self._re_question_dots = re.compile(r"\?\.+")

        self._re_space_before_punct = re.compile(r"\s[,.!?;:]")
        self._re_missing_space_after = re.compile(r"[,!?;:](?=[А-Яа-яЁё])")
        self._re_emoticon = re.compile(r"[:;]-?[)(DPp]")

    def filter(self, text: str) -> int:
        """Classify a single utterance.

        Args:
            text: Utterance text, already passed through :class:`TextNormalizer`.

        Returns:
            ``1`` if the text is normalized and the utterance can be used for
            training;
            ``0`` if it contains something the speaker pronounced
            differently from how it is written, and the utterance should be dropped.
        """
        text = unicodedata.normalize("NFC", text)
        lower = text.lower()

        checks = (
            self._re_digit.search(text),
            self._re_latin.search(text),
            self._re_forbidden_symbol.search(text),
            self._re_bad_quotes.search(text),
            self._re_invisible.search(text),
            text.count("(") != text.count(")"),
            "((" in text or "))" in text,
            text.count("«") != text.count("»"),
            self._re_repeat_bang_q.search(text),
            self._re_two_dots.search(text),
            self._re_bang_dots.search(text),
            self._re_question_dots.search(text),
            self._re_space_before_punct.search(text),
            self._re_missing_space_after.search(text),
            self._re_emoticon.search(text),
            self._re_interjection.search(text),
            self._re_abbr_dot.search(text),
            self._re_abbr_hyphen.search(text),
            self._re_initial.search(text),
            self._re_acronym.search(text),
            any(w in lower for w in _ACRONYM_WORDS),
        )
        return 0 if any(checks) else 1


if __name__ == "__main__":
    textfilter = TextFilter()

    dev_files = pd.read_csv(
        DEV_SET_PATH, sep="|", encoding="utf-8", quoting=csv.QUOTE_NONE, header=0
    )

    dev_files["predicted"] = dev_files["text"].apply(textfilter.filter)

    prc = precision_score(dev_files["is_normalized"], dev_files["predicted"])
    rec = recall_score(dev_files["is_normalized"], dev_files["predicted"])
    f1 = f1_score(dev_files["is_normalized"], dev_files["predicted"])
    print(f"F1 Score is {f1:.4f}, Precision is {prc:.4f}, Recall is {rec:.4f}")
