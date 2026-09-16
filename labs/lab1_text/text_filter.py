"""Normalized / non-normalized classifier — skeleton for lab 1.

Run as a script to score yourself on the development set::

    python text_filter.py
"""

import csv
import re
import unicodedata

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

DEV_SET_PATH = "data/dev_sentences.csv"


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
        """Prepare the classifier's resources.

        Compiled regular expressions, abbreviation and contraction dictionaries, a
        trained model — anything that should not be rebuilt for every utterance.
        """

        self.patterns = [
            # цифры, латиница, техсимволы, невидимые символы
            re.compile(r"\d"),
            re.compile(r"[A-Za-z]"),
            re.compile(r"[*/+@<>&%]"),
            re.compile(r"[\u200b\u200c\u200d\ufeff\u00a0]"),

            # пунктуация
            re.compile(r"\s+[,.!?;:]"),
            re.compile(r"[,.;:!?][А-Яа-яЁё]"),
            re.compile(r"[?!]{2,}"),
            re.compile(r"!\."),
            re.compile(r"(?<!\.)\.\.(?!\.)"),
            re.compile(r"\(\s"),
            re.compile(r"\((?![^()]*\))"),

            # кавычки
            re.compile(r"[“”„]"),

            # сокращения, инициалы, аббревиатуры
            re.compile(r"(?<![а-яА-ЯёЁ])(?:ул|г|г-н|г-жа|т\.д|т\.п|др|им|прим|руб|коп|тыс|млн|млрд|стр|рис)\.\s"),
            re.compile(r"(?<![а-яА-ЯёЁ])г-ж[аеиуой](?![а-яА-ЯёЁ])|(?<![а-яА-ЯёЁ])г-н(?![а-яА-ЯёЁ])", re.IGNORECASE),
            re.compile(r"(?<![а-яА-ЯёЁ])(?:Минюст|Минздрав|Минобр|Госдума|Госуслуги|ЦБ|ФНС|ФСБ|МЧС)(?![а-яА-ЯёЁ])", re.IGNORECASE),
            re.compile(r"(?<![а-яА-ЯёЁ])[А-ЯЁ]\.\s"),
            re.compile(r"(?<![а-яА-ЯёЁ])[А-ЯЁ]{2,4}(?![а-яА-ЯёЁ])"),

            # смесь алфавитов в слове
            re.compile(r"\b\w*[A-Za-z]\w*[а-яА-ЯёЁ]\w*\b|\b\w*[а-яА-ЯёЁ]\w*[A-Za-z]\w*\b"),

            # междометия
            re.compile(r"(?<![а-яА-ЯёЁ])(?:ммм|хм|хмм|хехе|хаха|мда|ыы|гмм|угу|ага|эх|ох|ах|ой)(?![а-яА-ЯёЁ])", re.IGNORECASE),

            # смайлики
            re.compile(r"[:;]-?[)(Рp3]"),
        ]

        #  список разрешенной пунктуации и символов
        self.allowed_punct = set(".,!?:;\"'«»…—- ")

    def _has_invalid_characters(self, text: str) -> bool:
        """Проверяет наличие символов вне разрешённого алфавита."""
        for ch in text:
            # Знак ударения U+0301
            if ch == "\u0301":
                continue
            # Кириллические буквы
            if "CYRILLIC" in unicodedata.name(ch, ""):
                continue
            # Разрешённая пунктуация и пробелы
            if ch in self.allowed_punct:
                continue
            # Всё остальное — недопустимо
            return True
        return False

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

        text = str(text)
        for p in self.patterns:
            if p.search(text):
                return 0

        #  проверка на неизвестные/нестандартные юникод-символы
        if self._has_invalid_characters(text):
            return 0

        return 1


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