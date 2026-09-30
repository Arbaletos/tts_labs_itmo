"""Normalized / non-normalized classifier — skeleton for lab 1.

Run as a script to score yourself on the development set::

    python text_filter.py
"""

import csv
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

DEV_SET_PATH = "D:/TTS_ITMO/tts_labs_itmo/labs/lab1_text/data/dev_sentences.csv"


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
        """Prepare the classifier's resources."""
        self.stop_words = {
            "он", "ты", "мы", "вы", "я", "она", "они", "-то", "все",
            "его", "ее", "же", "бы", "да", "их", "ей", "уж", "ну",
            "мне", "те", "се", "вас", "нас", "сам", "ах", "ох", "эх",
            "ой", "ли", "меня", "тебя", "себя", "вами", "нами",
            "мой", "твой", "свой", "это", "эти", "эта", "тот", "та", "те",
            "вон", "вот", "кто", "что", "чей", "чья", "чьё",
            "весь", "вся", "всё", "сама", "сами", "ага", "увы", "ого"
        }
        
        self.abbr_pattern = re.compile(r'\b[а-яА-Я]{1,2}\.')
        self.acronym_pattern = re.compile(r'\b[А-ЯЁ]{2,}\b')
        self.tech_pattern = re.compile(r'[\(\)\[\]\{\}\/\*<>]')
        self.digit_pattern = re.compile(r'\d')
        self.latin_pattern = re.compile(r'[a-zA-Z]')
        self.multi_punct_pattern = re.compile(r'[!?]{2,}')
        self.quote_pattern = re.compile(r'[«»“”„]')
        self.dash_pattern = re.compile(r'[–—]')
        self.bracket_pattern = re.compile(r'[\(\)]')
        self.bad_chars_pattern = re.compile(r'[^а-яА-ЯёЁ\s\.\,\!\?\-]')
        self.word_pattern = re.compile(r'\b[а-яА-ЯЁё]+\b')
        
        self.model = None
        self.is_fitted = False

    def _extract_features(self, text: str) -> np.ndarray:
        if not isinstance(text, str):
            text = str(text)
        
        text_len = len(text)
        words = self.word_pattern.findall(text)
        word_count = len(words)
        
        abbrs = self.abbr_pattern.findall(text)
        abbr_count = sum(1 for w in abbrs if w[:-1].lower() not in self.stop_words)
        
        acronyms = self.acronym_pattern.findall(text)
        vowels = 'АЕЁИОУЫЭЮЯ'
        acronym_count = sum(1 for w in acronyms if not any(v in w for v in vowels))
        
        tech_count = len(self.tech_pattern.findall(text))
        digit_count = len(self.digit_pattern.findall(text))
        latin_count = len(self.latin_pattern.findall(text))
        multi_punct_count = len(self.multi_punct_pattern.findall(text))
        quote_count = len(self.quote_pattern.findall(text))
        dash_count = len(self.dash_pattern.findall(text))
        bracket_count = len(self.bracket_pattern.findall(text))
        
        bad_chars = self.bad_chars_pattern.findall(text)
        bad_ratio = len(bad_chars) / text_len if text_len > 0 else 0
        
        punct_count = len(re.findall(r'[.,!?;:]', text))
        punct_ratio = punct_count / word_count if word_count > 0 else 0
        
        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        
        sent_count = len(re.split(r'[.!?]+', text)) - 1
        
        cap_words = sum(1 for w in words if w[0].isupper())
        cap_ratio = cap_words / word_count if word_count > 0 else 0
        
        features = np.array([
            abbr_count,
            acronym_count,
            tech_count,
            digit_count,
            latin_count,
            multi_punct_count,
            quote_count,
            dash_count,
            bracket_count,
            bad_ratio,
            punct_ratio,
            avg_word_len,
            word_count,
            sent_count,
            text_len / 100,
            cap_ratio,
            len(set(words)) / word_count if word_count > 0 else 0,
        ])
        
        return features

    def fit(self, X, y):
        features_list = [self._extract_features(text) for text in X]
        X_features = np.array(features_list)
        
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        self.model.fit(X_features, y)
        self.is_fitted = True
        
        return self

    def filter(self, text: str) -> int:
        if not isinstance(text, str):
            text = str(text)
        
        if self.is_fitted and self.model is not None:
            features = self._extract_features(text).reshape(1, -1)
            prediction = self.model.predict(features)[0]
            return int(prediction)
        
        return self._heuristic_filter(text)

    def _heuristic_filter(self, text: str) -> int:
        if self.tech_pattern.search(text):
            return 0
        
        candidates = self.abbr_pattern.findall(text)
        for word in candidates:
            if word[:-1].lower() not in self.stop_words:
                return 0
        
        words = self.acronym_pattern.findall(text)
        vowels = 'АЕЁИОУЫЭЮЯ'
        for w in words:
            if not any(v in w for v in vowels):
                return 0
        
        if self.digit_pattern.search(text):
            return 0
        if self.latin_pattern.search(text):
            return 0
        if self.multi_punct_pattern.search(text):
            return 0
        if self.bracket_pattern.search(text):
            return 0
        if self.quote_pattern.search(text):
            return 0
        if self.dash_pattern.search(text):
            return 0
        
        bad_chars = self.bad_chars_pattern.findall(text)
        bad_ratio = len(bad_chars) / len(text) if len(text) > 0 else 0
        if bad_ratio > 0.08:
            return 0
        
        return 1


if __name__ == "__main__":
    textfilter = TextFilter()
    
    dev_data = pd.read_csv(
        DEV_SET_PATH, sep="|", encoding="utf-8",
        quoting=csv.QUOTE_NONE, header=0
    )
    
    if not textfilter.is_fitted:
        X_train, X_val, y_train, y_val = train_test_split(
            dev_data['text'],
            dev_data['is_normalized'],
            test_size=0.2,
            random_state=42,
            stratify=dev_data['is_normalized']
        )
        textfilter.fit(X_train, y_train)
    
    predictions = [textfilter.filter(text) for text in dev_data['text']]
    
    f1 = f1_score(dev_data['is_normalized'], predictions)
    prc = precision_score(dev_data['is_normalized'], predictions)
    rec = recall_score(dev_data['is_normalized'], predictions)
    
    print(f"F1 Score is {f1:.4f}, Precision is {prc:.4f}, Recall is {rec:.4f}")