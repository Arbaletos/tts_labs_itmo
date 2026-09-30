"""Pause predictor — skeleton for lab 2.

Run as a script to score the predictor on the prepared data::

    python pause_predictor.py

Precision, recall and F1 are computed for `is_pause_after`, and MAE for `pause_duration`
on true positives only. The last word of every utterance is excluded.
"""
from __future__ import annotations

import csv
import os
import pickle
import re
import numpy as np
import tqdm
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import mean_absolute_error

PAUSE_PREDICTOR_DATA = 'data/RUSLAN_pause_metadata.csv'
MODEL_DIR = '.'


class PausePredictor():
    """Predicts where pauses fall in a sentence and how long they are.

    Input is one sentence as a sequence of `label_raw` tokens — words with their
    trailing punctuation, in order::

        ["Я", "вышел", "из", "дома,", "когда", "стемнело."]
    """

    def __init__(self, model_dir=MODEL_DIR):
        with open(os.path.join(model_dir, "pause_clf.pkl"), "rb") as f:
            self.clf = pickle.load(f)
        with open(os.path.join(model_dir, "pause_reg.pkl"), "rb") as f:
            self.reg = pickle.load(f)
        with open(os.path.join(model_dir, "word_to_emb.pkl"), "rb") as f:
            data = pickle.load(f)
            self.word_to_emb = data["word_to_emb"]
            self.unk_vector = data["unk"]
        with open(os.path.join(model_dir, "best_threshold.pkl"), "rb") as f:
            self.threshold = pickle.load(f)

    def _get_emb(self, word):
        return self.word_to_emb.get(str(word).lower(), self.unk_vector)

    def _has_punct(self, label_raw):
        return float(bool(re.search(r'[^\w\s]$', str(label_raw))))

    def _build_features(self, tokens):
        embs = np.array([self._get_emb(w) for w in tokens])
        lengths = np.array([len(str(w)) for w in tokens]).reshape(-1, 1)
        puncts = np.array([self._has_punct(w) for w in tokens]).reshape(-1, 1)
        return np.hstack([embs, lengths, puncts])

    def predict(self, tokens: list[str] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decide for each token whether a pause follows it.

        Args:
            tokens: Tokens of one sentence.

        Returns:
            `is_pause` (int, 1 if a pause follows the token) and `pause_duration`
            (float, seconds, 0.0 where there is no pause), both of length `len(tokens)`.

        Note:
            Wherever `is_pause` is 1 the duration must be positive:
            :meth:`predict_durations` relies on it.
        """
        n = len(tokens)
        is_pause = np.zeros(n, dtype=int)
        pause_duration = np.zeros(n, dtype=float)

        if n == 0:
            return is_pause, pause_duration

        X = self._build_features(tokens)
        y_proba = self.clf.predict_proba(X)[:, 1]
        is_pause = (y_proba > self.threshold).astype(int)

        pause_mask = is_pause == 1
        if pause_mask.sum() > 0:
            X_pause = X[pause_mask]
            pause_duration[pause_mask] = self.reg.predict(X_pause).flatten()

        return is_pause, pause_duration

    def predict_durations(self, tokens: list[str] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Insert predicted pauses into the token sequence.

        This is the form the acoustic model consumes in labs 4 and 5.

        Args:
            tokens: Tokens of one sentence.

        Returns:
            Tokens with ``"<SIL>"`` after every predicted pause, and one duration per
            output token: seconds for ``"<SIL>"``, ``-1.0`` for words (left to the
            acoustic model).
        """
        def expand_is_pause(token: str, is_pause: int) -> list[str]:
            if bool(is_pause):
                return [token, '<SIL>']
            return [token]

        def expand_durations(pause_duration: float) -> list[float]:
            if pause_duration > 0.:
                return [-1., pause_duration]
            return [-1.]

        is_pause, durations = self.predict(tokens)

        tokens_w_pauses = np.concatenate([expand_is_pause(a, b) for a, b in zip(tokens, is_pause)])
        durations_w_pauses = np.concatenate([expand_durations(dur) for dur in durations]).astype(np.float32)

        return tokens_w_pauses, durations_w_pauses


def calc_metrics(df: pd.DataFrame) -> None:
    """Print precision, recall and F1 for pause placement, and MAE for pause duration.

    MAE counts only rows where both the reference and the prediction have a pause.
    """
    if len(df) == 0:
        print("Empty dataframe")
        return

    rec = recall_score(df.is_pause_after, df.is_pause_hat, zero_division=0)
    prc = precision_score(df.is_pause_after, df.is_pause_hat, zero_division=0)
    f1 = f1_score(df.is_pause_after, df.is_pause_hat, zero_division=0)

    mask = (df.is_pause_after == 1) & (df.is_pause_hat == 1)
    mae = mean_absolute_error(df[mask].pause_duration, df[mask].pause_duration_hat) if mask.sum() > 0 else float('nan')

    print(f'PRC: {prc:.4f}, REC: {rec:.4f}, F1: {f1:.4f}; MAE: {mae:.4f};')


def test_pause_predictor() -> None:
    """Run the predictor on every sentence and print train and test metrics.

    Expects the layout written by `prepare_training_data.py`: rows grouped by utterance
    in order, each utterance ending with its `is_last_word` row.
    """
    pause_df = pd.read_csv(PAUSE_PREDICTOR_DATA, sep='|', quoting=csv.QUOTE_NONE)

    pp = PausePredictor()

    lens = {i: l for i, l in pause_df.groupby('id').count().reset_index(drop=False)[['id', 'label']].values}

    is_pause_after_hat = []
    pause_duration_hat = []

    idx = 0
    for i, is_last in tqdm.tqdm(pause_df[['id', 'is_last_word']].values):
        if not is_last:
            continue
        sentence = pause_df.iloc[idx:idx+lens[i]]
        idx += lens[i]

        is_pause_hat, pause_dur_hat = pp.predict(sentence.label_raw.values)
        is_pause_after_hat += list(is_pause_hat)
        pause_duration_hat += list(pause_dur_hat)

    pause_df['is_pause_hat'] = is_pause_after_hat
    pause_df['pause_duration_hat'] = pause_duration_hat

    print('Calculate metrics, traning fold; Exclude last tokens in every sentence!')
    calc_metrics(pause_df[(pause_df.set == 'train') & (pause_df.is_last_word == 0)])

    print('\nCalculate metrics, testing fold; Exclude last tokens in every sentence!')
    calc_metrics(pause_df[(pause_df.set == 'test') & (pause_df.is_last_word == 0)])


if __name__ == '__main__':
    test_pause_predictor()