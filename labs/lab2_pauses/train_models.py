"""Train pause predictor models — lab 2 (CatBoost + Navec)."""
from __future__ import annotations

import csv
import numpy as np
import pandas as pd
import pickle
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import f1_score, precision_score, recall_score, mean_absolute_error
from sklearn.model_selection import train_test_split


# === 1. Загрузка эмбеддингов Navec ===
print("Loading Navec embeddings...")
embeddings = np.load("navec_hudlit_embeddings.npy")
with open("navec_hudlit_words.txt", encoding="utf-8") as f:
    words = [line.strip() for line in f]

word_to_emb = dict(zip(words, embeddings))
UNK_VECTOR = np.load("navec_hudlit_unk.npy")
print(f"Words: {len(word_to_emb)}, UNK norm: {np.linalg.norm(UNK_VECTOR):.4f}")


def get_emb(word):
    return word_to_emb.get(str(word).lower(), UNK_VECTOR)


# === 2. Загрузка данных ===
print("\nLoading RUSLAN data...")
df = pd.read_csv(
    r"D:\TTS_ITMO\tts_labs_itmo\data\RUSLAN_pause_metadata.csv",
    sep='|', quoting=csv.QUOTE_NONE,
)

df = df[df['is_last_word'] == 0].copy()
train_df = df[df['set'] == 'train'].copy()
test_df = df[df['set'] == 'test'].copy()
print(f"Train: {len(train_df)}, Test: {len(test_df)}")


# === 3. Признаки (302: эмбеддинг + длина + пунктуация) ===
def build_features(df):
    embs = np.array([get_emb(w) for w in df['label'].values])
    lengths = np.array([len(str(w)) for w in df['label'].values]).reshape(-1, 1)
    has_punct = df['label_raw'].str.contains(r'[^\w\s]$', regex=True, na=False).values.reshape(-1, 1).astype(float)
    return np.hstack([embs, lengths, has_punct])


X_train = build_features(train_df)
X_test = build_features(test_df)
y_train = train_df['is_pause_after'].values
y_test = test_df['is_pause_after'].values
print(f"X_train: {X_train.shape}")


# === 4. Train/Val split ===
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train,
)
print(f"Train: {X_tr.shape}, Val: {X_val.shape}")


# === 5. Классификатор CatBoost ===
print("\nTraining classifier (CatBoost)...")
clf = CatBoostClassifier(
    iterations=1000,
    depth=8,
    learning_rate=0.03,
    l2_leaf_reg=10,
    auto_class_weights='Balanced',
    loss_function='Logloss',
    eval_metric='F1',
    random_seed=42,
    verbose=200,
)
clf.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)


# === 6. Тюнинг порога ===
print("\n" + "=" * 60)
print("THRESHOLD TUNING")
print("=" * 60)

y_proba_val = clf.predict_proba(X_val)[:, 1]
best_f1 = 0
best_threshold = 0.5

for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_pred = (y_proba_val > threshold).astype(int)
    f1 = f1_score(y_val, y_pred)
    prc = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    print(f"Threshold {threshold}: PRC={prc:.4f}, REC={rec:.4f}, F1={f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"\nЛучший порог: {best_threshold}, F1: {best_f1:.4f}")


# === 7. Финальные метрики ===
print("\n" + "=" * 60)
print("FINAL METRICS")
print("=" * 60)

for set_name, X, y in [('train', X_train, y_train), ('test', X_test, y_test)]:
    y_proba = clf.predict_proba(X)[:, 1]
    y_pred = (y_proba > best_threshold).astype(int)
    prc = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    print(f"  {set_name:5}: PRC={prc:.4f}, REC={rec:.4f}, F1={f1:.4f}")


# === 8. Регрессор CatBoost ===
print("\nTraining regressor (CatBoost)...")
train_pauses = train_df[train_df['is_pause_after'] == 1]
X_train_reg = build_features(train_pauses)
y_train_reg = train_pauses['pause_duration'].values

reg = CatBoostRegressor(
    iterations=500,
    depth=8,
    learning_rate=0.03,
    l2_leaf_reg=10,
    loss_function='MAE',
    random_seed=42,
    verbose=200,
)
reg.fit(X_train_reg, y_train_reg)

for set_name, df_set, X_set in [('train', train_df, X_train), ('test', test_df, X_test)]:
    y_proba = clf.predict_proba(X_set)[:, 1]
    y_pred_clf = (y_proba > best_threshold).astype(int)
    tp_mask = (df_set['is_pause_after'] == 1) & (y_pred_clf == 1)
    if tp_mask.sum() > 0:
        X_tp = X_set[tp_mask]
        y_tp = df_set['pause_duration'].values[tp_mask]
        y_pred_dur = reg.predict(X_tp).flatten()
        mae = mean_absolute_error(y_tp, y_pred_dur)
        print(f"  {set_name:5}: MAE={mae:.4f} (TP={tp_mask.sum()})")


# === 9. Сохранение ===
print("\nSaving models...")
with open("pause_clf.pkl", "wb") as f:
    pickle.dump(clf, f)
with open("pause_reg.pkl", "wb") as f:
    pickle.dump(reg, f)
with open("word_to_emb.pkl", "wb") as f:
    pickle.dump({"word_to_emb": word_to_emb, "unk": UNK_VECTOR}, f)
with open("best_threshold.pkl", "wb") as f:
    pickle.dump(best_threshold, f)

print("Saved: pause_clf.pkl, pause_reg.pkl, word_to_emb.pkl, best_threshold.pkl")