"""Подготовка метаданных для лабы 4 (эталонные и предсказанные)."""
import csv
from collections import defaultdict

import pandas as pd
import tqdm

from g2p import (
    G2P, load_tokens, build_lexicon, load_mfa_dict,
    normalize_token, SIL,
)


def build_metadata(tokens: pd.DataFrame, g2p: G2P | None = None) -> list[dict]:
    """Собирает метаданные: одна строка на предложение.

    Args:
        tokens: DataFrame с колонками id, token_id, token, phones, set.
        g2p: если задан — использовать его транскрипцию. Если None — эталонную.

    Returns:
        Список словарей: [{'id': ..., 'text': ..., 'phones': ...}, ...].
    """
    result = []
    for utt_id, sentence in tqdm.tqdm(tokens.groupby('id', sort=False),
                                      total=tokens['id'].nunique()):
        sentence = sentence.sort_values('token_id')
        words = sentence.token.values
        phones_ref = sentence.phones.values

        # Склеиваем текст
        text = ' '.join(str(w) for w in words)

        if g2p is None:
            # Эталон: склеиваем phones как есть
            phones_seq = ' '.join(str(p) for p in phones_ref)
        else:
            # Предсказание: прогоняем через G2P, склеиваем
            predicted = g2p.transcribe(words)
            phones_list = []
            for p_list in predicted:
                phones_list.extend(p_list)
            phones_seq = ' '.join(phones_list)

        result.append({
            'id': utt_id,
            'text': text,
            'phones': phones_seq,
        })
    return result


def write_metadata(rows: list[dict], path: str) -> None:
    """Записывает метаданные в файл с разделителем |."""
    with open(path, 'w', encoding='utf-8', newline='') as fh:
        writer = csv.writer(fh, delimiter='|', quoting=csv.QUOTE_NONE)
        writer.writerow(['id', 'text', 'phones'])
        for row in rows:
            writer.writerow([row['id'], row['text'], row['phones']])


def main() -> None:
    # Загружаем tokens_phones_sil.csv — с паузами
    print('Loading tokens_phones_sil.csv...')
    tokens_all = pd.read_csv(
        'data/tokens_phones_sil.csv',
        sep='|', quoting=csv.QUOTE_NONE,
    )

    # Фильтруем spn и пустые phones
    unpronounced = (
        (tokens_all.phones == '')
        | tokens_all.phones.isna()
        | tokens_all.phones.astype(str).str.split().map(lambda p: 'spn' in p)
    )
    print(f'Dropping {unpronounced.sum()} tokens with spn or empty phones')
    tokens = tokens_all[~unpronounced].reset_index(drop=True)

    # --- 1. Эталонные метаданные ---
    print('\n=== Building GT metadata ===')
    meta_gt = build_metadata(tokens, g2p=None)
    write_metadata(meta_gt, 'data/metadata_phones_gt.csv')
    print(f'Wrote {len(meta_gt)} utterances to data/metadata_phones_gt.csv')

    # --- 2. Предсказанные метаданные ---
    print('\n=== Building predicted metadata ===')
    # Загружаем train-only для лексикона (как в задании)
    tokens_no_sil = load_tokens('data/tokens_phones.csv')
    train = tokens_no_sil[tokens_no_sil.set == 'train']
    lexicon = build_lexicon(train)
    mfa_dict = load_mfa_dict('data/mfa_dict.csv')
    g2p = G2P(lexicon, mfa_dict)

    meta_pred = build_metadata(tokens, g2p=g2p)
    write_metadata(meta_pred, 'data/metadata_phones.csv')
    print(f'Wrote {len(meta_pred)} utterances to data/metadata_phones.csv')


if __name__ == '__main__':
    main()