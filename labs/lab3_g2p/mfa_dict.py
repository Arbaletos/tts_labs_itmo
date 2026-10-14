"""Утилиты для работы с MFA-словарём произношений.

Входной файл — probability-версия russian_mfa (6 колонок через таб):
    word  p_sil  p_nosil  p_pron1  p_pron2  phones
"""
from pathlib import Path


def load_mfa_dict(path: str | Path) -> dict[str, list[str]]:
    """Читает probability-формат russian_mfa.dict."""
    result: dict[str, list[str]] = {}
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.rstrip('\r\n')
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 6:
                continue
            word, _p_sil, _p_nosil, _p_pron1, _p_pron2, phones = parts
            result.setdefault(word, []).append(phones)
    return result


def save_mfa_dict_csv(d: dict[str, list[str]], path: str | Path) -> None:
    """Сохраняет распарсенный MFA-словарь в CSV (word|phones|rank)."""
    import csv
    with open(path, 'w', encoding='utf-8', newline='') as fh:
        writer = csv.writer(fh, delimiter='|', quoting=csv.QUOTE_NONE)
        writer.writerow(['word', 'phones', 'rank'])
        for word, prons in d.items():
            for rank, pron in enumerate(prons):
                writer.writerow([word, pron, rank])