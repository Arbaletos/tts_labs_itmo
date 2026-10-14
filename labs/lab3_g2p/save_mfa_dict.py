"""Сохраняет russian_mfa_source.dict в удобный CSV."""
from mfa_dict import load_mfa_dict, save_mfa_dict_csv

SRC = 'data/russian_mfa.dict'
DST = 'data/mfa_dict.csv'

print(f'Читаю {SRC}...')
d = load_mfa_dict(SRC)
print(f'  слов: {len(d)}, всего произношений: {sum(len(v) for v in d.values())}')

print(f'Сохраняю в {DST}...')
save_mfa_dict_csv(d, DST)
print('Готово.')