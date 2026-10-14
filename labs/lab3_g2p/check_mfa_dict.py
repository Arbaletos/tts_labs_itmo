"""Быстрая проверка парсера MFA-словаря."""
from mfa_dict import load_mfa_dict

PATH = 'data/russian_mfa.dict'

d = load_mfa_dict(PATH)

print(f'Уникальных слов: {len(d)}')
print(f'Всего произношений: {sum(len(v) for v in d.values())}')

# проверка знакомых слов
for w in ['с', 'тревожным', 'замок', 'все', 'всё', 'звонок', 'синтез', 'кпсс']:
    prons = d.get(w)
    if prons is None:
        print(f'{w!r}: НЕТ В СЛОВАРЕ')
    else:
        print(f'{w!r}: {prons}')

# сколько слов с несколькими произношениями
multi = sum(1 for v in d.values() if len(v) > 1)
print(f'Слов с 2+ произношениями: {multi} ({multi/len(d):.1%})')