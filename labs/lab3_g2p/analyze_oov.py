"""Анализ ошибок OOV-модуля G2P: печатает слово, эталон, предсказание."""
import sys
from g2p import G2P, load_tokens, build_lexicon, SIL


def main(n_examples=20):
    tokens = load_tokens()
    train = tokens[tokens.set == 'train']
    test = tokens[tokens.set == 'test']

    lexicon = build_lexicon(train)
    g2p = G2P(lexicon)

    shown = 0
    for _, sentence in test.groupby('id', sort=False):
        sentence = sentence.sort_values('token_id')
        words = sentence.token.values
        predicted = g2p.transcribe(words)
        reference = [p.split() for p in sentence.phones.values]

        for token, ref, hyp in zip(words, reference, predicted):
            if token in lexicon:
                continue
            ref_no_sil = [p for p in ref if p != SIL]
            hyp_no_sil = [p for p in hyp if p != SIL]
            if ref_no_sil == hyp_no_sil:
                continue
            print(f'{token:25}  REF: {" ".join(ref_no_sil)}')
            print(f'{"":25}  HYP: {" ".join(hyp_no_sil)}')
            print()
            shown += 1
            if shown >= n_examples:
                return


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    main(n)