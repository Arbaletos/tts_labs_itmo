"""Build pause predictor training data — lab 2.

Joins the lab 1 normalized metadata with the MFA word alignment and writes one row per
word to `data/RUSLAN_pause_metadata.csv`::

    id|label|label_raw|duration|is_last_word|is_pause_after|pause_duration|set

Utterances whose id ends in 0 or 5 go to `test`, the rest to `train`.

Run from the lab directory::

    python prepare_training_data.py
"""
from __future__ import annotations
import csv
import glob
import numpy as np
import os
import pandas as pd
from praatio import textgrid
import tqdm



RUSLAN_META = r'D:\TTS_ITMO\tts_labs_itmo\labs\lab1_text\data\metadata_RUSLAN_22200_normalized.csv'
ALIGN_DIR = r'D:\TTS_ITMO\tts_labs_itmo\data\RUSLAN_22050_align_v2'
RESULT_PATH = r'D:\TTS_ITMO\tts_labs_itmo\data\RUSLAN_pause_metadata.csv'

def read_text_grids(ruslan: pd.DataFrame, align_root: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:   
    """Read the MFA TextGrid of every utterance in `ruslan`.

    Args:
        ruslan: Metadata with `id` and `nrm` columns.
        align_root: Directory with `{id}.TextGrid` files.

    Returns:
        Word intervals (`label`, `duration`, `id`; silence has label ``""``), phone
        intervals (same columns; silence is ``"<SIL>"``), and one space-joined phone
        string per metadata row (``""`` when the TextGrid is missing).
    """
    word_docs = []
    phn_docs = []
    phoneme_sequences = []
    for wave_id, text in tqdm.tqdm(ruslan[['id', 'nrm']].values):
        try:
            tg = textgrid.openTextgrid(os.path.join(align_root, wave_id+'.TextGrid'), True)
        except:
            phoneme_sequences.append('')
            continue
        words, phones = tg.tiers

        words = words.entries
        for i in words:
            word_docs.append({'label':i.label, 'duration':i.end-i.start, 'id': wave_id})
        
        phoneme_sequences.append(' '.join([p.label for p in phones.entries]))
        phones = phones.entries
        for i in phones:
            if i.label=='':
                phn_docs.append({'label':'<SIL>', 'duration':i.end-i.start, 'id': wave_id})
            else:
                phn_docs.append({'label':i.label, 'duration':i.end-i.start, 'id': wave_id})
    word_df = pd.DataFrame(word_docs)
    phn_df = pd.DataFrame(phn_docs)
    return word_df, phn_df, phoneme_sequences


def align_text_and_textgrid(tokens: pd.DataFrame, text: str) -> pd.DataFrame:
    """Attach the original text form of each aligned word as `label_raw`.

    MFA labels are lowercase and carry no punctuation. `label_raw` restores the case
    and appends the punctuation that follows each word; text before the first word goes
    to the first word. Silence intervals get ``"<SIL>"``.

    Args:
        tokens: Word intervals of one utterance, as returned by :func:`read_text_grids`.
        text: Normalized text of the same utterance.

    Returns:
        `tokens` with a `label_raw` column, or `tokens` unchanged if a word is not
        found in `text` — such utterances are dropped later.
    """
    tokens = tokens.copy()
    raw_tokens = []
    text_lower = text.lower()
    previous_word = -1
    for t, d, i in tokens[['label', 'duration', 'id']].values:
        if t == '': # Empty, SIL token
            raw_tokens.append('<SIL>')
            continue
        splits = text_lower.split(t, maxsplit=1)
        if len(splits)==1: # Does not found content!
            print(f'Error aligning f{i}!')
            print(t, text, tokens)
            return tokens 
        if previous_word == -1: 
            raw_tokens.append(text[:len(splits[0]+t)].strip())
        else:
            raw_tokens[previous_word] += text[:len(splits[0])].strip()
            raw_tokens.append(text[len(splits[0]):len(splits[0] + t)])
        text_lower = splits[1]
        text = text[len(splits[0] + t):]
        previous_word = len(raw_tokens)-1
    if len(text) and (previous_word>=0):
        raw_tokens[previous_word] += text.strip()
    tokens['label_raw'] = raw_tokens
    return tokens

def add_pause_labels(align: pd.DataFrame) -> pd.DataFrame:
    """Mark which words are followed by a pause, and for how long.

    Adds `is_last_word`, `is_pause_after` and `pause_duration` (seconds). Silence rows
    themselves get ``False`` / ``0.0``; silence before the first word is ignored.

    Args:
        align: Word intervals of one utterance, in order.

    Returns:
        `align` with the three label columns.
    """
def add_pause_labels(align: pd.DataFrame) -> pd.DataFrame:
    """Mark which words are followed by a pause, and for how long.

    Adds `is_last_word`, `is_pause_after` and `pause_duration` (seconds). Silence rows
    themselves get ``False`` / ``0.0``; silence before the first word is ignored.

    Args:
        align: Word intervals of one utterance, in order.

    Returns:
        `align` with the three label columns.
    """
    align = align.copy()  # ← убирает SettingWithCopyWarning
    
    n = len(align)
    is_last_word = [False] * n
    pause_after = [False] * n
    pause_duration = [0.0] * n
    
    last_word = -1
    for idx, (label, dur) in enumerate(align[['label', 'duration']].values):
        if label == '':
            # Пауза: приклеиваем к предыдущему слову
            if last_word >= 0:
                pause_after[last_word] = True
                pause_duration[last_word] = dur
        else:
            # Слово: запоминаем его индекс
            last_word = idx
    
    # Последнее слово в предложении
    if last_word >= 0:
        is_last_word[last_word] = True
    
    align['is_last_word'] = is_last_word
    align['is_pause_after'] = pause_after
    align['pause_duration'] = pause_duration
    return align
    

def main() -> None:
    """Read metadata and alignments, label pauses, split into train/test and save."""
    ruslan =pd.read_csv(f'{RUSLAN_META}', sep='|', names=['id', 'raw', 'nrm'], quoting=csv.QUOTE_NONE)
    word_df, _, _ = read_text_grids(ruslan, ALIGN_DIR)

    
    #Additional normalization to ensure good alignment
    word_df.label = word_df.label.str.replace('‐', '-') # differen hyphen symbols
    word_df.label = word_df.label.str.replace('‑', '-')
    ruslan.nrm = ruslan.nrm.str.replace('‐', '-')
    ruslan.nrm = ruslan.nrm.str.replace('‑', '-')
    ruslan.nrm = ruslan.nrm.str.replace('’', "'") # Mfa replaces ’ by '
    ruslan.nrm = ruslan.nrm.str.replace('\\((.*?)\\)', '[bracketed]', regex=True) # Mfa spells () and <> text as [bracketed]
    ruslan.nrm = ruslan.nrm.str.replace('\\<(.*?)\\>', '[bracketed]', regex=True)

    #Aligning TextGrid-normalised tokens and text, including punctuation
    aligns = []
    for n, i in tqdm.tqdm(ruslan[['nrm', 'id']].values):
        tokens = word_df[word_df.id==i]
        aligns.append(align_text_and_textgrid(tokens, n))

    # Creating labels for pause predictor training
    aligns = [add_pause_labels(a) for a in aligns]

    # Creating dataframe
    pause_df = pd.concat(aligns)
    
    pause_df = pause_df[pause_df.label_raw!='<SIL>'] # Removing pause tokens -- all information about pauses is in word tokens now
    pause_df = pause_df[pause_df.label_raw.notna()] # Removing all the files, who failed to be aligned
    pause_df = pause_df.reset_index(drop=True) # Reseting the index after filtering
    
    pause_df.is_last_word = pause_df.is_last_word.astype(int) # Casting bool fields into int
    pause_df.is_pause_after = pause_df.is_pause_after.astype(int) #


    # Splitting into train and test deterministically: All the files with index, ending with 0 or 5 is considered test
    pause_df['set'] = 'train'
    pause_df.loc[pause_df.id.str.split('_', expand=True)[0].astype(int)%5==0, 'set'] = 'test'
    pause_df.to_csv(f'{RESULT_PATH}', sep='|', index=False, header=True, quoting=csv.QUOTE_NONE)
    
if __name__=='__main__':
    main()