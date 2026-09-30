"""Filter RUSLAN corpus using TextNormalizer and TextFilter."""

import csv
import pandas as pd
from text_normalizer import TextNormalizer
from text_filter import TextFilter


INPUT_PATH = "/labs/lab1_text/data/metadata_RUSLAN_22200.csv"
DEV_SET_PATH = "labs/lab1_text/data/dev_sentences.csv"
OUTPUT_PATH = "labs/lab1_text/data/metadata_RUSLAN_22200_normalized.csv"


def main():
    print("=" * 60)
    print("Filtering RUSLAN corpus")
    print("=" * 60)

    # БЕЗ заголовков
    df = pd.read_csv(
        INPUT_PATH,
        sep="|",
        encoding="utf-8",
        quoting=csv.QUOTE_NONE,
        header=None,
        names=["filename", "text"],
    )

    print(f"\nLoaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    print(df.head())


    normalizer = TextNormalizer()
    text_filter = TextFilter()

    print("\nTraining TextFilter on dev set...")
    dev_data = pd.read_csv(
        DEV_SET_PATH,
        sep="|",
        encoding="utf-8",
        quoting=csv.QUOTE_NONE,
        header=0,
    )
    text_filter.fit(dev_data["text"], dev_data["is_normalized"])

    # Нормализация + фильтрация
    print("\nNormalizing and filtering...")

    normalized_texts = []
    keep_flags = []

    for i, row in df.iterrows():
        raw_text = row["text"]

        if not isinstance(raw_text, str):
            raw_text = str(raw_text)

        normalized = normalizer.normalize(raw_text)
        keep = text_filter.filter(normalized)

        normalized_texts.append(normalized)
        keep_flags.append(keep)

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(df)}")


    df["normalized_text"] = normalized_texts
    df["keep"] = keep_flags

    #Статистика
    total = len(df)
    kept = sum(keep_flags)
    dropped = total - kept

    print(f"\nStatistics:")
    print(f"Total:   {total}")
    print(f"Kept:    {kept} ({kept/total*100:.1f}%)")
    print(f"Dropped: {dropped} ({dropped/total*100:.1f}%)")

    print(f"\nSaving: {OUTPUT_PATH}")

    output_df = df[df["keep"] == 1][["filename", "text", "normalized_text"]].copy()
    output_df.columns = ["filename", "raw_text", "normalized_text"]

    output_df.to_csv(
        OUTPUT_PATH,
        sep="|",
        encoding="utf-8",
        quoting=csv.QUOTE_NONE,
        index=False,
        header=True,
    )

    print(f"Saved {len(output_df)} rows")
    print("\nDone!")


if __name__ == "__main__":
    main()