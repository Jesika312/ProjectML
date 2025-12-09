"""
balance_undersample.py

Undersample majority class to match minority class (50:50) WITHOUT duplication.

Config:
- INPUT_FILE: path ke CSV Anda yang berlabel (memiliki kolom 'label')
- OUTPUT_FILE: path file balanced yang akan disimpan
- LABEL_COL: nama kolom label (0/1)
"""

import pandas as pd
from sklearn.utils import resample
import os

# ========== CONFIG ==========
INPUT_FILE = "combined_training_dataset.csv"
OUTPUT_FILE = "balanced_dataset_undersample.csv"
LABEL_COL = "label"
RANDOM_STATE = 42
# ============================

def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file tidak ditemukan: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    if LABEL_COL not in df.columns:
        raise ValueError(f"Kolom label '{LABEL_COL}' tidak ditemukan di {INPUT_FILE}")

    # pertahankan hanya label 0 dan 1
    df = df[df[LABEL_COL].isin([0,1])].copy()

    counts = df[LABEL_COL].value_counts()
    if len(counts) < 2:
        raise ValueError("Dataset tidak memiliki dua kelas (0 dan 1).")

    print("Jumlah sebelum balancing:")
    print(counts)

    # identifikasi mayoritas/minoritas
    majority_label = counts.idxmax()
    minority_label = counts.idxmin()
    majority_df = df[df[LABEL_COL] == majority_label]
    minority_df = df[df[LABEL_COL] == minority_label]

    # undersample majority tanpa pengembalian (replace=False)
    majority_downsampled = resample(
        majority_df,
        replace=False,
        n_samples=len(minority_df),
        random_state=RANDOM_STATE
    )

    # gabungkan hasil
    balanced = pd.concat([majority_downsampled, minority_df], axis=0)
    balanced = balanced.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    print("\nJumlah setelah undersampling (50:50):")
    print(balanced[LABEL_COL].value_counts())

    balanced.to_csv(OUTPUT_FILE, index=False)
    print(f"\nBalanced dataset tersimpan di: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
