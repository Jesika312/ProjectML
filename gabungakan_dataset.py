"""
combine_datasets_phase3.py

Script ini hanya untuk:
- Menggabungkan dataset Kaggle (berlabel) + dataset scraped (yang sudah divalidasi manual)
- Melakukan preprocessing ringan
- Menyimpan dataset gabungan untuk proses training ulang fase 3

Output disimpan dalam:
    combined_training_dataset.csv
"""

# ================= CONFIG =================
KAGGLE_TRAIN = "test_outputs/predicted_scraped_dataset.csv"  # HARUS sudah ada kolom 'label'
SCRAPED_VALIDATED = "balanced_dataset_undersample.csv"  # HARUS sudah ada kolom 'label'
OUTPUT_FILE = "combined_training_dataset.csv"
TEXT_COL = "label"   # auto detect unless forced (e.g. "komentar")
# ===========================================

import os, sys, re
import pandas as pd

# optional NLP libs
try:
    from unidecode import unidecode
    import emoji
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    stemmer = StemmerFactory().create_stemmer()
    sastrawi_available = True
except:
    sastrawi_available = False


# ---------- Cleaning ----------
def simple_clean(t):
    if not isinstance(t, str):
        return ""
    try: t = unidecode(t)
    except: pass
    try: t = emoji.demojize(t, delimiters=(" ", " "))
    except: pass
    t = t.lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"#\w+", " ", t)
    t = re.sub(r"[^0-9a-zA-Z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def maybe_stem(t):
    if not isinstance(t, str) or t.strip() == "":
        return ""
    if sastrawi_available:
        try: return stemmer.stem(t)
        except: return t
    return t


# ---------- Detect text column ----------
def detect_text_column(df):
    if TEXT_COL and TEXT_COL in df.columns:
        return TEXT_COL

    candidates = ["stemmed", "komentar_clean", "komentar", "comment", "text", "text_clean"]
    for c in candidates:
        if c in df.columns:
            return c

    for col in df.columns:
        if "komentar" in col.lower() or "comment" in col.lower():
            return col

    raise ValueError("Tidak menemukan kolom teks pada dataset!")


# ---------- MAIN ----------
def main():
    print("=== MENGGABUNGKAN DATASET (PHASE 3) ===")

    if not os.path.exists(KAGGLE_TRAIN):
        print("ERROR: Dataset Kaggle tidak ditemukan:", KAGGLE_TRAIN)
        sys.exit()

    if not os.path.exists(SCRAPED_VALIDATED):
        print("ERROR: Dataset scraped validasi tidak ditemukan:", SCRAPED_VALIDATED)
        sys.exit()

    # Load datasets
    df_kaggle = pd.read_csv(KAGGLE_TRAIN)
    df_scraped = pd.read_csv(SCRAPED_VALIDATED)

    if "label" not in df_scraped.columns:
        print("ERROR: Dataset scraped belum memiliki kolom label manual!")
        sys.exit()

    print("Kaggle shape :", df_kaggle.shape)
    print("Scraped shape:", df_scraped.shape)

    # Gabungkan dataset
    df_all = pd.concat([df_kaggle, df_scraped], ignore_index=True)
    print("Total combined:", df_all.shape)

    # Detect text column
    text_col = detect_text_column(df_all)
    print("Kolom teks terdeteksi:", text_col)

    # Preprocessing ringan untuk gabungan dataset
    df_all["text_clean"] = (
        df_all[text_col]
        .astype(str)
        .fillna("")
        .apply(simple_clean)
        .apply(maybe_stem)
    )

    # keep only rows with valid labels 0/1
    df_all = df_all[df_all["label"].isin([0,1])]

    # Save result
    df_all.to_csv(OUTPUT_FILE, index=False)
    print("\n=== Dataset gabungan berhasil disimpan ===")
    print("File :", OUTPUT_FILE)
    print("Total rows:", len(df_all))
    print("=========================================")


if __name__ == "__main__":
    main()
