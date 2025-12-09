"""
EDA + Preprocessing (DISesuaikan dengan dataset Anda)

Input expected (letakkan di folder yang sama):
- data_komentar_dengan_prediksi - data_komentar_dengan_prediksi(2).csv  (dataset BERLABEL)
- youtube_comments_judol.csv                                              (dataset SCRAPED, biasanya belum berlabel)

Output:
- preprocessed_kaggle.csv      (hasil preprocessing dataset berlabel)
- preprocessed_scraped.csv     (hasil preprocessing dataset scraped)
- outputs_kaggle/              (visualisasi EDA untuk dataset berlabel)
- outputs_scraped/             (visualisasi EDA untuk dataset scraped)
"""

# ========================= CONFIG =========================
KAGGLE_FILE   = "dataset/data_komentar_dengan_prediksi - data_komentar_dengan_prediksi(2).csv"
SCRAPED_FILE  = "dataset/youtube_comments_judol.csv"
TEXT_COL      = "komentar"   # kolom teks di kedua file
LABEL_COL     = "label"      # kolom label di file kaggle-like
OUT_KAGGLE    = "preprocessed_kaggle.csv"
OUT_SCRAPED   = "preprocessed_scraped.csv"
OUTDIR_KAGGLE = "outputs_kaggle"
OUTDIR_SCRAPED= "outputs_scraped"
# ========================================================

import os, re, unicodedata, string
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# external libs - pastikan terpasang
from unidecode import unidecode
import emoji
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

stemmer = StemmerFactory().create_stemmer()

# ---------------- preprocessing helpers ----------------
LEET_MAP = {
    "4":"a","3":"e","0":"o","1":"i","5":"s","7":"t","$":"s","@":"a"
}

URL_PATTERN = re.compile(r'http\S+|www\.\S+')
MENTION_PATTERN = re.compile(r'@\w+')
HASHTAG_PATTERN = re.compile(r'#\w+')
NON_ALPHANUM_RE = re.compile(r'[^0-9a-zA-Z\s]')

def normalize_unicode(text):
    if not isinstance(text, str):
        return ""
    return unidecode(unicodedata.normalize("NFKC", text))

def emoji_to_text(text):
    if not isinstance(text, str):
        return ""
    # demojize -> returns strings like ":smile:"; delimiters to separate tokens
    return emoji.demojize(text, delimiters=(" ", " "))

def leet_to_text(text):
    if not isinstance(text, str):
        return ""
    return "".join(LEET_MAP.get(ch, ch) for ch in text)

def clean_text(raw):
    if not isinstance(raw, str):
        return ""
    text = normalize_unicode(raw)
    text = text.lower()
    text = emoji_to_text(text)
    text = leet_to_text(text)
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    text = HASHTAG_PATTERN.sub(" ", text)
    text = NON_ALPHANUM_RE.sub(" ", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text):
    if not isinstance(text, str):
        return []
    return text.split()

def stem_indonesia(text):
    if not isinstance(text, str) or text == "":
        return ""
    try:
        return stemmer.stem(text)
    except Exception:
        return text

# ---------------- EDA helpers ----------------
def plot_label_distribution(df, out_path):
    plt.figure(figsize=(6,4))
    sns.countplot(x=df[LABEL_COL].astype(str))
    plt.title("Distribusi Label")
    plt.xlabel("Label")
    plt.ylabel("Jumlah")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_length_histogram(df, text_col, out_hist, out_box):
    df['char_count'] = df[text_col].astype(str).apply(len)
    plt.figure(figsize=(8,5))
    sns.histplot(df['char_count'], bins=50)
    plt.title("Histogram Panjang Komentar (karakter)")
    plt.xlabel("Panjang (karakter)")
    plt.ylabel("Frekuensi")
    plt.tight_layout()
    plt.savefig(out_hist)
    plt.close()

    plt.figure(figsize=(8,5))
    sns.boxplot(x=df[LABEL_COL].astype(str), y=df['char_count'])
    plt.title("Boxplot Panjang Komentar per Label")
    plt.xlabel("Label")
    plt.ylabel("Panjang (karakter)")
    plt.tight_layout()
    plt.savefig(out_box)
    plt.close()

def top_n_words(df, text_col='stemmed', n=20, label_filter=None):
    if label_filter is not None:
        df = df[df[LABEL_COL] == label_filter]
    tokens = []
    for txt in df[text_col].astype(str):
        tokens.extend(tokenize(txt))
    return Counter(tokens).most_common(n)

def plot_top_words_bar(top_words, out_path, title="Top words"):
    if not top_words:
        return
    words, freqs = zip(*top_words)
    plt.figure(figsize=(10,6))
    sns.barplot(x=list(freqs), y=list(words))
    plt.title(title)
    plt.xlabel("Frekuensi")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_wordcloud(text, out_path):
    if not isinstance(text, str) or text.strip() == "":
        return
    wc = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(text)
    plt.figure(figsize=(12,6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ---------------- main processing ----------------
def preprocess_df(df, text_col):
    df = df.copy()
    df['cleaned'] = df[text_col].fillna('').astype(str).apply(clean_text)
    df['stemmed'] = df['cleaned'].apply(stem_indonesia)
    df['tokens'] = df['cleaned'].apply(tokenize)
    df['token_count'] = df['tokens'].apply(len)
    df['char_count'] = df[text_col].fillna('').astype(str).apply(len)
    return df

def process_file(input_path, output_csv, outdir, is_kaggle=True):
    print(f"\n-- Memproses file: {input_path} --")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File tidak ditemukan: {input_path}")

    df = pd.read_csv(input_path)
    if TEXT_COL not in df.columns:
        raise ValueError(f"Kolom teks '{TEXT_COL}' tidak ditemukan di file {input_path}")

    if is_kaggle and LABEL_COL not in df.columns:
        raise ValueError(f"File {input_path} diharapkan memiliki kolom label: '{LABEL_COL}'")
    if not is_kaggle and LABEL_COL not in df.columns:
        # beri -1 kalau scraped tidak punya label
        df[LABEL_COL] = -1

    df = preprocess_df(df, TEXT_COL)

    os.makedirs(outdir, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print("Saved preprocessed ->", output_csv)

    # visualisasi
    try:
        if is_kaggle:
            plot_label_distribution(df, os.path.join(outdir, "label_distribution.png"))
        plot_length_histogram(df, TEXT_COL, os.path.join(outdir, "length_histogram.png"), os.path.join(outdir, "length_boxplot.png"))
        top_all = top_n_words(df, text_col='stemmed', n=20)
        plot_top_words_bar(top_all, os.path.join(outdir, "top20_overall.png"), title="Top 20 words - Overall")
        pd.DataFrame(top_all, columns=['word','freq']).to_csv(os.path.join(outdir,'top20_overall.csv'), index=False)

        if is_kaggle:
            top_prom = top_n_words(df, text_col='stemmed', n=20, label_filter=1)
            plot_top_words_bar(top_prom, os.path.join(outdir, "top20_promosi.png"), title="Top 20 words - Promosi (label=1)")
            pd.DataFrame(top_prom, columns=['word','freq']).to_csv(os.path.join(outdir,'top20_promosi.csv'), index=False)
            prom_text = " ".join(df[df[LABEL_COL] == 1]['stemmed'].astype(str).tolist())
            plot_wordcloud(prom_text, os.path.join(outdir, "wordcloud_promosi.png"))

        # wordcloud overall
        wc_text = " ".join(df['stemmed'].astype(str).tolist())
        plot_wordcloud(wc_text, os.path.join(outdir, "wordcloud_overall.png"))
    except Exception as e:
        print("Warning: gagal membuat beberapa visualisasi ->", e)

    print(f"-- Selesai memproses {input_path}. Visual & CSV tersimpan di {outdir} --")
    return df

if __name__ == "__main__":
    print("Mulai EDA + Preprocessing untuk dataset Anda.")
    # process labeled dataset (kaggle-like)
    try:
        df_kag = process_file(KAGGLE_FILE, OUT_KAGGLE, OUTDIR_KAGGLE, is_kaggle=True)
    except Exception as e:
        print("Gagal memproses dataset berlabel:", e)
        df_kag = None

    # process scraped dataset
    try:
        df_scr = process_file(SCRAPED_FILE, OUT_SCRAPED, OUTDIR_SCRAPED, is_kaggle=False)
    except Exception as e:
        print("Gagal memproses dataset scraped:", e)
        df_scr = None

    print("Selesai EDA + Preprocessing.")
