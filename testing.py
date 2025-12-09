"""
testing_manual_svm.py

COMPATIBLE WITH:
- Manual SVM model (vectorizer + weights + bias)
- Sklearn pipeline or estimator (fallback)

FEATURES:
- Load model (joblib)
- Load scraped CSV
- Auto-detect text column
- Clean + stem text
- Predict label (0/1)
- Add label_keterangan
- Save output CSV

"""

import os
import sys
import re
import json
import pandas as pd
import numpy as np
import joblib

# Optional libs
try:
    from unidecode import unidecode
except:
    unidecode = None

try:
    import emoji
except:
    emoji = None

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    stemmer = StemmerFactory().create_stemmer()
except:
    stemmer = None


# ================================================================
#  TEXT CLEANING FUNCTIONS
# ================================================================
def simple_clean(text):
    if not isinstance(text, str):
        return ""
    t = text
    if unidecode:
        try:
            t = unidecode(t)
        except:
            pass
    if emoji:
        try:
            t = emoji.demojize(t, delimiters=(" ", " "))
        except:
            pass
    t = t.lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"#\w+", " ", t)
    t = re.sub(r"[^0-9a-zA-Z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def maybe_stem(text):
    if not isinstance(text, str) or text.strip() == "":
        return ""
    if stemmer:
        try:
            return stemmer.stem(text)
        except:
            return text
    return text


# ================================================================
#  MANUAL SVM WRAPPER (for model_manual_svm.joblib)
# ================================================================
class ManualSVMWrapper:
    def __init__(self, vectorizer, w, b):
        self.vectorizer = vectorizer
        self.w = w
        self.b = b

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def decision_function(self, X):
        if hasattr(X, "dot"):
            return X.dot(self.w) + self.b
        return np.dot(X, self.w) + self.b

    def predict(self, texts):
        X = self.transform(texts)
        scores = self.decision_function(X)
        return (scores >= 0).astype(int)


# ================================================================
#  MODEL LOADER (detect manual SVM / sklearn model)
# ================================================================
def load_model(model_path):
    obj = joblib.load(model_path)

    # Case 1: dictionary (manual model)
    if isinstance(obj, dict):
        if "vectorizer" in obj and "w" in obj and "b" in obj:
            print("Detected MANUAL SVM model (vectorizer + w + b)")
            model = ManualSVMWrapper(obj["vectorizer"], obj["w"], obj["b"])
            return model, obj["vectorizer"]

        print("ERROR: Dict loaded but no manual SVM keys found.")
        sys.exit(1)

    # Case 2: sklearn model or pipeline
    if hasattr(obj, "predict"):
        print("Detected sklearn-like model (pipeline/estimator)")
        return obj, None

    print("ERROR: Model type not recognized.")
    sys.exit(1)


# ================================================================
#  AUTO DETECT TEXT COLUMN
# ================================================================
def detect_text_column(df):
    candidates = ["text_clean", "stemmed", "komentar_clean", "komentar", "comment", "text"]
    for c in candidates:
        if c in df.columns:
            return c

    # fallback
    for col in df.columns:
        l = col.lower()
        if "komentar" in l or "comment" in l:
            return col

    raise ValueError("Tidak menemukan kolom teks pada dataset scraped!")


# ================================================================
#  MAIN TESTING FUNCTION
# ================================================================
def main():

    MODEL_PATH = "training_outputs_manualsvm/model_manual_svm.joblib"
    SCRAPED_FILE = "dataset/youtube_comments_judol.csv"
    OUTPUT_DIR = "test_outputs"

    print("=== TESTING MANUAL SVM MODEL ===")

    if not os.path.exists(MODEL_PATH):
        print("ERROR: Model tidak ditemukan:", MODEL_PATH)
        sys.exit(1)

    if not os.path.exists(SCRAPED_FILE):
        print("ERROR: File scraped tidak ditemukan:", SCRAPED_FILE)
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model
    model, vect = load_model(MODEL_PATH)

    # Load scraped dataset
    df = pd.read_csv(SCRAPED_FILE)
    print("Loaded scraped:", df.shape)

    # Detect column
    text_col = detect_text_column(df)
    print("Detected text column:", text_col)

    # Clean + stem
    df["text_clean"] = df[text_col].astype(str).fillna("").apply(simple_clean).apply(maybe_stem)

    # Predict
    try:
        preds = model.predict(df["text_clean"].values)
    except Exception as e:
        print("ERROR saat memprediksi:", e)
        sys.exit(1)

    # Decision scores
    try:
        scores = model.decision_function(model.transform(df["text_clean"].values))
    except:
        scores = None

    df["predicted_label"] = preds.astype(int)
    df["label"] = df["predicted_label"]   # untuk retraining
    df["label_keterangan"] = df["predicted_label"].apply(
        lambda x: "iklan judol" if x == 1 else "bukan iklan judol"
    )

    if scores is not None:
        df["predicted_score"] = scores

    # Save output
    out_path = os.path.join(OUTPUT_DIR, "predicted_scraped_dataset.csv")
    df.to_csv(out_path, index=False)

    print("\n=== DONE ===")
    print("Output saved to:", out_path)
    print("Total predicted:", len(df))


if __name__ == "__main__":
    main()
