"""
training_manual_svm.py

Training linear SVM implemented manually (SGD on hinge loss + L2 reg).
Does NOT use sklearn.svm. Uses TfidfVectorizer for feature extraction.

CONFIG: edit TRAIN_FILE and LABEL_COL as needed.
"""

# ============== CONFIG ==============
TRAIN_FILE = "balanced_dataset_undersample.csv"   # path to CSV
LABEL_COL = "label"      # name of label column (0/1)
OUT_DIR = "training_outputs_manualsvm"
RANDOM_STATE = 42

# Manual SVM hyperparams
EPOCHS = 20
LEARNING_RATE = 0.5
BATCH_SIZE = 256
L2_REG = 1e-4   # lambda (regularization strength)
CLASS_WEIGHT = "balanced"  # "balanced" or None
TOP_K_FEATURES = 100
# =====================================

import os, sys, json, re
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

# optional nlp libs
try:
    from unidecode import unidecode
    import emoji
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    stemmer = StemmerFactory().create_stemmer()
    SASTRAWI = True
except Exception:
    SASTRAWI = False

# ---------------- text helpers ----------------
def simple_clean(text):
    if not isinstance(text, str):
        return ""
    try:
        text = unidecode(text)
    except:
        pass
    try:
        text = emoji.demojize(text, delimiters=(" ", " "))
    except:
        pass
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#\w+', ' ', text)
    text = re.sub(r'[^0-9a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def maybe_stem(text):
    if not isinstance(text, str) or text.strip() == "":
        return ""
    if SASTRAWI:
        try:
            return stemmer.stem(text)
        except:
            return text
    return text

def ensure_text_column(df):
    candidates = ['stemmed', 'komentar_clean', 'komentar', 'comment', 'text', 'text_clean']
    for c in candidates:
        if c in df.columns:
            return df, c
    for col in df.columns:
        low = col.lower()
        if 'komentar' in low or 'comment' in low or 'kom' in low:
            return df, col
    raise ValueError("Tidak menemukan kolom teks. Set TEXT_COL or ensure column exists.")

# ---------------- manual linear SVM class ----------------
class ManualLinearSVM:
    def __init__(self, n_features, lr=0.1, l2=1e-4, random_state=0):
        rng = np.random.RandomState(random_state)
        # initialize weights small
        self.w = rng.normal(scale=0.01, size=(n_features, ))
        self.b = 0.0
        self.lr = lr
        self.l2 = l2

    def decision_function(self, X):
        # X: array shape (n_samples, n_features) or sparse matrix
        if hasattr(X, "dot"):
            return X.dot(self.w) + self.b
        else:
            return np.dot(X, self.w) + self.b

    def predict(self, X):
        scores = self.decision_function(X)
        return (scores >= 0).astype(int)  # returns 0/1

    def fit(self, X, y, epochs=10, batch_size=128, sample_weights=None, verbose=True):
        """
        SGD on hinge loss:
        L = (1/n) * sum(max(0, 1 - y_i * (w·x_i + b))) + 0.5 * l2 * ||w||^2
        y must be in {-1, +1}
        sample_weights: array of length n to weight samples (optional)
        """
        n_samples = X.shape[0]
        n_features = X.shape[1]
        if sample_weights is None:
            sample_weights = np.ones(n_samples, dtype=float)

        for epoch in range(epochs):
            # shuffle
            idx = np.arange(n_samples)
            np.random.shuffle(idx)
            # mini-batch loop
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = idx[start:end]
                Xb = X[batch_idx]
                yb = y[batch_idx]
                sw = sample_weights[batch_idx]

                # compute margins: y * (w·x + b)
                if hasattr(Xb, "toarray") and not isinstance(Xb, np.ndarray):
                    # sparse
                    margins = yb * (Xb.dot(self.w) + self.b)
                else:
                    margins = yb * (np.dot(Xb, self.w) + self.b)

                # indicator for hinge loss active
                mask = margins < 1  # shape (batch,)
                if mask.sum() == 0:
                    # only regularization gradient
                    grad_w = self.l2 * self.w
                    grad_b = 0.0
                else:
                    # compute gradient for hinge loss part:
                    # grad_w_hinge = - (1/|batch|) * sum_i (w_i) ??? Actually:
                    # For hinge: gradient = - sum_i (y_i * x_i * 1) for those with margin < 1
                    # incorporating sample weights:
                    if hasattr(Xb, "dot") and not isinstance(Xb, np.ndarray):
                        # sparse handling: compute weighted sums
                        y_mask = yb[mask] * sw[mask]
                        X_mask = Xb[mask]
                        grad_w_hinge = - (X_mask.T).dot(y_mask) / batch_idx.shape[0]
                    else:
                        y_mask = (yb[mask] * sw[mask]).reshape(-1,1)  # shape (m,1)
                        X_mask = Xb[mask]  # (m, n_features)
                        grad_w_hinge = - np.sum(X_mask * y_mask, axis=0) / batch_idx.shape[0]
                    # bias gradient
                    grad_b = - np.sum(yb[mask] * sw[mask]) / batch_idx.shape[0]

                    # total grad with regularization
                    grad_w = grad_w_hinge + self.l2 * self.w

                # update
                self.w -= self.lr * grad_w
                self.b -= self.lr * grad_b
            # optionally decay lr a bit
            if verbose:
                # compute epoch loss (approx) on whole set occasionally
                if (epoch % max(1, epochs//5) == 0) or epoch == epochs-1:
                    scores = self.decision_function(X)
                    margins_all = y * scores
                    hinge_losses = np.maximum(0, 1 - margins_all)
                    loss = hinge_losses.mean() + 0.5 * self.l2 * np.dot(self.w, self.w)
                    print(f"Epoch {epoch+1}/{epochs} — loss {loss:.6f}")
        return self

# ---------------- plotting helpers ----------------
def save_grouped_performance_plot(metrics_dict, out_png):
    if isinstance(metrics_dict, dict):
        dfm = pd.DataFrame([metrics_dict])
        dfm['model'] = ['Model']
    else:
        dfm = pd.DataFrame(metrics_dict)
        if 'model' not in dfm.columns:
            dfm['model'] = [f"Model{i+1}" for i in range(len(dfm))]
    metrics = ['accuracy','precision','recall','f1']
    models = dfm['model'].tolist()
    n_models = len(models)
    vals = np.array([[dfm.loc[i, m] for i in range(n_models)] for m in metrics])
    x = np.arange(len(metrics))
    total_width = 0.6
    bar_width = total_width / max(1, n_models)
    plt.figure(figsize=(10,6))
    ax = plt.gca()
    palette = sns.color_palette("tab10", n_models)
    for idx, model_name in enumerate(models):
        offset = (idx - (n_models-1)/2) * bar_width
        pos = x + offset
        bars = ax.bar(pos, vals[:, idx], width=bar_width*0.9, color=palette[idx % len(palette)], label=model_name)
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h+0.005, f"{h:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics], fontsize=11)
    ax.set_ylim(0,1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance")
    ax.legend(title="Model")
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

# ---------------- main training flow ----------------
def main():
    print("=== Manual Linear SVM Training (no sklearn.svm) ===")
    if not os.path.exists(TRAIN_FILE):
        print("ERROR: TRAIN_FILE not found:", TRAIN_FILE)
        sys.exit(1)

    df = pd.read_csv(TRAIN_FILE)
    print("Loaded:", TRAIN_FILE, "shape:", df.shape)

    if LABEL_COL not in df.columns:
        print(f"ERROR: label column '{LABEL_COL}' missing.")
        sys.exit(1)

    # detect text column
    try:
        df, text_col = ensure_text_column(df)
        print("Using text column:", text_col)
    except Exception as e:
        print("ERROR detecting text column:", e)
        sys.exit(1)

    # prepare text_for_model
    df['text_for_model'] = df[text_col].astype(str).fillna('').apply(simple_clean).apply(maybe_stem)

    # filter valid labels 0/1
    df = df[df[LABEL_COL].isin([0,1])].copy()
    if df.empty:
        print("ERROR: no rows with label 0/1.")
        sys.exit(1)

    X_text = df['text_for_model'].values
    y0 = df[LABEL_COL].astype(int).values
    print("Usable samples:", len(X_text))

    # train/test split
    X_train_text, X_test_text, y_train0, y_test0 = train_test_split(
        X_text, y0, test_size=0.2, random_state=RANDOM_STATE, stratify=y0
    )

    # Vectorize text -> TF-IDF
    vect = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9)
    X_train = vect.fit_transform(X_train_text)  # sparse matrix
    X_test = vect.transform(X_test_text)

    # convert labels to -1 / +1 for hinge
    y_train = (y_train0 * 2) - 1
    y_test = (y_test0 * 2) - 1

    # compute sample_weights if balanced
    sample_weights = None
    if CLASS_WEIGHT == "balanced":
        # inverse frequency
        unique, counts = np.unique(y_train0, return_counts=True)
        freq = dict(zip(unique, counts))
        # map 0->weight,1->weight
        w0 = 1.0 / freq.get(0,1)
        w1 = 1.0 / freq.get(1,1)
        # normalize so mean weight = 1
        meanw = (w0*freq.get(0,0) + w1*freq.get(1,0)) / (freq.get(0,0)+freq.get(1,0))
        w0 = w0 / meanw
        w1 = w1 / meanw
        sample_weights = np.array([w1 if yy==1 else w0 for yy in y_train0])  # note y_train0 is 0/1

    # instantiate manual SVM
    n_features = X_train.shape[1]
    svm = ManualLinearSVM(n_features=n_features, lr=LEARNING_RATE, l2=L2_REG, random_state=RANDOM_STATE)

    # fit (handles sparse matrices)
    print("Start training manual SVM (epochs=%d, batch_size=%d)..." % (EPOCHS, BATCH_SIZE))
    svm.fit(X_train, (y_train), epochs=EPOCHS, batch_size=BATCH_SIZE, sample_weights=sample_weights, verbose=True)
    print("Training finished.")

    # predictions
    y_pred_test = svm.predict(X_test)  # returns 0/1
    # convert y_test back to 0/1 for metrics
    y_test01 = (y_test + 1) // 2

    # decision scores
    try:
        scores = svm.decision_function(X_test)
    except Exception:
        scores = None

    # metrics
    acc = accuracy_score(y_test01, y_pred_test)
    prec = precision_score(y_test01, y_pred_test, zero_division=0)
    rec = recall_score(y_test01, y_pred_test, zero_division=0)
    f1 = f1_score(y_test01, y_pred_test, zero_division=0)
    creport = classification_report(y_test01, y_pred_test, output_dict=True)
    cm = confusion_matrix(y_test01, y_pred_test)

    # prepare outputs
    os.makedirs(OUT_DIR, exist_ok=True)

    # Save model (vectorizer + weights + bias) as a dict
    model_obj = {
        'vectorizer': vect,
        'w': svm.w,
        'b': svm.b,
        'meta': {
            'type': 'manual_linear_svm',
            'epochs': EPOCHS,
            'lr': LEARNING_RATE,
            'l2': L2_REG
        }
    }
    joblib.dump(model_obj, os.path.join(OUT_DIR, "model_manual_svm.joblib"))
    print("Saved model ->", os.path.join(OUT_DIR, "model_manual_svm.joblib"))

    # save classification report json/csv
    cr_json = os.path.join(OUT_DIR, "classification_report.json")
    cr_csv = os.path.join(OUT_DIR, "classification_report.csv")
    with open(cr_json, "w", encoding="utf8") as f:
        json.dump(creport, f, indent=2, ensure_ascii=False)
    pd.DataFrame(creport).transpose().to_csv(cr_csv, index=True)
    print("Saved classification report ->", cr_json, cr_csv)

    # save summary metrics
    summary = {'model': 'manual_svm', 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
               'n_train': X_train.shape[0], 'n_test': X_test.shape[0]}
    pd.DataFrame([summary]).to_csv(os.path.join(OUT_DIR, "summary_metrics.csv"), index=False)

    # confusion matrix plot
    cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
    plt.tight_layout(); plt.savefig(cm_path); plt.close()

    # ROC & PR
    try:
        roc_path = os.path.join(OUT_DIR, "roc_curve.png")
        fpr, tpr, _ = roc_curve(y_test01, scores)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
        plt.plot([0,1],[0,1],'--', color='gray')
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curve')
        plt.legend(loc='lower right'); plt.tight_layout(); plt.savefig(roc_path); plt.close()
    except Exception:
        pass

    try:
        pr_path = os.path.join(OUT_DIR, "pr_curve.png")
        precision_vals, recall_vals, _ = precision_recall_curve(y_test01, scores)
        ap = average_precision_score(y_test01, scores)
        plt.figure(figsize=(6,5))
        plt.plot(recall_vals, precision_vals, label=f'AP = {ap:.4f}')
        plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left'); plt.tight_layout(); plt.savefig(pr_path); plt.close()
    except Exception:
        pass

    # extract top features by weight
    try:
        feat_names = vect.get_feature_names_out()
        coef = svm.w
        coef_df = pd.DataFrame({'feature': feat_names, 'coef': coef})
        coef_df = coef_df.sort_values('coef', ascending=False)
        coef_df.head(TOP_K_FEATURES).to_csv(os.path.join(OUT_DIR, f"top_positive_features.csv"), index=False)
        coef_df.tail(TOP_K_FEATURES).to_csv(os.path.join(OUT_DIR, f"top_negative_features.csv"), index=False)
    except Exception as e:
        print("Could not extract feature names:", e)

    # performance plot
    perf_plot_path = os.path.join(OUT_DIR, "performance_comparison.png")
    perf_metrics = {'model': 'manual_svm', 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
    save_grouped_performance_plot([perf_metrics], perf_plot_path)

    # print summary
    print("\n=== Summary Metrics ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("Outputs saved in:", OUT_DIR)
    print("Done.")

if __name__ == "__main__":
    main()
