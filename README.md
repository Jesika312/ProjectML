# ML-JESIL: Text Classification with SVM

Proyek machine learning untuk klasifikasi teks dalam bahasa Indonesia menggunakan Support Vector Machine (SVM). Proyek ini dirancang untuk menganalisis komentar YouTube dan melakukan prediksi label berdasarkan model yang dilatih.

---

## 📋 Daftar Isi
1. [Deskripsi Proyek](#deskripsi-proyek)
2. [Struktur Direktori](#struktur-direktori)
3. [Persyaratan Sistem](#persyaratan-sistem)
4. [Instalasi](#instalasi)
5. [Alur Kerja](#alur-kerja)
6. [Cara Menjalankan](#cara-menjalankan)
7. [File Output](#file-output)
8. [Konfigurasi](#konfigurasi)

---

## 🎯 Deskripsi Proyek

Proyek ini mengimplementasikan pipeline lengkap untuk text classification:

- **Input**: Komentar YouTube dalam bahasa Indonesia (berlabel dan tidak berlabel)
- **Proses**: EDA, preprocessing, balancing dataset, training model SVM
- **Output**: Model terlatih + prediksi untuk dataset baru

Teknologi yang digunakan:
- **Framework**: scikit-learn (SVM, TF-IDF)
- **NLP**: Sastrawi (stemmer Indonesia), Unidecode, Emoji handling
- **Data Processing**: Pandas, NumPy
- **Visualisasi**: Matplotlib, Seaborn, WordCloud

---

## 📁 Struktur Direktori

```
ml-jesil/
├── README.md                                  # Dokumentasi proyek (File ini)
├── eda_preprocessing.py                       # Script EDA & preprocessing
├── balance_dataset.py                         # Script balancing dataset
├── training.py                                # Script training model
├── testing.py                                 # Script testing/prediksi
│
├── dataset/                                   # Folder input data
│   ├── data_komentar_dengan_prediksi - 
│   │   data_komentar_dengan_prediksi(2).csv  # Dataset berlabel (Kaggle-like) ✓
│   └── youtube_comments_judol.csv            # Dataset scraped (tanpa label) ✓
│
├── preprocessed_kaggle.csv                    # Data berlabel setelah preprocessing ✓
├── balanced_dataset_undersample.csv           # Data balanced untuk training ✓
│
├── training_outputs/                          # Output dari training ✓
│   ├── svm_model_training_only.joblib        # Model SVM terlatih ✓
│   ├── classification_report.json            # Report dalam format JSON ✓
│   ├── classification_report.csv             # Report dalam format CSV ✓
│   ├── summary_metrics.csv                   # Ringkasan metrik performa ✓
│   ├── confusion_matrix.png                  # Confusion matrix heatmap ✓
│   ├── roc_curve.png                         # ROC curve plot ✓
│   ├── pr_curve.png                          # Precision-Recall curve ✓
│   ├── performance_comparison.png            # Grouped bar chart metrics ✓
│   ├── top_positive_features.csv             # Top 20 fitur positif ✓
│   └── top_negative_features.csv             # Top 20 fitur negatif ✓
│
├── outputs_kaggle/                            # Visualisasi EDA (dataset berlabel) ✓
│   ├── label_distribution.png                # Distribusi label chart ✓
│   ├── length_boxplot.png                    # Boxplot panjang teks ✓
│   ├── length_histogram.png                  # Histogram panjang teks ✓
│   ├── wordcloud_overall.png                 # Word cloud keseluruhan ✓
│   ├── wordcloud_promosi.png                 # Word cloud per label ✓
│   ├── top20_overall.csv                     # Top 20 kata keseluruhan ✓
│   ├── top20_overall.png                     # Bar chart top 20 kata ✓
│   ├── top20_promosi.csv                     # Top 20 kata per label ✓
│   └── top20_promosi.png                     # Bar chart top 20 kata per label ✓
│
└── test_outputs/                              # Output dari testing ✓
      └── predicted_scraped_dataset.csv         # Data + prediksi + score ✓
```

---

## 🔧 Persyaratan Sistem

- **Python**: 3.8 atau lebih baru
- **OS**: Windows, macOS, atau Linux
- **RAM**: Minimal 4GB
- **Disk Space**: 500MB (untuk dataset dan model)

---

## 📦 Instalasi

### 1. Clone atau Download Proyek

```bash
cd e:\ml-jesil
```

### 2. Install Dependencies

Jalankan perintah berikut untuk install semua package yang diperlukan:

```bash
pip install pandas numpy scikit-learn joblib matplotlib seaborn wordcloud unidecode emoji sastrawi
```

**Penjelasan package:**
- `pandas`, `numpy`: Data manipulation
- `scikit-learn`: ML algorithms (SVM, TF-IDF, metrics)
- `joblib`: Model serialization
- `matplotlib`, `seaborn`: Plotting
- `wordcloud`: Visualisasi word frequency
- `unidecode`: Unicode normalization
- `emoji`: Emoji processing
- `sastrawi`: Stemmer untuk Bahasa Indonesia

### 3. Verifikasi Instalasi

```bash
python -c "import pandas, sklearn, joblib; print('✓ Dependencies OK')"
```

---

## 🔄 Alur Kerja

### Fase 1: Exploratory Data Analysis (EDA) & Preprocessing
**Script**: `eda_preprocessing.py`

```
Dataset Raw (CSV)
        ↓
    [EDA]
    - Analisis distribusi label
    - Visualisasi word frequency
    - Deteksi missing values
        ↓
  [Preprocessing]
  - Lowercase
  - Remove URLs, mentions, hashtags
  - Remove special characters
  - Stemming (Sastrawi)
  - Remove emoji
        ↓
Preprocessed CSV (siap training)
```

**Input**: 
- Dataset berlabel: `dataset/data_komentar_dengan_prediksi - data_komentar_dengan_prediksi(2).csv`
- Dataset scraped (optional): `dataset/youtube_comments_judol.csv`

**Output**:
- `preprocessed_kaggle.csv` (data berlabel setelah preprocessing)
- `outputs_kaggle/` (visualisasi EDA)

---

### Fase 2: Dataset Balancing
**Script**: `balance_dataset.py`

```
Preprocessed Data (mungkin imbalanced)
        ↓
  [Undersample]
  - Identifikasi majority & minority class
  - Undersample majority class ke jumlah minority
  - Ratio akhir: 50:50
        ↓
Balanced Dataset (siap training)
```

**Input**: `preprocessed_kaggle.csv`

**Output**: `balanced_dataset_undersample.csv`

---

### Fase 3: Model Training
**Script**: `training.py`

```
Balanced Dataset
        ↓
  [Train-Test Split]
  - Train: 80%
  - Test: 20%
        ↓
  [Pipeline]
  ├─ TF-IDF Vectorizer
  └─ Linear SVC
        ↓
  [Training]
  - Fit model pada training data
        ↓
  [Evaluation]
  - Prediksi pada test data
  - Hitung: Accuracy, Precision, Recall, F1
  - Generate: Confusion Matrix, ROC Curve
  - Extract: Top features
        ↓
Model + Reports (dalam training_outputs/)
```

**Input**: `balanced_dataset_undersample.csv`

**Outputs**:
- `svm_model_training_only.joblib` (model terlatih)
- Classification report (JSON & CSV)
- Performance plots (PNG)
- Top features (CSV)

---

### Fase 4: Testing & Prediksi
**Script**: `testing.py`

```
Model + Dataset Baru (tanpa label)
        ↓
  [Detect Text Column]
  - Otomatis cari kolom teks ("komentar", "comment", dll)
        ↓
  [Preprocessing]
  - Sama seperti fase 1
        ↓
  [Predict]
  - Gunakan model untuk prediksi
  - Hitung prediction score
        ↓
CSV dengan Prediksi
```

**Input**: `dataset/youtube_comments_judol.csv` (atau file lainnya)

**Output**: `test_outputs/predicted_scraped_dataset.csv`

---

## 🚀 Cara Menjalankan

### Opsi 1: Jalankan Semua Script Secara Berurutan

#### Langkah 1: EDA & Preprocessing
```bash
python eda_preprocessing.py
```
✅ Output: `preprocessed_kaggle.csv`, `outputs_kaggle/`

#### Langkah 2: Balancing Dataset
```bash
python balance_dataset.py
```
✅ Output: `balanced_dataset_undersample.csv`

#### Langkah 3: Training Model
```bash
python training.py
```
✅ Output: Model + reports dalam `training_outputs/`

#### Langkah 4: Testing & Prediksi
```bash
python testing.py
```
✅ Output: `test_outputs/predicted_scraped_dataset.csv`

---

### Opsi 2: Jalankan Hanya Testing (Jika Model Sudah Ada)

Jika model `svm_model_training_only.joblib` sudah tersedia di `training_outputs/`:

```bash
python testing.py
```

Script ini akan:
- Otomatis load model
- Detect kolom teks di file scraped
- Melakukan preprocessing
- Generate prediksi
- Simpan hasil ke `test_outputs/`

---

## 📊 File Output

### Dari Training

| File | Deskripsi |
|------|-----------|

### Dari Testing

| File | Deskripsi |
|------|-----------|
| `predicted_scraped_dataset.csv` | Data scraped + kolom `predicted_label` + `predicted_score` |
| File | Deskripsi |
|------|-----------|
| `confusion_matrix.png` | Heatmap confusion matrix ✓ |
| `roc_curve.png` | ROC curve plot ✓ |
| `pr_curve.png` | Precision-Recall curve ✓ |
| `performance_comparison.png` | Bar chart perbandingan metrik ✓ |
| `classification_report.json` | Metrics dalam format JSON ✓ |
| `classification_report.csv` | Metrics dalam format CSV ✓ |
| `summary_metrics.csv` | Ringkasan: Accuracy, Precision, Recall, F1 ✓ |
| `top_positive_features.csv` | Top 20 fitur dengan koefisien positif terbesar ✓ |
| `top_negative_features.csv` | Top 20 fitur dengan koefisien negatif terbesar ✓ |
| `svm_model_training_only.joblib` | Model SVM terlatih (binary format) ✓ |
| `predicted_scraped_dataset.csv` | Data scraped + kolom `predicted_label` + `predicted_score` ✓ |

---

## ⚙️ Konfigurasi

Setiap script memiliki section `CONFIG` di awal file untuk kustomisasi:

### `eda_preprocessing.py`
```python
KAGGLE_FILE   = "dataset/data_komentar_dengan_prediksi - data_komentar_dengan_prediksi(2).csv"
SCRAPED_FILE  = None  # Set ke path file jika ingin preprocess dataset scraped
TEXT_COL      = "komentar"   # Nama kolom teks
LABEL_COL     = "label"      # Nama kolom label
OUT_KAGGLE    = "preprocessed_kaggle.csv"
OUT_SCRAPED   = "preprocessed_scraped.csv"
OUTDIR_KAGGLE = "outputs_kaggle"
OUTDIR_SCRAPED= "outputs_scraped"
```

### `balance_dataset.py`
```python
INPUT_FILE = "preprocessed_kaggle.csv"
OUTPUT_FILE = "balanced_dataset_undersample.csv"
LABEL_COL = "label"
RANDOM_STATE = 42
```

### `training.py`
```python
TRAIN_FILE = "balanced_dataset_undersample.csv"
LABEL_COL = "label"
OUT_DIR = "training_outputs"
RANDOM_STATE = 42
```

### `testing.py`
```python
MODEL_PATH = "training_outputs/svm_model_training_only.joblib"
SCRAPED_FILE = "dataset/youtube_comments_judol.csv"
TEXT_COLUMN = None  # Set ke nama kolom jika ingin memaksa (misal "komentar")
OUTPUT_DIR = "test_outputs"
```

---

## 🔍 Troubleshooting

### Masalah: `ModuleNotFoundError: No module named 'sklearn'`
**Solusi**: Install scikit-learn
```bash
pip install scikit-learn
```

### Masalah: `FileNotFoundError: [Errno 2] No such file or directory`
**Solusi**: Pastikan file input ada di lokasi yang benar. Cek konfigurasi di awal script.

### Masalah: Kolom teks tidak terdeteksi di `testing.py`
**Solusi**: Set `TEXT_COLUMN` secara manual dengan nama kolom yang tepat di config.

### Masalah: Model tidak ditemukan saat testing
**Solusi**: Jalankan `training.py` terlebih dahulu untuk generate model.

### Masalah: Dataset imbalanced di fase 3
**Solusi**: Pastikan sudah menjalankan `balance_dataset.py` di fase 2.

---

## 📈 Metrics Performa

Model akan menghasilkan metrik berikut:

- **Accuracy**: Persentase prediksi benar dari total prediksi
- **Precision**: Dari prediksi positif, berapa yang benar-benar positif
- **Recall**: Dari positif asli, berapa yang terdeteksi oleh model
- **F1-Score**: Harmonic mean dari Precision dan Recall
- **ROC-AUC**: Area Under Curve dari ROC curve
- **PR-AUC**: Area Under Curve dari Precision-Recall curve

---

## 🛠️ Tips & Best Practices

1. **Selalu preview data baru**: Pastikan format CSV sesuai sebelum testing
2. **Jaga konsistensi nama kolom**: Gunakan nama yang konsisten di semua script
3. **Check log output**: Setiap script menampilkan progress dan error messages
4. **Backup model**: Simpan model penting sebelum melakukan training ulang
5. **Monitor file size**: Dataset besar dapat memperlambat processing

---

## 📞 Support

Jika ada pertanyaan atau issue, periksa:
- Console output messages
- File `classification_report.json` untuk detail metrics
- Pastikan semua dependencies terinstall dengan benar

---

## 📝 Lisensi

Proyek ini dibuat untuk keperluan pembelajaran dan analisis teks dalam Bahasa Indonesia.

---

**Last Updated**: Desember 2025
