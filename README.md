Elisabeth J. Jerau - 71220899  
Hermanus R. Horo - 7122082  

# ML Deteksi Komentar “Judol” YouTube
Proyek ini melakukan deteksi komentar promosi judi online (“judol”) pada YouTube melalui pipeline pra-pemrosesan, eksplorasi data, dan pelatihan model menggunakan kumpulan data gabungan dari YouTube dan Kaggle.

## Struktur Workspace
- [cleaning_scrap.ipynb](cleaning_scrap.ipynb) — notebook pembersihan awal hasil scraping.
- [EDA.IPYNB](EDA.IPYNB) — eksplorasi data dan distribusi label/fitur.
- [Praprocess.ipynb](Praprocess.ipynb) — normalisasi teks, penanganan emoji, dan pembuatan kolom bersih.
- [tess.ipynb](tess.ipynb) — pengujian cepat fitur/model.
- [train_kaggle.ipynb](train_kaggle.ipynb) — pelatihan model pada dataset Kaggle.
- [train_kaggle_YT.IPYNB](train_kaggle_YT.IPYNB) — pelatihan gabungan Kaggle + YouTube.
- Folder [dataset/](dataset/)
  - [dataset_preprocessed.csv](dataset/dataset_preprocessed.csv) — data hasil pra-pemrosesan.
  - [datasetKaggle.csv](dataset/datasetKaggle.csv) — data komentar sumber Kaggle.
  - [dataset_youtube_clean.csv](dataset/dataset_youtube_clean.csv) — komentar YouTube yang sudah dibersihkan.
  - [train_clean_label.csv](dataset/train_clean_label.csv) — data latih terlabel bersih.
  - [train_final.csv](dataset/train_final.csv) — gabungan final untuk training.

## Format Dataset
Label biner pada kolom terakhir:
- 1 = komentar “judol” (promosi situs/turnamen, kata kunci seperti “alexis17”, “pulau777”, “weton88”).
- 0 = komentar non-judol.

Contoh baris:
- [`dataset/datasetKaggle.csv`](dataset/datasetKaggle.csv): berisi kolom video_id, judul, channel, timestamp/user, komentar asli, label, komentar bersih, label ulang.
- [`dataset/dataset_preprocessed.csv`](dataset/dataset_preprocessed.csv): kolom teks telah dinormalisasi (emoji ke token, angka ke format konsisten).

## Alur Kerja
1. Pembersihan awal: jalankan [cleaning_scrap.ipynb](cleaning_scrap.ipynb).
2. Pra-pemrosesan: jalankan [Praprocess.ipynb](Praprocess.ipynb) untuk:
   - lowercasing, normalisasi spasi/angka,
   - konversi emoji ke token konsisten,
   - penyelarasan kolom label 0/1.
3. EDA: gunakan [EDA.IPYNB](EDA.IPYNB) untuk analisis distribusi label, kata kunci, dan outlier.
4. Pelatihan:
   - Kaggle-only: [train_kaggle.ipynb](train_kaggle.ipynb) dengan sumber [`dataset/datasetKaggle.csv`](dataset/datasetKaggle.csv).
   - Gabungan: [train_kaggle_YT.IPYNB](train_kaggle_YT.IPYNB) dengan sumber utama [`dataset/train_final.csv`](dataset/train_final.csv) dan data YouTube bersih [`dataset/dataset_youtube_clean.csv`](dataset/dataset_youtube_clean.csv).

## Kesimpulan  
