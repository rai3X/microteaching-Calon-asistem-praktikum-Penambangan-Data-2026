# Studi Kasus: Prediksi Tingkat Kepuasan Pengguna Aplikasi Pembelajaran Daring

## 📋 Deskripsi Proyek

Proyek ini merupakan implementasi studi kasus prediksi tingkat kepuasan pengguna aplikasi pembelajaran daring menggunakan metodologi CRISP-DM (Cross-Industry Standard Process for Data Mining). Proyek ini menggunakan 4 algoritma machine learning untuk klasifikasi biner.

## 🎯 Tujuan

1. Menerapkan tahapan CRISP-DM dalam menyelesaikan permasalahan klasifikasi
2. Melakukan eksplorasi dan pemahaman terhadap dataset survei pengguna
3. Melakukan preprocessing data agar siap digunakan dalam pemodelan
4. Membangun model klasifikasi untuk memprediksi tingkat kepuasan pengguna
5. Mengevaluasi performa model menggunakan metrik evaluasi klasifikasi yang sesuai
6. Membuat dashboard interaktif untuk deployment

## 🔬 Metodologi CRISP-DM

### 1. Data Understanding
- Eksplorasi struktur dataset
- Analisis missing values dan data duplikat
- Visualisasi distribusi data
- Analisis korelasi antar fitur

### 2. Data Preprocessing
- Penanganan missing values (imputasi median)
- Penghapusan data duplikat
- Encoding data kategorikal
- Feature scaling
- Train-test split

### 3. Modelling
Menggunakan 4 algoritma klasifikasi:
- **Logistic Regression** (Baseline Model)
- **Naive Bayes** (Probabilistic Model)
- **Decision Tree** (Non-linear Model)
- **KNN** (Distance-based Model)

### 4. Evaluation
Evaluasi menggunakan metrik:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- Classification Report

### 5. Deployment
Dashboard interaktif menggunakan Streamlit untuk prediksi data baru

## 📁 Struktur Proyek

```
.
├── dataset_kepuasan_pengguna_elearning.csv         # Dataset survei pengguna
├── Technical_Test_Datmin_Terstruktur.ipynb        # Notebook analisis CRISP-DM & model training
├── app.py                                         # Dashboard Streamlit (deployment)
├── requirements.txt                               # Daftar dependencies Python
├── README.md                                      # Dokumentasi proyek
├── logistic_regression.joblib                     # Model Logistic Regression (terlatih)
├── naive_bayes.joblib                            # Model Naive Bayes (terlatih)
├── decision_tree.joblib                          # Model Decision Tree (terlatih)
├── knn.joblib                                    # Model KNN (terlatih)
└── scaler.joblib                                 # StandardScaler untuk preprocessing
```

## 🚀 Cara Menjalankan

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Jalankan Notebook

Buka `Technical_Test_Datmin_Terstruktur.ipynb` di Jupyter Notebook atau JupyterLab dan jalankan semua sel untuk:
- Melakukan analisis data
- Membangun dan melatih model
- Mengevaluasi performa model
- Menyimpan model yang telah dilatih

### 3. Jalankan Dashboard Streamlit

```bash
streamlit run app.py
```

Dashboard akan terbuka di browser pada `http://localhost:8501`

## 📊 Dataset

Dataset berisi data survei pengguna aplikasi pembelajaran daring dengan atribut:
- `id_responden`: Kode unik responden
- `usia`: Usia pengguna (tahun)
- `jenis_kelamin`: Jenis kelamin (L/P)
- `durasi_penggunaan`: Rata-rata durasi penggunaan per hari (jam)
- `frekuensi_login`: Frekuensi login per minggu
- `kualitas_materi`: Penilaian kualitas materi (1-5)
- `kemudahan_penggunaan`: Penilaian kemudahan penggunaan (1-5)
- `stabilitas_aplikasi`: Penilaian stabilitas aplikasi (1-5)
- `interaksi_pengajar`: Penilaian interaksi dengan pengajar (1-5)
- `kepuasan_pengguna`: Target klasifikasi (0: Tidak Puas, 1: Puas)

## 🛠️ Teknologi yang Digunakan

- **Python 3.8+**
- **Pandas**: Manipulasi data
- **NumPy**: Komputasi numerik
- **Scikit-learn**: Machine learning
- **Matplotlib & Seaborn**: Visualisasi
- **Streamlit**: Dashboard interaktif
- **Joblib**: Penyimpanan model

## 📈 Hasil Evaluasi

Setelah menjalankan notebook, akan diperoleh hasil evaluasi untuk setiap model dengan berbagai metrik. Model terbaik dapat dipilih berdasarkan performa keseluruhan.

## 👤 Author

Dibuat untuk Studi Kasus Penambangan Data - DASPRO Laboratory

## 📝 License

Proyek ini dibuat untuk keperluan akademik.

