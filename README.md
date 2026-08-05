# StuntBot — Sistem Tanya Jawab RAG Seputar Stunting

Repositori ini berisi implementasi antarmuka dan *serving pipeline* untuk penelitian skripsi berjudul **"Implementasi dan Analisis Kinerja Retrieval-Augmented Generation (RAG) Menggunakan Model Qwen 2.5 7B Instruct dalam Sistem Tanya Jawab Seputar Stunting"** di Program Studi Teknik Informatika, Universitas Komputer Indonesia (UNIKOM).

Repositori ini difokuskan pada modul *serving/inference*: proses *retrieval* dokumen, *generation* jawaban berbasis LLM lokal, serta *user interface* berbasis Gradio. Proses pra-pemrosesan korpus (ekstraksi, *chunking*, penyusunan indeks), sintesis dataset evaluasi, dan evaluasi kuantitatif menggunakan RAGAS dilakukan pada *notebook* terpisah.

---

## Alur Kerja Sistem (Pipeline)

1. **Query Processing & Embedding**: Pertanyaan dari pengguna diformat menggunakan pemodelan kueri `intfloat/multilingual-e5-base` (ditambahkan prefiks `query:` jika belum ada).
2. **Vector Retrieval**: Dokumen relevan diambil dari indeks FAISS lokal menggunakan metode *Maximal Marginal Relevance* (MMR) dengan parameter:
   - $k = 5$ (jumlah dokumen akhir yang diambil)
   - $fetch\_k = 30$ (kandidat awal yang diperiksa)
   - $\lambda_{mult} = 0.85$ (bobot keberagaman vs relevansi)
3. **Context Formatting**: Dokumen hasil *retrieval* dirangkai beserta metadata (judul, sumber, tahun) ke dalam *prompt template*.
4. **Generation**: Model **Qwen 2.5 7B Instruct** dimuat secara lokal (*Hugging Face Transformers*) dengan kuantisasi 4-bit (NF4) via `bitsandbytes` untuk menghasilkan jawaban yang terstruktur dan mematuhi batasan domain.
5. **Serving & UI**: Antarmuka interaktif disajikan menggunakan Gradio, menampilkan jawaban utama beserta panel *accordion* untuk dokumen referensi pendukung dan skor jarak $L_2$.

---

## Struktur Direktori

```text
.
├── app.py              # Main script (Retrieval, LLM Pipeline, & Gradio UI)
├── requirements.txt    # Daftar dependensi Python
├── .env.example        # Template konfigurasi environment variables
└── faiss_index/        # Checkpoint indeks vektor FAISS lokal
    ├── index.faiss     # Indeks matriks FAISS
    └── index.pkl       # Serialisasi metadata dokumen
```

---

## Persyaratan Sistem

- **Sistem Operasi**: Linux / Windows / macOS
- **Python**: Versi 3.10 atau lebih baru
- **Akselerasi Hardware**: NVIDIA GPU dengan dukungan CUDA direkomendasikan untuk menjalankan inferensi LLM 7B dengan kuantisasi 4-bit NF4 secara optimal. (Pada lingkungan CPU, inferensi akan berjalan dalam mode unquantized/standard).
- **Hugging Face Account**: Diperlukan User Access Token (`HF_TOKEN`) untuk mengakses repositori Hugging Face Hub.

---

## Variabel Lingkungan (Environment Variables)

Aplikasi membaca konfigurasi dari *environment variables* atau file `.env` lokal:

| Variabel | Wajib | Nilai Default | Deskripsi |
|---|---|---|---|
| `HF_TOKEN` | Ya | `-` | Hugging Face Access Token untuk autentikasi API/model |
| `FAISS_FOLDER` | Tidak | `faiss_index` | Path folder yang memuat file `index.faiss` dan `index.pkl` |
| `MODEL_ID` | Tidak | `Qwen/Qwen2.5-7B-Instruct` | Identifier repositori model LLM di Hugging Face Hub |
| `EMBEDDING_MODEL_ID` | Tidak | `intfloat/multilingual-e5-base` | Identifier model embedding di Hugging Face Hub |
| `PORT` | Tidak | `7860` | Port HTTP untuk menjalankan server Gradio |

*Catatan: Jangan pernah melakukan commit terhadap file `.env` yang berisi token rahasia ke repositori.*

---

## Cara Instalasi dan Memulai

### 1. Klon Repositori
```bash
git clone https://github.com/username/stuntbot.git
cd stuntbot
```

### 2. Buat dan Aktifkan Virtual Environment
```bash
# Pengguna Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Pengguna Windows (PowerShell)
python -m venv venv
.env\Scripts\Activate.ps1
```

### 3. Instal Dependensi
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Konfigurasi File Environment
Salin file `.env.example` menjadi `.env`, lalu isi `HF_TOKEN` Anda:
```bash
cp .env.example .env
```
Isi file `.env`:
```env
HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
FAISS_FOLDER=faiss_index
MODEL_ID=Qwen/Qwen2.5-7B-Instruct
EMBEDDING_MODEL_ID=intfloat/multilingual-e5-base
PORT=7860
```

### 5. Jalankan Aplikasi
```bash
python app.py
```
Aplikasi Gradio akan secara otomatis berjalan di `http://localhost:7860` atau `http://0.0.0.0:7860`.

---

## Catatan Teknis & Lingkungan Penelitian

1. **Indeks FAISS Pre-built**: Folder `faiss_index/` yang tersedia di repositori ini merupakan hasil ekstraksi dan pengindeksan dari 60 dokumen resmi kebijakan, panduan teknis, dan regulasi stunting di Indonesia. Indeks ini dibangkitkan secara *offline* melalui *notebook* eksternal dan tidak dibentuk ulang setiap kali aplikasi dijalankan.
2. **Kuantisasi Model**: Penggunaan `bitsandbytes` (4-bit NF4 quantization) membutuhkan perangkat keras berbasis GPU NVIDIA dengan CUDA. Jika dijalankan pada *environment* tanpa GPU CUDA, pemuatan konfigurasi kuantisasi akan dilewati secara otomatis.
3. **Versi Skripsi / Tag Rilis**: Kode yang dijadikan rujukan pada dokumen Skripsi diarsip dalam bentuk *Release Tag* pada repositori ini. Silakan periksa bagian **Releases** untuk mencocokkan kondisi kode saat pengujian skripsi dilakukan.

---

## Informasi Penelitian

- **Judul Skripsi**: Implementasi dan Analisis Kinerja Retrieval-Augmented Generation (RAG) Menggunakan Model Qwen 2.5 7B Instruct dalam Sistem Tanya Jawab Seputar Stunting
- **Program Studi**: Teknik Informatika
- **Institusi**: Universitas Komputer Indonesia (UNIKOM)
