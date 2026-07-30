# StuntBot — Sistem Tanya Jawab RAG Seputar Stunting

Penelitian mengenai **"Implementasi dan Analisis Kinerja Retrieval-Augmented Generation (RAG) Menggunakan Model Qwen 2.5 7B Instruct dalam Sistem Tanya Jawab Seputar Stunting"** — Program Studi Teknik Informatika, Universitas Komputer Indonesia (UNIKOM).

Repo ini cuma berisi bagian serving: retrieval, generation, dan antarmuka Gradio-nya. Proses membangun korpus, chunking dokumen, sintesis data evaluasi, dan evaluasi RAGAS dikerjakan di notebook terpisah dan tidak dimasukkan ke sini.

## Cara kerjanya

Alurnya sederhana: pertanyaan pengguna di-embed dengan `intfloat/multilingual-e5-base`, dicocokkan ke FAISS index pakai MMR (`k=5, fetch_k=30, lambda_mult=0.85`), lalu dokumen yang ketemu dirangkai jadi konteks dan dikirim ke Qwen 2.5 7B Instruct untuk dijawab.

## Struktur direktori

```
Stuntbot/
├── app.py              # retrieval + generation + UI, semuanya di satu file
├── requirements.txt
├── .env.example        # template, salin jadi .env dan isi sendiri
├── faiss_index/
│   ├── index.faiss
│   └── index.pkl
├── index.html
├── style.css
└── README.md
```

## Yang dibutuhkan

- Python 3.10 ke atas
- Akun [OpenRouter](https://openrouter.ai) untuk panggil Qwen 2.5 7B Instruct
- Akun [ngrok](https://ngrok.com) untuk expose antarmuka ke publik

Daftar lengkap dependency Python ada di `requirements.txt`.

## Environment variables

Salin `.env.example` jadi `.env`, lalu isi:

| Variabel | Wajib | Default |
|---|---|---|
| `OPENROUTER_API_KEY` | Ya | — |
| `NGROK_TOKEN` | Ya | — |
| `OPENROUTER_MODEL_ID` | Tidak | `qwen/qwen-2.5-7b-instruct` |
| `FAISS_FOLDER` | Tidak | `faiss_index` |
| `EMBEDDING_MODEL_ID` | Tidak | `intfloat/multilingual-e5-base` |
| `PORT` | Tidak | `7860` |

`.env` sudah masuk `.gitignore` — jangan pernah commit file ini dengan isi API key asli.

## Menjalankan

```bash
python -m venv venv

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
python app.py
```

Kalau berhasil, terminal akan menampilkan link ngrok yang bisa langsung dibuka di browser. Halaman peringatan ngrok yang muncul di kunjungan pertama itu normal, tinggal klik "Visit Site".

## Catatan

Folder `faiss_index/` di sini adalah hasil build dari notebook persiapan korpus (60 dokumen kebijakan/panduan stunting yang sudah di-chunk dan di-embed), bukan dibangun ulang setiap kali aplikasi dijalankan.

Versi yang dilampirkan pada skripsi mengacu pada tag rilis tertentu (lihat bagian **Releases** di repo ini), bukan branch `main` yang kemungkinan masih berubah setelah sidang.