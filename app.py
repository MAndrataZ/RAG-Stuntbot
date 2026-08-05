"""
StuntBot — Sistem Tanya Jawab RAG Seputar Stunting
- Antarmuka Gradio (Cell 9), tanpa bagian ngrok
"""

import os
import torch
from typing import List, Dict

from dotenv import load_dotenv
load_dotenv()  # hanya efektif untuk development lokal;
               # di HuggingFace Spaces, Repository Secrets otomatis
               # tersedia sebagai environment variable tanpa perlu file .env

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import gradio as gr

# ============================================================
# 1. KONFIGURASI (dibaca dari environment variable / secrets)
# ============================================================
HF_TOKEN            = os.environ.get("HF_TOKEN")                 # wajib diisi via Secrets
FAISS_FOLDER        = os.environ.get("FAISS_FOLDER", "faiss_index")
MODEL_ID            = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
EMBEDDING_MODEL_ID  = os.environ.get("EMBEDDING_MODEL_ID", "intfloat/multilingual-e5-base")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Perangkat  : {device.upper()}")
print(f"Model LLM  : {MODEL_ID}")
print(f"Embedding  : {EMBEDDING_MODEL_ID}")

# ============================================================
# 2. LOAD EMBEDDING MODEL + FAISS INDEX (tanpa proses embedding ulang)
# ============================================================
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_ID,
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},
)

_faiss_file = os.path.join(FAISS_FOLDER, "index.faiss")
_pkl_file   = os.path.join(FAISS_FOLDER, "index.pkl")

if not (os.path.isfile(_faiss_file) and os.path.isfile(_pkl_file)):
    raise FileNotFoundError(
        f"FAISS index tidak ditemukan di folder '{FAISS_FOLDER}/'. "
        f"Pastikan 'index.faiss' dan 'index.pkl' sudah diupload ke repo Space ini "
        f"(lihat panduan struktur direktori)."
    )

print("Memuat FAISS index dari checkpoint...")
vector_db = FAISS.load_local(
    folder_path=FAISS_FOLDER,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True,
)
print(f"FAISS index dimuat. Jumlah vektor: {vector_db.index.ntotal:,}")

retriever = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 30, "lambda_mult": 0.85},
)

# ============================================================
# 3. RETRIEVAL (identik dengan Cell 6)
# ============================================================
def _format_query(query: str) -> str:
    q = query.strip()
    if q.startswith("query:"):
        return q
    return f"query: {q}"


def clean_retrieved_content(raw_content: str) -> str:
    content = raw_content.strip()
    if content.startswith("passage: "):
        content = content[len("passage: "):]
    elif content.startswith("passage:"):
        content = content[len("passage:"):]
    return content.strip()


def retrieve_documents_for_eval(user_query: str, top_k: int = 5) -> List[Dict]:
    formatted_query = _format_query(user_query)
    docs_retrieved  = retriever.invoke(formatted_query)

    score_map_k = max(top_k * 6, 30)
    docs_score  = vector_db.similarity_search_with_score(
        formatted_query, k=score_map_k
    )

    score_map_by_idx: Dict[int, float] = {}
    score_map_by_body: Dict[str, float] = {}
    for d, s in docs_score:
        ci = d.metadata.get("chunk_index")
        if ci is not None:
            score_map_by_idx[int(ci)] = float(s)
        body_key = clean_retrieved_content(d.page_content)[:200]
        if body_key not in score_map_by_body:
            score_map_by_body[body_key] = float(s)

    retrieved_data = []
    for doc in docs_retrieved[:top_k]:
        meta       = doc.metadata
        body_clean = clean_retrieved_content(doc.page_content)
        chunk_idx  = meta.get("chunk_index")

        if chunk_idx is not None and int(chunk_idx) in score_map_by_idx:
            approx_score = score_map_by_idx[int(chunk_idx)]
        else:
            approx_score = score_map_by_body.get(body_clean[:200])

        retrieved_data.append({
            "chunk_index"       : chunk_idx if chunk_idx is not None else -1,
            "id_dokumen"        : meta.get("id_dokumen",        "Unknown"),
            "content"           : body_clean,
            "judul_dokumen"     : meta.get("judul_dokumen",     "Unknown"),
            "sumber"            : meta.get("sumber",            "Unknown"),
            "tahun"             : meta.get("tahun",              0),
            "jenis_dokumen"     : meta.get("jenis_dokumen",     "Unknown"),
            "tier_kredibilitas" : meta.get("tier_kredibilitas",  99),
            "status_berlaku"    : meta.get("status_berlaku",    "Unknown"),
            "keywords"          : meta.get("keywords",          ""),
            "chunk_keywords"    : meta.get("chunk_keywords",    ""),
            "content_hint"      : meta.get("content_hint",     "narrative"),
            "page"              : meta.get("page",               -1),
            "similarity_score"  : round(approx_score, 4) if approx_score is not None else None,
            "retriever_mode"    : "mmr",
        })

    return retrieved_data


def retrieve_raw_docs(user_query: str, top_k: int = 5):
    formatted_query = _format_query(user_query)
    docs = retriever.invoke(formatted_query)
    for doc in docs[:top_k]:
        doc.page_content = clean_retrieved_content(doc.page_content)
    return docs[:top_k]


# ============================================================
# 4. GENERATION — QWEN 2.5 7B INSTRUCT (identik dengan Cell 7)
# ============================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print(f"Memuat model: {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)

_model_kwargs = dict(device_map="auto", token=HF_TOKEN, trust_remote_code=True)
if device == "cuda":
    # Kuantisasi 4-bit hanya didukung di GPU. Di Space CPU-only, bagian ini
    # dilewati otomatis (lihat catatan GPU di README / panduan deployment).
    _model_kwargs["quantization_config"] = bnb_config

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **_model_kwargs)
print("Model Qwen 2.5 berhasil dimuat.")

text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    temperature=0.1,
    top_p=0.95,
    repetition_penalty=1.15,
    return_full_text=False,
)

llm_qwen = HuggingFacePipeline(pipeline=text_generation_pipeline)

# Prompt template — identik dengan Cell 7
prompt_template_qwen = """<|im_start|>system
# PERAN DAN IDENTITAS
Anda adalah StuntBot, asisten tanya-jawab berbasis kecerdasan buatan yang \
dirancang untuk membantu masyarakat, tenaga kesehatan, dan pemangku kepentingan \
memperoleh informasi akurat seputar stunting dan gizi anak di Indonesia. \
Anda menjawab secara akurat dan formal, HANYA menggunakan informasi dari \
[Konteks] yang disediakan.

# KONTEKS DOMAIN
Stunting adalah kondisi gagal tumbuh pada anak balita akibat kekurangan gizi \
kronis, terutama pada 1.000 Hari Pertama Kehidupan (HPK). Di Indonesia, stunting \
diukur berdasarkan indikator tinggi badan per usia (TB/U) di bawah -2 standar \
deviasi. Percepatan penurunan stunting dikoordinasikan melalui BKKBN, Kemenkes, \
dan berbagai kementerian/lembaga terkait. Sumber data prevalensi utama mencakup \
SSGI (Survei Status Gizi Indonesia) dan SKI (Survei Kesehatan Indonesia).

════════════════════════════════════════
LARANGAN KERAS — TIDAK BOLEH DILANGGAR
════════════════════════════════════════
- JANGAN menyebutkan nama dokumen, judul, atau nama penerbit/sumber di dalam jawaban.
- JANGAN menyebutkan "ID Dokumen", "Tier Kredibilitas", "Status", "chunk_index",
  "Keyword", atau label metadata teknis apapun.
- JANGAN menyebut kata "konteks", "retrieval", "dokumen yang diberikan",
  "basis pengetahuan", atau frasa sejenisnya dalam jawaban substantif.
- JANGAN menggunakan frasa self-referential berikut dalam jawaban:
    "penelitian ini", "studi ini", "artikel ini", "makalah ini", "kajian ini",
    "tulisan ini", "buku ini", "dokumen ini", "jurnal ini",
    "tabel di atas", "gambar di atas", "grafik di atas", "data di atas",
    "tabel tersebut", "gambar tersebut",
    "penulis", "penyusun", "penulis berpendapat", "penulis menyatakan",
    "berdasarkan tabel", "berdasarkan grafik", "berdasarkan gambar",
    "sebagaimana disebutkan", "seperti disebutkan di atas",
    "seperti terlihat pada", "referensi yang ada",
    "sumber yang tersedia", "informasi yang tersedia"
  → Ganti dengan parafrase impersonal, contoh:
      "penelitian ini menunjukkan X"  →  "sebuah kajian menemukan X"
      "berdasarkan data di atas, X"   →  "data menunjukkan X"
      "penulis menyimpulkan X"        →  "kesimpulan dari analisis tersebut adalah X"
- JANGAN menggunakan pengetahuan di luar [Konteks].
- JANGAN menambahkan fakta, angka, atau nama program yang tidak ada di [Konteks].
- JANGAN membuat daftar referensi di akhir jawaban.

════════════════════════════════════════
LARANGAN TEMPORAL
════════════════════════════════════════
- JANGAN menggunakan kata atau frasa temporal ambigu berikut:
    "saat ini", "sekarang", "terkini", "terbaru", "kini", "hari ini",
    "tahun ini", "dewasa ini", "belakangan ini", "data terbaru"
- Jika [Konteks] memuat data statistik dengan tahun spesifik,
  WAJIB menyebut tahun tersebut dalam jawaban.
  Contoh BENAR : "prevalensi stunting pada 2024 adalah 21,5%"
  Contoh SALAH : "prevalensi stunting terkini adalah 21,5%"
- Jika [Konteks] tidak menyebut tahun, hilangkan klaim temporal sepenuhnya.

════════════════════════════════════════
ATURAN DOMAIN & PENOLAKAN
════════════════════════════════════════
Jawab HANYA pertanyaan yang berkaitan dengan:
  - Stunting: definisi, penyebab, dampak, pencegahan, dan tata laksana
  - Gizi ibu hamil dan anak balita (0–59 bulan)
  - Program pemerintah terkait stunting (BKKBN, Kemenkes, Posyandu, PMT, ASI)
  - Data prevalensi stunting (SSGI, SKI, data provinsi/kabupaten)
  - Regulasi dan kebijakan yang berkaitan langsung dengan penurunan stunting
  - Intervensi gizi spesifik dan sensitif untuk pencegahan stunting
  - Pertanyaan dalam lingkup stunting & kesehatan gizi yang terhubung secara langsung lainnya

Jika pertanyaan berada di luar domain di atas:
→ Tolak dengan sopan:
  "Mohon maaf, pertanyaan tersebut berada di luar cakupan topik stunting
   dan gizi anak yang dapat saya bantu."

Jika informasi tidak tersedia di [Konteks]:
→ "Mohon maaf, informasi spesifik tersebut tidak tersedia dalam
   cakupan pengetahuan yang dapat saya gunakan untuk menjawab."

Jika [Konteks] tidak cukup kuat untuk menjawab dengan yakin:
→ Tolak dengan sopan. Jangan menebak atau berspekulasi.

════════════════════════════════════════
ATURAN PENULISAN JAWABAN
════════════════════════════════════════
1. Buka jawaban dengan kalimat yang langsung menjawab inti pertanyaan —
   framing ulang inti pertanyaan dalam kalimat pembuka.
   Contoh pola:
   - Untuk pertanyaan "berapa..."  : "Angka/jumlah/dosis [X] adalah..."
   - Untuk pertanyaan "mengapa..." : "[X] terjadi karena..."
   - Untuk pertanyaan "bagaimana...": "Cara/mekanisme [X] adalah..."
   - Untuk pertanyaan "apa itu..."  : "[X] adalah kondisi/program/kebijakan..."

2. Gunakan bahasa Indonesia formal dan jelas.
3. Panjang jawaban:
   - Pertanyaan tunggal (faktual, kausal, prosedural): 2–5 kalimat.
   - Pertanyaan yang membutuhkan integrasi beberapa informasi: maksimal 8 kalimat.
   - Hindari pengulangan kalimat dan elaborasi yang tidak didukung [Konteks].
4. Setiap fakta dalam jawaban HARUS berasal dari [Konteks].
   Inferensi langsung satu langkah dari fakta [Konteks] diizinkan —
   misalnya menyimpulkan implikasi dari data yang tersedia.
   DILARANG menambahkan fakta eksternal meskipun terdengar relevan.
5. Pastikan jawaban konsisten apabila pertanyaan serupa diajukan kembali.

[Konteks]:
{context}
<|im_end|>
<|im_start|>user
Pertanyaan: {question}
<|im_end|>
<|im_start|>assistant
"""

prompt = PromptTemplate(
    template=prompt_template_qwen,
    input_variables=["context", "question"]
)


def format_docs_for_llm(docs) -> str:
    if docs and isinstance(docs[0], dict):
        formatted_chunks = []
        for doc in docs:
            header = (
                f"Judul  : {doc.get('judul_dokumen', 'N/A')}\n"
                f"Sumber : {doc.get('sumber',        'N/A')}\n"
                f"Tahun  : {doc.get('tahun',         'N/A')}"
            )
            text = f"{header}\n\n{doc.get('content', '')}"
            formatted_chunks.append(text)
        return "\n\n---\n\n".join(formatted_chunks)

    formatted_chunks = []
    for doc in docs:
        header = (
            f"Judul  : {doc.metadata.get('judul_dokumen', 'N/A')}\n"
            f"Sumber : {doc.metadata.get('sumber',        'N/A')}\n"
            f"Tahun  : {doc.metadata.get('tahun',         'N/A')}"
        )
        text = f"{header}\n\n{doc.page_content}"
        formatted_chunks.append(text)
    return "\n\n---\n\n".join(formatted_chunks)


def generate_answer_for_eval(query: str, top_k: int = 5) -> dict:
    retrieved_docs = retrieve_documents_for_eval(query, top_k=top_k)
    context_text   = format_docs_for_llm(retrieved_docs)
    final_prompt   = prompt.format(context=context_text, question=query)
    answer         = llm_qwen.invoke(final_prompt)

    return {
        "query"               : query,
        "generated_answer"    : answer.strip(),
        "retrieved_documents" : retrieved_docs,
    }


rag_chain = (
    {
        "context" : RunnableLambda(retrieve_raw_docs) | format_docs_for_llm,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm_qwen
    | StrOutputParser()
)


# ============================================================
# 5. ANTARMUKA GRADIO (identik dengan Cell 9, tanpa ngrok)
# ============================================================
def _render_doc_card(i: int, doc: dict) -> list:
    score     = doc.get("similarity_score")
    score_str = f"{score:.4f}" if score is not None else "N/A"
    tier      = doc.get("tier_kredibilitas", "?")
    status    = doc.get("status_berlaku", "Unknown")
    page      = doc.get("page", -1)
    page_str  = str(page) if page != -1 else "—"
    content   = doc.get("content", "")[:300]
    ellipsis  = "..." if len(doc.get("content", "")) > 300 else ""

    return [
        f"#### [{i}] {doc.get('judul_dokumen', 'Unknown')}",
        "",
        f"| Field | Nilai |",
        f"|:---|:---|",
        f"| Sumber | {doc.get('sumber', '—')} |",
        f"| Tahun | {doc.get('tahun', '—')} |",
        f"| Tipe Dokumen | {doc.get('jenis_dokumen', '—')} |",
        f"| Tier Kredibilitas | {tier} |",
        f"| Status | {status} |",
        f"| Halaman | {page_str} |",
        f"| L2 Distance Score | `{score_str}` |",
        f"| Chunk Index | `{doc.get('chunk_index', '—')}` |",
        "",
        f"**Cuplikan isi:**",
        f"> {content}{ellipsis}",
        "",
        "---",
        "",
    ]


def format_reference_panel(retrieved_docs: list) -> str:
    if not retrieved_docs:
        return "_Tidak ada dokumen yang ditemukan._"

    lines = [
        f"**Jumlah dokumen retrieved:** {len(retrieved_docs)}",
        "",
        "_L2 Distance Score: nilai lebih kecil menunjukkan relevansi lebih tinggi._",
        "",
        "---",
        "",
    ]

    sorted_docs = sorted(
        retrieved_docs,
        key=lambda d: d.get("similarity_score") or 999
    )

    for i, doc in enumerate(sorted_docs, 1):
        lines += _render_doc_card(i, doc)

    return "\n".join(lines)


def process_query_logic(user_input: str):
    if not user_input.strip():
        yield "Mohon ketikkan pertanyaan Anda.", ""
        return

    yield "_Mencari dokumen relevan..._", ""

    yield "_Menyusun jawaban..._", ""

    try:
        result = generate_answer_for_eval(user_input, top_k=5)
        answer         = result["generated_answer"]
        retrieved_docs = result["retrieved_documents"]
        formatted_refs = format_reference_panel(retrieved_docs)
    except Exception as e:
        yield f"Terjadi kesalahan: {str(e)}", ""
        return

    partial = ""
    for char in answer:
        partial += char
        yield partial, formatted_refs


with gr.Blocks(title="StuntBot") as demo:

    gr.Markdown("""
# StuntBot
Sistem tanya jawab stunting berbasis RAG
    """)

    with gr.Row():
        with gr.Column(scale=4):
            input_box = gr.Textbox(
                label="Pertanyaan",
                placeholder="Contoh: Berapa prevalensi stunting di Jawa Barat tahun 2022?",
                lines=4
            )
            with gr.Row():
                submit_btn = gr.Button("Kirim", variant="primary", scale=3)
                clear_btn  = gr.Button("Reset", variant="secondary", scale=1)

            gr.Examples(
                examples=[
                    ["Apa definisi stunting menurut Perpres Nomor 72 Tahun 2021?"],
                    ["Bagaimana tata laksana balita gizi buruk?"],
                    ["Berapa target penurunan stunting tahun 2024?"],
                    ["Berapa prevalensi stunting di Provinsi Maluku Utara tahun 2021?"],
                    ["Apa saja indikator keberhasilan intervensi stunting?"],
                ],
                inputs=input_box,
                label="Contoh Pertanyaan"
            )

        with gr.Column(scale=6):
            output_answer = gr.Markdown(
                value="_Jawaban akan muncul di sini._",
                label="Jawaban StuntBot"
            )
            with gr.Accordion("Dokumen Referensi", open=False):
                output_refs = gr.Markdown(
                    value="_Referensi akan muncul setelah pertanyaan diproses._"
                )

    gr.Markdown("---\n_Penelitian Skripsi · Teknik Informatika · 2025_")

    submit_btn.click(
        fn=process_query_logic,
        inputs=input_box,
        outputs=[output_answer, output_refs],
        show_progress="full"
    )
    input_box.submit(
        fn=process_query_logic,
        inputs=input_box,
        outputs=[output_answer, output_refs]
    )
    clear_btn.click(
        fn=lambda: (
            "_Jawaban akan muncul di sini._",
            "_Referensi akan muncul setelah pertanyaan diproses._",
            ""
        ),
        inputs=None,
        outputs=[output_answer, output_refs, input_box]
    )


# ============================================================
# 6. LAUNCH
# ============================================================
# Catatan: di HuggingFace Spaces, server_name/server_port ditangani otomatis
# oleh platform. Parameter di bawah aman untuk Spaces maupun run lokal.
if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))