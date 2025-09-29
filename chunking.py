import os
from dotenv import load_dotenv
import mysql.connector
import hashlib
import fitz  # PyMuPDF for PDFs
import spacy
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import docx  # python-docx for .docx
import textract  # for legacy .doc

# ========= LOAD .env =========
load_dotenv()

PDF_DIRECTORY = os.getenv("PDF_DIRECTORY")

MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

# Chunking configuration (from .env or defaults)
MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", 500))
OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

if not PDF_DIRECTORY:
    raise ValueError("PDF_DIRECTORY is not set in .env file")

# Load spaCy once
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")

# MySQL connection setup
conn = mysql.connector.connect(
    host=MYSQL_HOST,
    user=MYSQL_USER,
    password=MYSQL_PASSWORD,
    database=MYSQL_DATABASE
)
cursor = conn.cursor()

# ---------- Table Creation ----------
def ensure_pdf_chunks_table_exists():
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pdf_chunks (
            id INT AUTO_INCREMENT PRIMARY KEY,
            source_file VARCHAR(255),
            chunk_number INT,
            content LONGTEXT,
            file_hash VARCHAR(255),
            last_modified DATETIME,
            chunked BOOLEAN,
            vector_embedded BOOLEAN,
            page_number INT,
            created_at DATETIME
        )
    """)
    conn.commit()

# ---------- MySQL Helpers ----------
def fetch_existing_hashes():
    cursor.execute("SELECT DISTINCT file_hash FROM pdf_chunks")
    return set(row[0] for row in cursor.fetchall())

def insert_chunks_to_db(chunks):
    for chunk in chunks:
        cursor.execute("""
            INSERT INTO pdf_chunks
            (source_file, chunk_number, content, file_hash, last_modified,
             chunked, vector_embedded, page_number, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            chunk['source_file'],
            int(chunk['chunk_number']),
            chunk['content'],
            chunk['file_hash'],
            chunk['last_modified'],
            chunk['status']['chunked'],
            chunk['status']['vector_embedded'],
            chunk['page_number'],
            chunk['created_at']
        ))
    conn.commit()

# ---------- Utility Functions ----------
def calculate_sha256_from_bytes(data):
    return hashlib.sha256(data).hexdigest()

def extract_blocks_from_pdf(pdf_bytes):
    pdf_document = fitz.open("pdf", pdf_bytes)
    results = []
    for page_number, page in enumerate(pdf_document, start=1):
        blocks = page.get_text("blocks")
        for block in blocks:
            text = block[4].strip()
            if not text:
                continue
            results.append({
                "page_number": page_number,
                "text": text
            })
    pdf_document.close()
    return results

def extract_text_from_docx(path):
    doc = docx.Document(path)
    results = []
    for i, para in enumerate(doc.paragraphs, start=1):
        text = para.text.strip()
        if text:
            results.append({"page_number": i, "text": text})
    return results

def extract_text_from_doc(path):
    text = textract.process(path).decode("utf-8")
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    return [{"page_number": i+1, "text": p} for i, p in enumerate(paragraphs)]

def split_with_spacy(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def regroup_sentences(sentences, max_tokens=MAX_TOKENS, overlap=OVERLAP):
    chunks, current_chunk, word_count = [], [], 0
    i = 0
    while i < len(sentences):
        words = sentences[i].split()
        if word_count + len(words) <= max_tokens:
            current_chunk.append(sentences[i])
            word_count += len(words)
            i += 1
        else:
            chunks.append(" ".join(current_chunk))
            # step back for overlap
            overlap_words = 0
            overlap_chunk = []
            j = len(current_chunk) - 1
            while j >= 0 and overlap_words < overlap:
                overlap_chunk.insert(0, current_chunk[j])
                overlap_words += len(current_chunk[j].split())
                j -= 1
            current_chunk = overlap_chunk
            word_count = sum(len(s.split()) for s in current_chunk)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# ---------- Main Logic ----------
def process_all_files_from_directory(data_dir):
    ensure_pdf_chunks_table_exists()

    files = [f for f in os.listdir(data_dir) if f.lower().endswith((".pdf", ".doc", ".docx"))]
    if not files:
        print("No supported files found in directory.")
        return

    existing_hashes = fetch_existing_hashes()
    new_chunks = []

    def process_file(file_name):
        path = os.path.join(data_dir, file_name)
        with open(path, "rb") as f:
            file_data = f.read()
        file_hash = calculate_sha256_from_bytes(file_data)
        last_modified_str = datetime.now().isoformat()
        created_at_str = datetime.utcnow().isoformat()

        if file_hash in existing_hashes:
            print(f"Skipping {file_name} (already processed).")
            return []

        print(f"Processing {file_name}...")

        if file_name.lower().endswith(".pdf"):
            blocks = extract_blocks_from_pdf(file_data)
        elif file_name.lower().endswith(".docx"):
            blocks = extract_text_from_docx(path)
        elif file_name.lower().endswith(".doc"):
            blocks = extract_text_from_doc(path)
        else:
            return []

        if not blocks:
            return []

        processed_chunks = []
        for block in blocks:
            sentences = split_with_spacy(block["text"])
            sentence_chunks = regroup_sentences(sentences, max_tokens=MAX_TOKENS, overlap=OVERLAP)

            for chunk_text in sentence_chunks:
                processed_chunks.append({
                    "content": chunk_text,
                    "page_number": block["page_number"]
                })

        return [{
            "source_file": file_name,
            "chunk_number": f"{idx+1}",
            "content": chunk["content"],
            "file_hash": file_hash,
            "last_modified": last_modified_str,
            "created_at": created_at_str,
            "page_number": chunk["page_number"],
            "status": {
                "chunked": True,
                "vector_embedded": False
            }
        } for idx, chunk in enumerate(processed_chunks)]

    with ThreadPoolExecutor() as executor:
        results = executor.map(process_file, files)
        for chunks in results:
            new_chunks.extend(chunks)

    if new_chunks:
        insert_chunks_to_db(new_chunks)
        print(f"Inserted {len(new_chunks)} new chunk(s) from directory.")
    else:
        print("No new chunks to insert from directory.")

# ---------- Entry ----------
if __name__ == "__main__":
    process_all_files_from_directory(PDF_DIRECTORY)
