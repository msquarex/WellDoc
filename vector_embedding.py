import os
import uuid
import argparse
import requests
import mysql.connector
import weaviate
from dotenv import load_dotenv
from typing import List

# ========= Load .env =========
load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "rough")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/embeddings")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "nomic-embed-text")

# --- Config ---
RAG_CLASS = "RAG"
EMBEDDING_BATCH_SIZE = 16

# ---------- DB Helpers ----------

def get_chunks_from_db() -> List[dict]:
    """Fetch unvectorized chunks from DB."""
    conn = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE
    )
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT id, source_file, chunk_number, content, page_number
        FROM pdf_chunks
        WHERE vector_embedded = 0
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

def update_vectorized_flag(chunk_ids: List[int]):
    """Mark chunks as vectorized in DB."""
    if not chunk_ids:
        return
    conn = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE
    )
    cursor = conn.cursor()
    format_ids = ",".join(["%s"] * len(chunk_ids))
    cursor.execute(f"UPDATE pdf_chunks SET vector_embedded = 1 WHERE id IN ({format_ids})", chunk_ids)
    conn.commit()
    cursor.close()
    conn.close()

# ---------- Embedding Helpers ----------

def get_embedding(text: str) -> List[float]:
    """Fetch embedding for a single text from Ollama."""
    resp = requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": text})
    resp.raise_for_status()
    data = resp.json()
    return data["embedding"]

def batch_chunks(chunks: List[dict], batch_size: int):
    """Yield batches of chunks for efficient processing."""
    for i in range(0, len(chunks), batch_size):
        yield chunks[i:i + batch_size]

# ---------- Weaviate Setup ----------

def setup_weaviate_schema(client):
    """Ensure Weaviate schema has RAG class with minimal properties."""
    existing = client.collections.list_all()  # returns list[str]

    if RAG_CLASS not in existing:
        client.collections.create(
            name=RAG_CLASS,
            vector_config=weaviate.classes.config.Configure.Vectors.self_provided(),  # ✅ correct format for v4
            properties=[
                weaviate.classes.config.Property(
                    name="content", data_type=weaviate.classes.config.DataType.TEXT
                ),
                weaviate.classes.config.Property(
                    name="source_file", data_type=weaviate.classes.config.DataType.TEXT
                ),
                weaviate.classes.config.Property(
                    name="page_number", data_type=weaviate.classes.config.DataType.INT
                ),
            ],
        )
        print(f"Created collection '{RAG_CLASS}'.")

    return client.collections.get(RAG_CLASS)

# ---------- Logic ----------

def stable_uuid(chunk: dict) -> str:
    """Stable UUID from source_file + chunk_number."""
    base = f"{chunk['source_file']}_{chunk['chunk_number']}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, base))

def store_embeddings(chunks: List[dict], collection):
    """Generate embeddings for chunks and push to Weaviate."""
    print(f"Processing {len(chunks)} chunks in batches of {EMBEDDING_BATCH_SIZE}...")

    for batch in batch_chunks(chunks, EMBEDDING_BATCH_SIZE):
        successful_ids = []
        for chunk in batch:
            try:
                if not chunk.get("content"):
                    continue
                embedding = get_embedding(chunk["content"])
                obj_uuid = stable_uuid(chunk)

                data_object = {
                    "content": chunk["content"],
                    "source_file": chunk["source_file"],
                    "page_number": chunk.get("page_number", 0)
                }

                collection.data.insert(
                    properties=data_object,
                    vector=embedding,
                    uuid=obj_uuid
                )

                successful_ids.append(chunk["id"])
                print(f"✅ Vectorized chunk {chunk['chunk_number']} from {chunk['source_file']}")
            except Exception as e:
                print(f"❌ Failed to vectorize chunk {chunk['id']}: {e}")
                continue

        update_vectorized_flag(successful_ids)

# ---------- Main Entry ----------

def main(args):
    client = weaviate.connect_to_local()  # connect
    try:
        if args.weaviate:
            collection = setup_weaviate_schema(client)
            chunks = get_chunks_from_db()
            if not chunks:
                print("No chunks to vectorize.")
                return

            store_embeddings(chunks, collection)
            print("🎉 Vectorization complete.")
        else:
            print("Use --weaviate to start embedding.")
    finally:
        client.close()  # ✅ avoid ResourceWarning

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weaviate", action="store_true", help="Use Weaviate for vector storage")
    args = parser.parse_args()
    main(args)
