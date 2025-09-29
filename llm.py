import os
import mysql.connector
import requests
import weaviate
from dotenv import load_dotenv
from datetime import datetime

# ========== Load ENV ==========
load_dotenv()

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "rough")

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2:7b")

RAG_CLASS = "RAG"

# ========== MySQL Setup ==========
def init_mysql():
    conn = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE
    )
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_input TEXT,
            bot_response LONGTEXT,
            created_at DATETIME
        )
    """)
    conn.commit()
    return conn

def save_chat(conn, user_input, bot_response):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO chat_history (user_input, bot_response, created_at)
        VALUES (%s, %s, %s)
    """, (user_input, bot_response, datetime.now()))
    conn.commit()

# ========== Hybrid Search ==========
def hybrid_search(client, query, top_k=3):
    """Hybrid search with manual query embedding from Ollama."""
    try:
        # Step 1: Get embedding for query from Ollama embeddings API
        resp = requests.post(
            "http://localhost:11434/api/embeddings",
            json={
                "model": os.getenv("EMBED_MODEL", "nomic-embed-text"),
                "prompt": query
            },
            timeout=30
        )
        resp.raise_for_status()
        embedding = resp.json()["embedding"]

        # Step 2: Run hybrid search with BM25 + vector
        collection = client.collections.get(RAG_CLASS)
        results = collection.query.hybrid(query=query, vector=embedding, limit=top_k)

        hits = []
        for o in results.objects:
            file = o.properties.get("source_file", "unknown")
            page = o.properties.get("page_number", "?")
            content = o.properties.get("content", "")

            # ✅ include file + page metadata with context
            hits.append(f"(Source: {file}, Page {page}) {content}")

        if not hits:
            return "⚠️ No relevant information found in current knowledge base."

        return "\n".join(hits)

    except requests.exceptions.RequestException as e:
        print(f"⚠️ Error getting embeddings: {e}")
        return "⚠️ Unable to search knowledge base due to embedding service error."
    except Exception as e:
        print(f"⚠️ Error in hybrid search: {e}")
        return "⚠️ Unable to search knowledge base due to search error."

# ========== LLM Query ==========
def ask_llm(prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 256,
            "num_ctx": 2048,
            "num_gpu": 0
        }
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()
    except requests.exceptions.RequestException as e:
        return f"⚠️ Connection error: {e}. Please check if Ollama is running."
    except Exception as e:
        return f"⚠️ Unexpected error: {e}"

# ========== Prompt Engineering ==========
def build_prompt(context, user_input, answer_mode):
    """Build WellDoc-style prompt with explicit source references."""
    mode_instruction = (
        "Respond concisely using clear bullet points."
        if answer_mode == "concise"
        else "Respond with a detailed, logically structured explanation."
    )

    return f"""
You are WellDoc’s AI Assistant, a trusted source of accurate, clinical-grade information.

Your task:
- Provide answers **strictly based on the context provided below**.
- Always include the source file name and page number in your response.
- Do not invent or assume. If no info is present, reply with:
  "The provided context does not contain enough information to answer this question."

--- CONTEXT START ---
{context}
--- CONTEXT END ---

User Question: {user_input}

Response Guidelines:
- {mode_instruction}
- Use a professional, clinical, user-friendly tone.
- Explicitly mention the file name + page number where the info was found.

Begin your answer below:
"""

# ========== Main Chat Loop ==========
def chat():
    conn = init_mysql()
    client = weaviate.connect_to_local()

    print("🤖 WellDoc RAG Chatbot (Llama2:7b + Weaviate Hybrid Search)")
    print("Type 'exit' to quit.\n")

    while True:
        answer_mode = input("Choose answer style ('concise' or 'detailed'): ").strip().lower()
        if answer_mode in {"concise", "detailed"}:
            break
        print("❌ Invalid choice. Please type 'concise' or 'detailed'.")

    try:
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print("👋 Goodbye!")
                break

            if not user_input:
                print("⚠️ Please enter a question.")
                continue

            print("🔍 Searching knowledge base...")
            context = hybrid_search(client, user_input)

            print("🤖 Generating response...")
            prompt = build_prompt(context, user_input, answer_mode)
            bot_response = ask_llm(prompt)

            print(f"Bot: {bot_response}\n")

            try:
                save_chat(conn, user_input, bot_response)
            except Exception as e:
                print(f"⚠️ Warning: Could not save chat history: {e}")

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        try:
            conn.close()
        except:
            pass
        try:
            client.close()
        except:
            pass

if __name__ == "__main__":
    chat()
