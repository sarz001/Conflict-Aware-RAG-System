import os
from chromadb import PersistentClient
from dotenv import load_dotenv
from google.generativeai import embed_content, configure

load_dotenv()
configure(api_key=os.getenv("GEMINI_API_KEY"))

EMBED_MODEL = "models/text-embedding-004"


# ----------------------------
# Embedding function
# ----------------------------
def embed(text):
    return embed_content(
        model=EMBED_MODEL,
        content=text
    )["embedding"]


# ----------------------------
# Connect to Chroma DB
# ----------------------------
def get_collection():
    client = PersistentClient(path="./chroma_db")
    return client.get_collection("nebula_policies")


# ----------------------------
# Convert YYYY-MM-DD → integer 20240601
# ----------------------------
def date_to_int(date_str):
    return int(date_str.replace("-", ""))


# ----------------------------
# Metadata-aware reranker
# ----------------------------
def rerank(chunks, user_role):
    ranked = []

    # role = intern → "interns"
    target_scope = f"{user_role}s"

    for ch in chunks:
        meta = ch["meta"]

        # 1. Role match
        role_match = 1 if meta["role_scope"] == target_scope else 0

        # 2. Convert effective_date to sortable integer
        date_value = date_to_int(meta["effective_date"])

        # 3. Cosine similarity distance from Chroma
        distance = ch["dist"]

        ranked.append({
            "text": ch["text"],
            "meta": meta,
            "score": (
                -role_match,       # role match first
                -date_value,       # newer documents first
                distance           # then similarity
            )
        })

    # Sort by score
    ranked = sorted(ranked, key=lambda x: x["score"])

    # Remove score field before returning
    return [{"text": r["text"], "meta": r["meta"]} for r in ranked]


# ----------------------------
# Main Retrieve Function
# ----------------------------
def retrieve(query, user_role="unknown"):
    collection = get_collection()
    q_emb = embed(query)

    # ---- 1. Vector search from Chroma ----
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=7
    )

    # Flatten Chroma output
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "text": doc,
            "meta": meta,
            "dist": dist
        })

    # ---- 2. Metadata-aware reranking ----
    ranked = rerank(chunks, user_role)

    # ---- 3. Return top-3 best chunks ----
    return ranked[:3]
