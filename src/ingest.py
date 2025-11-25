from chromadb import PersistentClient
from google.generativeai import embed_content, configure
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
configure(api_key=os.getenv("GEMINI_API_KEY"))


# -----------------------------------------
# 1. Embedding Function
# -----------------------------------------
def embed(text):
    response = embed_content(
        model="models/text-embedding-004",
        content=text
    )
    return response["embedding"]


# -----------------------------------------
# 2. Chunking
# -----------------------------------------
def chunk_text(text, size=600, overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap
    )
    return splitter.split_text(text)


# -----------------------------------------
# 3. Metadata Extraction (Important!)
# -----------------------------------------
def extract_metadata(filename):
    """
    Extract effective_date, role_scope, doc_type based on file name.
    """
    if "employee_handbook" in filename:
        return {
            "effective_date": "2024-01-15",
            "role_scope": "all_employees",
            "doc_type": "handbook"
        }

    if "manager_updates" in filename:
        return {
            "effective_date": "2024-06-01",
            "role_scope": "all_employees",
            "doc_type": "policy_update"
        }

    if "intern_onboarding" in filename:
        return {
            "effective_date": "2024-06-01",
            "role_scope": "interns",
            "doc_type": "role_specific"
        }

    return {
        "effective_date": "2024-01-01",
        "role_scope": "all_employees",
        "doc_type": "general"
    }


# -----------------------------------------
# 4. Ingestion Pipeline
# -----------------------------------------
def ingest_data():
    client = PersistentClient(path="./chroma_db")

    # Make a collection
    collection = client.get_or_create_collection(
        name="nebula_policies",
        metadata={"hnsw:space": "cosine"}
    )

    # Input files
    files = [
        "employee_handbook_v1.txt",
        "manager_updates_2024.txt",
        "intern_onboarding_faq.txt"
    ]

    ids, docs, metas, embs = [], [], [], []

    for filename in files:
        path = os.path.join("data", filename)

        with open(path, "r", encoding="utf-8") as f:
            full_text = f.read()

        # Extract metadata (NOT HARDCODED)
        base_meta = extract_metadata(filename)

        # Chunk text
        chunks = chunk_text(full_text)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{filename}_chunk_{i}"

            ids.append(chunk_id)
            docs.append(chunk)

            # Combine chunk info + doc metadata
            metas.append({
                "filename": filename,
                "effective_date": base_meta["effective_date"],
                "role_scope": base_meta["role_scope"],
                "doc_type": base_meta["doc_type"],
                "chunk_index": i
            })

            embs.append(embed(chunk))

    # Store in DB
    collection.add(
        ids=ids,
        documents=docs,
        metadatas=metas,
        embeddings=embs
    )

    print("✅ Ingestion complete: All documents stored with metadata!")


# -----------------------------------------
# Run ingestion
# -----------------------------------------
if __name__ == "__main__":
    ingest_data()
