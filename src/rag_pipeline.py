import google.generativeai as genai
import os
from dotenv import load_dotenv
from src.retriever import retrieve   # uses reranker internally

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

LLM = "gemini-2.0-flash"


# -------------------------------------------------
# STEP 1 — Detect User Role (intern / employee / manager / unknown)
# -------------------------------------------------
def detect_role(query):
    model = genai.GenerativeModel(LLM)

    prompt = f"""
You are a role extraction assistant.

Extract the user's role from this query.
Return only ONE WORD from this set:
- intern
- employee
- manager
- unknown

If the role is not explicitly stated, return "unknown".

Query: "{query}"
"""

    response = model.generate_content(prompt)
    role = response.text.strip().lower()

    if role not in ["intern", "employee", "manager", "unknown"]:
        role = "employee"

    return role


# -------------------------------------------------
# STEP 2 — Build Context (include metadata)
# -------------------------------------------------
def build_context(chunks):
    ctx = ""

    for ch in chunks:
        meta = ch["meta"]

        ctx += f"""
[DOCUMENT]
filename: {meta.get("filename")}
effective_date: {meta.get("effective_date")}
role_scope: {meta.get("role_scope")}
doc_type: {meta.get("doc_type")}

TEXT:
{ch['text']}
---------------------------------------------------------
"""
    return ctx


# -------------------------------------------------
# STEP 3 — Main Answering Pipeline
# -------------------------------------------------
def answer_query(query):
    # 1. Detect user role
    user_role = detect_role(query)

    # 2. Retrieve document chunks (reranking happens inside retriever!)
    chunks = retrieve(query, user_role)

    # 3. Build context for LLM
    context = build_context(chunks)

    # 4. Final conflict-aware prompt
    prompt = f"""
You are NebulaGears' official policy assistant.

Below are policy snippets with metadata such as:
- effective_date
- role_scope
- doc_type

You MUST answer ONLY using these documents.

=========================================================
CONFLICT RESOLUTION RULES (MANDATORY — FOLLOW STRICTLY)
=========================================================

1. ROLE-SPECIFIC RULES override general rules.
   - If a policy’s metadata says role_scope="interns" and the user is an intern,
     this overrides handbook & manager updates.

2. NEWER DOCUMENTS override older ones.
   - Compare effective_date (YYYY-MM-DD).
   - Example: 2024-06-01 overrides 2024-01-15.

3. If an older rule allowed something that a newer rule restricts,
   the newer restriction MUST be followed.

4. If conflicts still remain, APPLY THE MOST RESTRICTIVE policy.
   (Company safety-first interpretation.)

5. YOU MUST CITE EXACT FILES used, using EXACT format:
   - Source: filename.txt

=========================================================
DOCUMENT CONTEXT
=========================================================
{context}

=========================================================
USER QUERY
=========================================================
"{query}"

Now produce the correct, authoritative answer following ALL rules above.
"""

    model = genai.GenerativeModel(LLM)
    response = model.generate_content(prompt)

    return response.text
