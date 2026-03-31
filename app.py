import os
import re
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


APP_ROOT = Path(__file__).resolve().parent

# OpenAI client expects OPENAI_API_KEY; notebook uses OpenAI_API_KEY in .env — support both
load_dotenv(APP_ROOT / ".env")
openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("OpenAI_API_KEY")
if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key

app = Flask(__name__)

RESEARCH_PAPERS_DIR = APP_ROOT / "research_papers"
VECTOR_DB_DIR = APP_ROOT / "chroma_db"
COLLECTION_NAME = "research_papers"
EMBEDDING_MODEL = "text-embedding-3-small"
RAG_CHAT_MODEL = os.getenv("RAG_CHAT_MODEL", "gpt-4o-mini")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "20"))

# Based on Assingment-2 .ipynb (V6). Refusal wording relaxed slightly: "exact answer" caused false refusals
# when the answer was paraphrasable from context.
RAG_SYSTEM_PROMPT = """You are an elite academic research assistant specializing in Power Systems and AI.

CRITICAL RAG RULE: Answer the user's question using ONLY information from the provided CONTEXT.
Each chunk header shows its source paper and page number — use these to identify papers.
If the CONTEXT is completely unrelated to the question, output EXACTLY: "I don't have information about this in my knowledge base."
Otherwise, answer using whatever relevant information the context provides, even if the answer is partial.

[CONSTRAINTS]
1. Explanation Level: {level} (1=PhD, 2=Master, 3=Bachelor, 4=School).
2. Clean Citations: You MUST cite your sources numerically corresponding to the explicit CHUNK numbers provided below (e.g., [1], [2]).
3. Best Source Only: ONLY cite the single most relevant chunk that provided the primary answer. Do not list multiple citations for the same fact (e.g., NEVER write [1][2][3]).

CRITICAL INSTRUCTION: CHAIN OF THOUGHT
Before generating your final output, you MUST reason step-by-step inside <thinking> tags.
Step 1: Safety/Relevance - Is this a safe query?
Step 2: Context Scan - Scan the chunks. Which specific CHUNK [X] contains the best answer?
Step 3: Verification - Does the context support a reasonable answer?

IMPORTANT: After your thinking process, close with </thinking>.
Your final response MUST be written OUTSIDE and AFTER the </thinking> tags.

CONTEXT:
{context}
"""

RESEARCH_PAPERS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
def index():
    return render_template("index.html")


def pdf_to_chunks(pdf_path: str):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4o-mini",
        chunk_size=512,
        chunk_overlap=75,
    )
    return text_splitter.split_documents(documents)


def get_target_paper_path(filename: str):
    return RESEARCH_PAPERS_DIR / Path(filename).name


def add_chunks_to_chroma(chunks):
    """
    Embed document chunks and persist them in the local Chroma store (same idea as the notebook).
    First batch creates the DB; later calls append with add_documents.
    """
    if not chunks:
        return

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    persist = str(VECTOR_DB_DIR)

    # First run: empty persist dir → create collection from documents
    if not any(VECTOR_DB_DIR.iterdir()):
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist,
            collection_name=COLLECTION_NAME,
        )
        return

    db = Chroma(
        persist_directory=persist,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    db.add_documents(chunks)


def chroma_db_exists():
    return VECTOR_DB_DIR.exists() and any(VECTOR_DB_DIR.iterdir())


def get_vector_store():
    """Open persisted Chroma with the same embedding model used when indexing."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=str(VECTOR_DB_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def retrieve_similar_chunks(vector_db, query, k):
    """Same pattern as notebook: retriever with top-k then invoke."""
    retriever = vector_db.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)


def deduplicate_docs(docs):
    """
    Chroma often returns near-duplicate chunks (same page/source). Collapse repeats so the LLM
    sees more diverse context within top-k.
    """
    seen = set()
    out = []
    for d in docs:
        text = (d.page_content or "").strip()
        key = (
            d.metadata.get("source"),
            d.metadata.get("page"),
            text[:240],
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def analyze_context_for_debug(docs, context_text):
    """Structured check that retrieved text is non-empty and sized as expected."""
    chunk_sizes = [len((d.page_content or "").strip()) for d in docs]
    total_chars = len(context_text.strip())
    min_sz = min(chunk_sizes) if chunk_sizes else 0
    max_sz = max(chunk_sizes) if chunk_sizes else 0
    valid = (
        len(docs) > 0
        and total_chars > 0
        and min_sz > 0
    )
    previews = []
    for i, d in enumerate(docs[:5]):
        src = os.path.basename(str(d.metadata.get("source", "Unknown")))
        prev = (d.page_content or "").strip().replace("\n", " ")[:220]
        previews.append(
            {"chunk_index": i + 1, "source": src, "page": d.metadata.get("page"), "preview": prev}
        )
    return {
        "context_valid": valid,
        "num_chunks": len(docs),
        "total_context_chars": total_chars,
        "min_chunk_chars": min_sz,
        "max_chunk_chars": max_sz,
        "chunk_char_lengths": chunk_sizes,
        "chunk_previews": previews,
    }


def build_numbered_context(docs):
    """Notebook V6: numbered CHUNK blocks + source mapping string."""
    context_text = ""
    sources_detail = []
    for i, doc in enumerate(docs):
        source_file = os.path.basename(doc.metadata.get("source", "Unknown"))
        page_num = doc.metadata.get("page", "N/A")
        sources_detail.append(
            {"chunk": i + 1, "source": source_file, "page": page_num}
        )
        context_text += f"\n--- CHUNK [{i + 1}] | Source: {source_file} (Page {page_num}) ---\n{doc.page_content}\n"
    sources_mapping = "\n".join(
        f"[{s['chunk']}] {s['source']} (Page {s['page']})" for s in sources_detail
    )
    return context_text, sources_mapping.strip(), sources_detail


def clean_output(text):
    """Strip <thinking> blocks (notebook helper)."""
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    return text.strip()


def run_rag_query(query, level=2, k=None, include_context_debug=True):
    """
    Chroma retrieval + numbered context + ChatPromptTemplate + gpt-4o-mini (notebook flow; single model).
    """
    if k is None:
        k = RAG_TOP_K
    if not chroma_db_exists():
        return {
            "error": "No vector database yet. Add at least one paper before querying.",
        }
    try:
        vector_db = get_vector_store()
    except Exception as e:
        return {"error": f"Could not open Chroma: {e!s}"}

    docs = retrieve_similar_chunks(vector_db, query, k)
    docs = deduplicate_docs(docs)
    if not docs:
        out = {
            "answer": "I don't have information about this in my knowledge base.",
            "sources": "",
            "sources_detail": [],
        }
        if include_context_debug:
            out["context_validation"] = {
                "context_valid": False,
                "note": "Retriever returned zero usable chunks (empty after deduplication).",
            }
        return out

    context_text, sources_mapping, sources_detail = build_numbered_context(docs)
    validation = analyze_context_for_debug(docs, context_text)
    if not validation["context_valid"]:
        out = {
            "answer": "I don't have information about this in my knowledge base.",
            "sources": sources_mapping,
            "sources_detail": sources_detail,
        }
        if include_context_debug:
            out["context_validation"] = validation
        return out

    system_msg = RAG_SYSTEM_PROMPT.replace("{context}", context_text).replace("{level}", str(level))
    llm = ChatOpenAI(model=RAG_CHAT_MODEL, temperature=0.1)
    raw = llm.invoke([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": query},
    ])
    answer = clean_output(raw.content)

    out = {
        "answer": answer,
        "sources": sources_mapping,
        "sources_detail": sources_detail,
    }
    if include_context_debug:
        out["context_validation"] = validation
    return out


def add_new_paper_pipeline(uploaded_file):
    """
    Run the add-new-paper flow in order:
      1. add_file_if_new — keep PDF in research_papers only if new
      2. pdf_to_chunks — load and chunk the PDF
      3. add_chunks_to_chroma — embed chunks and persist to Chroma
    Returns dict: {"added": True} or {"added": False, "reason": str}
    """
    target_path = get_target_paper_path(uploaded_file.filename)
    if target_path.exists():
        return {
            "added": False,
            "reason": "A file with this name already exists in research_papers.",
        }

    try:
        # Save upload to temp path first; only persist in research_papers after DB insert succeeds.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            uploaded_file.save(tmp.name)
            temp_pdf_path = tmp.name

        chunks = pdf_to_chunks(temp_pdf_path)
        if not chunks:
            return {
                "added": False,
                "reason": "No text chunks could be extracted from this PDF.",
            }

        # Ensure metadata source points to final persisted file path.
        for chunk in chunks:
            chunk.metadata["source"] = str(target_path)

        add_chunks_to_chroma(chunks)

        # Move file into research_papers only after successful vector DB insert.
        shutil.move(temp_pdf_path, target_path)
    except Exception as e:
        return {
            "added": False,
            "reason": f"Could not store embeddings in Chroma: {e!s}",
        }
    finally:
        if "temp_pdf_path" in locals() and os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

    return {"added": True}


@app.post("/add_new_paper")
def add_new_paper():
    if "file" not in request.files:
        return jsonify({"added": False, "reason": "No file field in request (expected multipart key 'file')."}), 400

    uploaded = request.files["file"]
    if not uploaded or uploaded.filename == "":
        return jsonify({"added": False, "reason": "No file selected."}), 400
    if not uploaded.filename.lower().endswith(".pdf"):
        return jsonify({"added": False, "reason": "Only .pdf files are supported."}), 400

    result = add_new_paper_pipeline(uploaded)
    if result.get("added"):
        return jsonify({"added": True})
    return jsonify({"added": False, "reason": result.get("reason", "Unknown error.")})


@app.post("/query_research_papers")
def query_research_papers():
    """
    JSON body: { "query": str, "level": 1-4, optional "k": int }
    Runs Chroma retrieval + RAG (same flow as Assingment-2 .ipynb V6; OpenAI only).
    """
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    level_raw = data.get("level", 2)
    k_raw = data.get("k", RAG_TOP_K)

    try:
        level = int(level_raw)
        k = int(k_raw)
    except (TypeError, ValueError):
        return jsonify({"error": "level and k must be integers."}), 400

    if not query:
        return jsonify({"error": "Missing query."}), 400
    if level < 1 or level > 4:
        return jsonify({"error": "level must be between 1 and 4."}), 400
    if k < 1 or k > 100:
        return jsonify({"error": "k must be between 1 and 100."}), 400

    include_ctx = data.get("include_context_debug", True)
    if isinstance(include_ctx, str):
        include_ctx = include_ctx.lower() in ("1", "true", "yes")

    try:
        result = run_rag_query(
            query, level=level, k=k, include_context_debug=bool(include_ctx)
        )
    except Exception as e:
        return jsonify({"error": f"RAG query failed: {e!s}"}), 500

    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@app.get("/debug_retrieval")
def debug_retrieval():
    """
    Inspect retrieval only (no LLM): validates chunks and shows previews.
    Query params: q (required), k (optional, default RAG_TOP_K).
    """
    q = (request.args.get("q") or "").strip()
    k_raw = request.args.get("k", str(RAG_TOP_K))
    try:
        k = int(k_raw)
    except ValueError:
        return jsonify({"error": "k must be an integer."}), 400
    if not q:
        return jsonify({"error": "Missing q parameter."}), 400
    if k < 1 or k > 100:
        return jsonify({"error": "k must be between 1 and 100."}), 400

    if not chroma_db_exists():
        return jsonify({"error": "No vector database yet."}), 400
    try:
        vector_db = get_vector_store()
    except Exception as e:
        return jsonify({"error": f"Could not open Chroma: {e!s}"}), 500

    docs = deduplicate_docs(retrieve_similar_chunks(vector_db, q, k))
    if not docs:
        return jsonify(
            {
                "query": q,
                "k": k,
                "num_docs": 0,
                "context_validation": {
                    "context_valid": False,
                    "note": "No documents returned.",
                },
            }
        )

    context_text, sources_mapping, sources_detail = build_numbered_context(docs)
    validation = analyze_context_for_debug(docs, context_text)
    return jsonify(
        {
            "query": q,
            "k": k,
            "num_docs": len(docs),
            "sources_mapping": sources_mapping,
            "sources_detail": sources_detail,
            "context_validation": validation,
            "context_text_sample": context_text[:2500],
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
