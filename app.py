import os
import re
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OpenAI_API_KEY")

app = Flask(__name__)

papers_folder = 'research_papers'
VECTOR_DB_NAME = "chroma_db"
COLLECTION = "research_papers"


RAG_PROMPT = """You are an academic research assistant specializing in Power Systems and AI.

Answer the user's question using ONLY the CONTEXT chunks provided below.
Each chunk header includes its source paper filename and page number.

Rules:
1. Explanation Level: {level} (1=PhD, 2=Master, 3=Bachelor, 4=School). Adjust vocabulary accordingly.
2. Cite sources using the chunk number shown in the header, e.g. [1], [2].
3. If the context chunks are completely unrelated to the question, say: "I don't have information about this in my knowledge base."
4. If the context is partially relevant, answer with what is available.

CONTEXT:
{context}
"""


# ── helpers ──────────────────────────────────────────────────────────────

def is_reference_page(text):
    lines = [l for l in text.strip().splitlines() if l.strip()]
    if len(lines) < 3:
        return False
    pat = re.compile(r'^\s*(?:\[?\d{1,3}\]?\.?\s+[A-Z]|https?://\S+|doi[\s:.]+10\.\S+)', re.IGNORECASE)
    return sum(1 for l in lines if pat.match(l)) / len(lines) > 0.4


def pdf_to_chunks(pdf_path):
    docs = PyMuPDFLoader(pdf_path).load()
    docs = [d for d in docs if not is_reference_page(d.page_content)]
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4o-mini", chunk_size=512, chunk_overlap=75
    )
    return splitter.split_documents(docs)


def get_chroma():
    emb = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(persist_directory=str(CHROMA_DIR), embedding_function=emb, collection_name=COLLECTION)


def add_to_chroma(chunks):
    emb = OpenAIEmbeddings(model="text-embedding-3-small")
    if not any(CHROMA_DIR.iterdir()):
        Chroma.from_documents(chunks, emb, persist_directory=str(CHROMA_DIR), collection_name=COLLECTION)
    else:
        get_chroma().add_documents(chunks)


def build_context(docs):
    context = ""
    sources = ""
    for i, d in enumerate(docs, 1):
        src = os.path.basename(d.metadata.get("source", "Unknown"))
        page = d.metadata.get("page", "N/A")
        context += f"\n--- CHUNK [{i}] | Source: {src} (Page {page}) ---\n{d.page_content}\n"
        sources += f"[{i}] {src} (Page {page})\n"
    return context, sources.strip()


# ── routes ───────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return render_template("index.html")


@app.post("/add_new_paper")
def add_new_paper():
    if "file" not in request.files:
        return jsonify({"added": False, "reason": "No file uploaded."}), 400
    f = request.files["file"]
    if not f or not f.filename.lower().endswith(".pdf"):
        return jsonify({"added": False, "reason": "Only .pdf files are supported."}), 400

    target = PAPERS_DIR / Path(f.filename).name
    if target.exists():
        return jsonify({"added": False, "reason": "File already exists in research_papers."})

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            f.save(tmp.name)
            tmp_path = tmp.name

        chunks = pdf_to_chunks(tmp_path)
        if not chunks:
            return jsonify({"added": False, "reason": "Could not extract text from this PDF."})

        for c in chunks:
            c.metadata["source"] = str(target)

        add_to_chroma(chunks)
        shutil.move(tmp_path, target)
    except Exception as e:
        return jsonify({"added": False, "reason": str(e)})
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return jsonify({"added": True})


@app.post("/query_research_papers")
def query_research_papers():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    level = int(data.get("level", 2))

    if not query:
        return jsonify({"error": "Missing query."}), 400

    if not any(CHROMA_DIR.iterdir()):
        return jsonify({"error": "No papers indexed yet. Upload a paper first."}), 400

    docs = get_chroma().similarity_search(query, k=20)
    if not docs:
        return jsonify({"answer": "I don't have information about this in my knowledge base.", "sources": ""})

    context, sources = build_context(docs)

    system_msg = RAG_PROMPT.replace("{context}", context).replace("{level}", str(level))
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    response = llm.invoke([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": query},
    ])

    return jsonify({"answer": response.content.strip(), "sources": sources})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
