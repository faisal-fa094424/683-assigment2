# Assignment 2 — RAG Enhancement (CMPE 682/683/782/783)

This repository contains work in progress for **Assignment 2: Enhancing Your AI Application** (Spring 2026, Qatar University), as specified in **`Assignment 2 V4.pdf`**. The implementation follows **Track A: Retrieval-Augmented Generation (RAG)** and extends themes from Assignment 1 (prompting, level calibration, citations) with a document-grounded research assistant focused on **power systems and related AI / LLM literature**.

> **Status:** The assignment is not finished. Some notebook sections (for example Part C metrics and consolidation of the full 25-case suite) are still being developed.

## Course deliverables (from the brief)

Per the assignment document, the full submission typically includes:

1. **GitHub repository** — documented code (this repo).
2. **Deployed app** — e.g. Streamlit or Gradio (to be completed per course requirements).
3. **Technical report** — 4–6 pages, IEEE format.
4. **Demo video** — 3–5 minutes.

This README focuses on what is implemented in code here and how to run it.

## What this project implements (aligned with Part B — Track A)

| Component | Assignment requirement | Implementation in this repo (notebook) |
|-----------|------------------------|----------------------------------------|
| **3.1.1** Document collection & processing | 10–20 documents; chunking ~256–512 tokens, overlap ~50–100; metadata (source, page, etc.) | PDFs under `research_papers/`; `RecursiveCharacterTextSplitter` with tiktoken encoder (**512** tokens, **75** overlap); `PyMuPDFLoader` preserves source and page metadata. |
| **3.1.2** Embedding & retrieval | Embeddings + vector DB; similarity search with **k = 3, 5, 7**; optional hybrid retrieval | OpenAI `text-embedding-3-small` + **Chroma** (`chroma_db/`); `similarity_search`; **bonus**: `EnsembleRetriever` (dense + **BM25**). |
| **3.1.3** Grounded generation | Context-only answers; citations; explicit handling when context is insufficient | System prompt with chunk-indexed citations, refusal phrase (*“I don’t have information about this in my knowledge base.”*), and chain-of-thought in `<thinking>` tags; builds on A1-style **explanation level** (1–4). |

## Repository layout

| Path | Purpose |
|------|---------|
| `Assignment 2 V4.pdf` | Official assignment specification (V4). |
| `Assingment-2 .ipynb` | Main Jupyter notebook: ingestion, vector store, retrieval, RAG chains, evaluation experiments, and plotting. |
| `research_papers/` | Curated PDF corpus for the RAG knowledge base. |
| `chroma_db/` | Persisted Chroma store (ignored by git via `.gitignore` after local runs). |
| `.env` | API keys (ignored by git); see below. |
| `requirements.txt` | Present in the repo; **the notebook installs its own dependency set via `pip`** — use that line or reconcile with your environment. |

## Environment and secrets

1. Create a virtual environment (recommended).
2. Install packages using the **`pip install`** cell at the top of `Assingment-2 .ipynb` (LangChain, Chroma, OpenAI, `rank_bm25`, etc.).
3. Create a **`.env`** file in the project root (never commit it). The notebook expects variables such as:
   - **`OpenAI_API_KEY`** — OpenAI API access for embeddings and `gpt-4o-mini`.
   - **`Open_Router_API_KEY`** — used for comparison runs via OpenRouter (e.g. Llama 3.3 70B).

Some cells use **Google Colab-style paths** (`/content/research_papers`, `/content/chroma_db`). For local runs, use paths relative to this repo (e.g. `research_papers`, `chroma_db`) consistently with the earlier cells in the same notebook.

## Part C — evaluation (work in progress)

The assignment asks for:

- **25 test cases** — 15 carried from A1 plus **10** new cases targeting RAG (precision, cross-document, multi-doc, etc.).
- **Metrics** — A1 criteria for continuity, plus RAG-specific metrics: **groundedness**, **citation accuracy**, **retrieval relevance** (and optional extraction-style metrics in your analysis).
- **At least three visualizations** and a short written analysis in the report.

The notebook contains dataset definitions, batch runs that export **CSV** results, comparison logic (**OpenAI** vs **OpenRouter / Llama**), optional **Mendeley-style reference appendices** mapped from PDF filenames, and **matplotlib/seaborn** figures. Sections for formal metric tables and full end-to-end consolidation may still need to be completed for submission.

## Hybrid retrieval (bonus)

The notebook implements **dense vector retrieval + BM25** with `EnsembleRetriever` and configurable weights, satisfying the assignment’s hybrid retrieval bonus item.

## References

- Course assignment: **`Assignment 2 V4.pdf`** in this repository.
- Huyen, *AI Engineering* — MVP progression (prompting → RAG → agents → fine-tuning), as cited in the assignment overview.

---

*This README describes the current state of the project only; it does not replace the official PDF instructions or rubric.*
