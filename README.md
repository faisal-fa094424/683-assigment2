# Verifiable RAG-Based Research Paper Summarizer & Explainer

## Project Overview
This repository documents the evolution of an AI-driven academic tool from a simple prompting baseline in Assignment 1 to a high-fidelity, citation-aware Retrieval-Augmented Generation (RAG) system for Assignment 2. The project focuses on transforming a standard LLM into a reliable research assistant capable of providing level-calibrated summaries and verifiable academic citations.

## System Architecture

![RAG Architecture](architecture_diagram.png)
*Figure: RAG pipeline with manual chunk grounding and citation enforcement.*

## Core Assignment 2 Enhancements (Track A)

**Persistent Knowledge Base:** Transitioned from processing single documents to a vectorized library of 15 core research papers using ChromaDB.

**Advanced Chunking Strategy:** Implemented recursive token-based splitting (512-token chunks with 75-token overlap) to maintain technical context.

**Manual Citation Injection:** Developed a manual re-numbering step for retrieved passages (--- CHUNK [i] ---) to force the LLM to provide exact numeric citations.

**Mendeley Reference Mapping:** Integrated an advanced REFERENCE_DB dictionary that automatically translates raw filenames into professionally formatted academic bibliographies.

**Dual-Model Evaluation:** Conducted comparative testing between GPT-4o-mini and Llama-3.3-70B across 25 diverse test cases.

## Repository Contents

Assingment_2_with_outputs_.ipynb: The primary Python notebook containing environment setup, RAG initialization, extended evaluation logs, and the Mendeley citation system.

research_papers/: A directory containing the 15 research PDFs used as the foundation for the vectorized knowledge base.

architecture_diagram.png:  A visual representation of the RAG pipeline and system components.

## Setup and Run Instructions

### 1. Prerequisites

You will need API keys for the following:

OpenAI API Key: Required for the text-embedding-3-small model and GPT-4o-mini.

OpenRouter API Key: Required to access the Llama-3.3-70B-Instruct model.

### 2. Installation

Run the following command in your environment to install necessary dependencies:

```bash
pip install -U langchain langchain-community langchain-openai langchain-chroma pypdf pymupdf chromadb tiktoken
