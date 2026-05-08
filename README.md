# Ask-my-dock

Ask-my-dock is a small Retrieval-Augmented Generation (RAG) prototype that lets you ask natural-language questions over a local set of PDF documents.

What happens in this project
- Ingest: PDF files placed in the `data/` folder are loaded and split into smaller chunks by `modules/ingest.py`.
- Embed & Persist: Each chunk is embedded using OpenAI embeddings and stored in a persistent Chroma vector store (folder `db/`).
- Retrieve: Given a user query, an ensemble retriever (semantic + keyword) pulls relevant chunks from Chroma and BM25 (see `modules/retriver.py`).
- Rerank: Retrieved chunks are reranked using a CrossEncoder reranker (`sentence-transformers`) to pick the most relevant context (`modules/reranker.py`).
- Answer: A prompt pipeline composes the chosen context and sends it to an LLM (OpenAI via LangChain) to generate a concise answer (`modules/rag_chain.py`).

Key files
- `modules/ingest.py`: Loads PDFs from `data/`, splits into chunks, creates embeddings, and stores them in Chroma.
- `modules/retriver.py`: Builds an ensemble retriever that combines semantic search (Chroma/OpenAI embeddings) and keyword search (BM25).
- `modules/reranker.py`: Reranks the candidate chunks using a CrossEncoder model.
- `modules/pipeline_utils.py`: Glue code: retrieve, rerank, and prepare the context for the LLM.
- `modules/rag_chain.py`: Prompt template and LLM invocation. Use `answer_question(query)` to get answers.
- `test.py`: Small example showing how to call the chain.

Technologies used and why
- Python: Primary language for LangChain and related ML libraries.
- LangChain: Composes prompts, chains, and runnable pipelines in a clean way.
- OpenAI (Chat / Embeddings): High-quality language model and embeddings used for generation and semantic search.
- Chroma: Lightweight, persistent vector store (stored in `db/`) that integrates well with LangChain.
- FAISS (`faiss-cpu` listed in requirements): optional/local vector index implementation for fast nearest-neighbor search.
- `rank_bm25`: Provides fast keyword-based retrieval (BM25) which complements semantic search.
- `sentence-transformers` / CrossEncoder: Used to rerank retrieved passages for higher precision before generation.
- `streamlit`: Included for building a simple web UI if desired (not required by the core pipeline).
- `pypdf`: PDF loader for ingesting document content.

Why this architecture?
- Combining semantic search (embeddings) with keyword search (BM25) helps cover cases where embeddings miss precise keywords or terminology.
- Reranking with a CrossEncoder improves answer relevance by rescoring candidates with a model trained for ranking.
- Persisting embeddings to Chroma avoids re-embedding documents on every run and enables fast local retrieval.
- Using LangChain and small modular scripts keeps the pipeline readable and easy to extend.

Quick start
1. Create a Python virtual environment and install requirements:

```bash
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

2. Set your OpenAI API key in environment or in a `.env` file at the project root:

```
OPENAI_API_KEY=sk-...
```

3. Place PDFs into the `data/` folder, then run ingestion to build the vector DB:

```bash
python modules/ingest.py
```

4. Query the system (simple example):

```bash
python test.py
```

Notes & next steps
- `app.py` is currently empty; you can build a Streamlit UI that calls `modules.rag_chain.answer_question()` for an interactive interface.
- If you want local-only embeddings (no OpenAI), consider using `sentence-transformers` to create embeddings and switch the embedding function.
- To speed up retrieval for large datasets, evaluate FAISS-backed indexes or optimize chunk sizes.

If you want, I can: (a) add a Streamlit UI, (b) implement local `sentence-transformers` embeddings, or (c) add a simple `README` badge and usage examples. Which would you like next?
