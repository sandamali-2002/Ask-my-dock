import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_openai import OpenAIEmbeddings

# Allow running this module directly while importing root-level config.py
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config  # noqa: F401  # Loads OPENAI_API_KEY from environment/.env


def _load_documents_from_chroma(vectorstore: Chroma) -> list[Document]:
    """Rebuild Document objects from persisted Chroma entries."""
    data = vectorstore.get(include=["documents", "metadatas"])
    docs = data.get("documents", []) or []
    metas = data.get("metadatas", []) or []

    if len(metas) < len(docs):
        metas = metas + [{}] * (len(docs) - len(metas))

    return [
        Document(page_content=doc, metadata=meta or {})
        for doc, meta in zip(docs, metas)
        if doc
    ]


def get_ensemble_retriever(k: int = 2) -> EnsembleRetriever:
    """Use persisted chunks from Chroma for both semantic and keyword retrieval."""
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory="db", embedding_function=embeddings)

    vectorstore_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    stored_docs = _load_documents_from_chroma(vectorstore)
    if not stored_docs:
        raise RuntimeError("No documents found in db. Run modules/ingest.py first.")

    keyword_retriever = BM25Retriever.from_documents(stored_docs)
    keyword_retriever.k = k

    return EnsembleRetriever(
        retrievers=[vectorstore_retriever, keyword_retriever],
        weights=[0.5, 0.5],
    )

