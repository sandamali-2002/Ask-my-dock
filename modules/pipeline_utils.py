from modules.retriver import get_ensemble_retriever
from modules.reranker import rerank_documents

def retrieve_and_rerank(inputs: dict):
    query = inputs["question"]

    retriever = get_ensemble_retriever(k=5)
    docs = retriever.invoke(query)

    docs = rerank_documents(query, docs, top_k=3)

    context = "\n\n".join([doc.page_content for doc in docs])

    return {
        "context": context,
        "question": query,
        "docs": docs
    }