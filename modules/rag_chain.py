from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from modules.pipeline_utils import retrieve_and_rerank

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

prompt = ChatPromptTemplate.from_messages([
    (
        "human", """
Answer using ONLY the context.
Treat minor spacing/case differences in names as equivalent (example: "codeprolk" and "CodePRO LK").
If not found, say "I don't know".

Question: {question}
Context: {context}

"""
    )
])

rag_chain = (
    RunnableLambda(retrieve_and_rerank)
    | prompt
    | llm
)

def answer_question(query: str):
    data = retrieve_and_rerank({"question": query})
    response = rag_chain.invoke({"question": query})

    unique_sources = []
    for doc in data["docs"]:
        source = doc.metadata.get("source", "unknown")
        if source not in unique_sources:
            unique_sources.append(source)

    sources = "\n".join(unique_sources)

    return f"{response.content}\n\nSources:\n{sources}"





