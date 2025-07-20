from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv()
# ✅ Define state schema
class RAGState(TypedDict):
    question: Optional[str]
    context: Optional[str]
    answer: Optional[str]
    citations: Optional[List[str]]
    chat_history: Optional[List[tuple[str, str]]]


def create_rag_graph(vectorstore, bm25):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    def retrieve_step(state: RAGState) -> RAGState:
        question = state["question"]
        chat_history = state.get("chat_history", [])

        vector_results = vectorstore.similarity_search(question, k=4)
        bm25_results = bm25.get_relevant_documents(question)

        combined = vector_results + bm25_results
        seen = set()
        unique_docs = []
        for doc in combined:
            key = doc.page_content.strip()[:50]
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)

        context = ""
        sources = []
        for doc in unique_docs:
            page = doc.metadata.get("page", "N/A")
            source = doc.metadata.get("source", "unknown.pdf")
            sources.append(f"{source}, page {page}")
            context += f"\n[Source: {source}, page {page}]\n{doc.page_content}\n"
    
        return {
            "question": question,
            "context": context,
            "answer": None,         # Add this line
            "citations": sources,
            "chat_history": chat_history
        }
    
    def generate_step(state: RAGState) -> RAGState:
        question = state["question"]
        context = state["context"]
        sources = state["citations"]
        chat_history = state.get("chat_history") or []

        # Format history
        formatted_history = ""
        for role, msg in chat_history:
            prefix = "User" if role == "user" else "AI"
            formatted_history += f"{prefix}: {msg}\n"
            
        prompt = f"""You are a helpful assistant. Use the context below to answer the question. You have to cite the source and source text if you got the answer from the context.

Context:
{context}

Conversation so far:
{formatted_history}

Latest Question:
{question}

Answer:"""
        answer = llm.invoke(prompt).content
        if not isinstance(answer, str):
            answer = str(answer)
        return {
            "question": question,
            "context": context,
            "answer": answer,
            "citations":sources,
            "chat_history": chat_history + [("user", question or ""), ("ai", answer or "")] # Return all keys
        }

    # ✅ Fixed: Add state schema
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_step)
    graph.add_node("generate", generate_step)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.set_finish_point("generate")

    return graph.compile()
