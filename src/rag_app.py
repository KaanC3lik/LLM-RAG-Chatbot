import os

from typing import Annotated, List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    context: Optional[str]
    citations: Optional[List[str]]

graph_builder = StateGraph(State)

tool = TavilySearch(
    max_results=2,
    topic="general",
    # include_answer=False,
    # include_raw_content=False,
    # include_images=False,
    # include_image_descriptions=False,
    # search_depth="basic",
    # time_range="day",
    # include_domains=None,
    # exclude_domains=None
)
tools = [tool]
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def create_rag_graph(vectorstore, bm25):

    def retrieve_step(state: State) -> State:
        messages = state["messages"]

        user_messages = [m.content for m in messages if m.type == "human"]
        if not user_messages:
            raise ValueError("No user message found in state['messages']")

        question = user_messages[-1]

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
            "messages": messages,
            "context": context,
            "citations": sources,
        }
    
    def generate_step(state: State) -> State:
        question = state["messages"]
        context = state["context"]
        sources = state["citations"]
            
        prompt = f"""You are a helpful assistant. Use the context below to answer the question. You have to cite the source and source text if you got the answer from the context. You don't have to use the context if it is irrelevant.

Context:
{context}

Conversation history:
{question}"""
        
        return {
            "messages": [llm.invoke(prompt)],
            "context": context,
            "citations":sources,
        }

    graph_builder.add_node("retrieve", retrieve_step)
    graph_builder.add_node("generate", generate_step)

    graph_builder.set_entry_point("retrieve")
    graph_builder.add_edge("retrieve", "generate")
    graph_builder.set_finish_point("generate")
    
    memory = MemorySaver()
    
    return graph_builder.compile(memory)