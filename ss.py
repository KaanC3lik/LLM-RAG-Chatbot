import streamlit as st
from rag_utils import load_and_split_pdf, embed_documents
from agent_graph import create_rag_graph
import tempfile
import os

st.set_page_config(page_title="LangGraph RAG Chat", layout="wide")
st.title("📄 RAG Chat with Gemini and LangGraph")

# File uploader
uploaded_files = st.file_uploader("Upload PDF file(s)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    all_docs = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name
        docs = load_and_split_pdf(tmp_path)
        all_docs.extend(docs)
        os.remove(tmp_path)

    vectorstore = embed_documents(all_docs)
    rag_app = create_rag_graph(vectorstore)

    # User input
    question = st.text_input("Ask a question about your documents:")

    if question:
        # Run LangGraph RAG with question only
        state = rag_app.invoke({"question": question})

        # Retrieve chat history from LangGraph state
        chat_history = state.get("chat_history", [])

        # Show chat history
        for msg in chat_history:
            if msg[0] != "user":
                st.markdown(f"**User:** {msg[1]}")
            else:
                st.markdown(f"**AI:** {msg[1]}")
else:
    st.info("Please upload at least one PDF file to start.")
