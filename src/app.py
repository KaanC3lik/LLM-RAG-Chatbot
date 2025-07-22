import os
import tempfile
import streamlit as st
from rag_utils import load_and_split_pdf, embed_documents, build_hybrid_retriever
from rag_app import create_rag_graph
from langchain_core.messages import HumanMessage, AIMessage

import uuid  # for generating unique conversation IDs

if "conversations" not in st.session_state:
    st.session_state.conversations = {}

# Initialize current_conversation_id if not already set
if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = 1

# Make sure the conversation dict entry exists
if st.session_state.current_conversation_id not in st.session_state.conversations:
    st.session_state.conversations[st.session_state.current_conversation_id] = []
    
    

st.set_page_config(page_title="Multi-PDF RAG Agent", layout="centered")
st.title("📄 Multi-PDF RAG Agent")

# File uploader
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if "rag_app" not in st.session_state:
    st.session_state.rag_app = None


upload_dir = tempfile.mkdtemp()
# Only process PDFs if new files are uploaded
if uploaded_files and st.session_state.rag_app is None:
    all_docs = []

    with st.spinner("Processing PDFs..."):
        for file in uploaded_files:
            file_path = os.path.join(upload_dir, file.name)  # Keep original file name
            with open(file_path, "wb") as f:
                f.write(file.getvalue())
            docs = load_and_split_pdf(file_path)
            all_docs.extend(docs)
        vectorstore, bm25 = build_hybrid_retriever(all_docs)
        st.session_state.rag_app = create_rag_graph(vectorstore, bm25)
    st.success("PDFs processed!")

if st.session_state.rag_app is not None:
    # Show chat history
    conversation_changed = False
    with st.sidebar:
        if st.button("➕ New Conversation"):
            new_id = st.session_state.current_conversation_id + 1
            st.session_state.conversations[new_id] = []
            st.session_state.current_conversation_id = new_id
            st.session_state.conversation_selector = new_id

        # Initialize selectbox stored value if missing
        if "conversation_selector" not in st.session_state:
            st.session_state.conversation_selector = st.session_state.current_conversation_id

        # Dropdown to switch between conversations
        st.markdown("### 🧠 Conversations")
        conversation_ids = list(st.session_state.conversations.keys())
        selected = st.selectbox("Select conversation", conversation_ids, key="conversation_selector", format_func=lambda x: f"Conversation   {conversation_ids.index(x)+1}")

        # Sync current_conversation_id if selectbox changes
        if selected != st.session_state.current_conversation_id:
            st.session_state.current_conversation_id = selected

    if prompt := st.chat_input("Ask a question based on the PDFs:"):
        question = prompt

    
    
    if "question" in locals():
        with st.spinner("Getting answer from Gemini..."):
            
             # Append user question to chat history

            # Run LangGraph RAG with question only
            state = st.session_state.rag_app.invoke(
                {
                "messages": [question],
                "context": None,
                "citations":None
                },
                config={"configurable": {"thread_id": st.session_state.current_conversation_id}},
            )
            
            # Append AI answer to chat history
            
                                                                                            
            st.session_state.conversations[st.session_state.current_conversation_id] = (state["messages"])

chat_history = st.session_state.conversations[st.session_state.current_conversation_id]

if chat_history != []:

    chat_history = st.session_state.conversations[st.session_state.current_conversation_id]
    flat_messages = []

    for item in chat_history:
        if isinstance(item, list):
            flat_messages.extend(item)
        else:
            flat_messages.append(item)
    # Sonra yazdır
    for msg in flat_messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("ai"):
                st.markdown(msg.content)

else:
    st.info("Please upload at least one PDF file to start.")