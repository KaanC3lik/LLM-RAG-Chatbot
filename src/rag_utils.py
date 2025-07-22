from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from langchain.retrievers import BM25Retriever

import os
import re
from dotenv import load_dotenv

load_dotenv()

def clean_text(text: str) -> str:
    """Basic text cleaning: remove headers/footers/extra whitespace."""
    # Remove multiple newlines and excess whitespace
    text = re.sub(r"\n\s*\n", "\n", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def load_and_split_pdf(file):

    loader = PyMuPDFLoader(file)
    docs = loader.load()

    cleaned_docs = []
    for doc in docs:
        cleaned_content = clean_text(doc.page_content)
        metadata = {
            "source": os.path.basename(file),
            "page": doc.metadata.get("page", "unknown") + 1,
        }
        cleaned_docs.append(Document(
            page_content=cleaned_content,
            metadata=metadata
        ))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    return text_splitter.split_documents(cleaned_docs)


def embed_documents(docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",
transport="rest")
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    return vectorstore

def retrieve_documents(vectorstore, query):
    docs = vectorstore.similarity_search(query, k=4)
    return docs

def build_hybrid_retriever(docs):

    # 2. BM25 retriever
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 4

    return embed_documents(docs), bm25