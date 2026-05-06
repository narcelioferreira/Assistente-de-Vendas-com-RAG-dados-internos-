"""
Módulo de Vector Store: cria e gerencia o índice FAISS com embeddings locais.
"""
import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index"


@st.cache_resource(show_spinner=False)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Carrega o modelo de embeddings (cached no Streamlit)."""
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(documents: list[Document]) -> FAISS:
    """Constrói o índice FAISS a partir dos documentos."""
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store


def get_retriever(vector_store: FAISS, k: int = 5):
    """Retorna um retriever que busca os k documentos mais relevantes."""
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
