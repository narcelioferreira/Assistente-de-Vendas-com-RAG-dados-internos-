"""
Módulo da chain RAG: conecta retriever + LLM para gerar respostas contextualizadas.
"""
import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from rag.ingestor import load_all_documents
from rag.vector_store import build_vector_store, get_retriever


SYSTEM_PROMPT = """Você é um assistente de vendas inteligente e experiente, especializado em ajudar vendedores a fechar negócios e gerenciar relacionamentos com clientes.

Você tem acesso ao histórico completo de clientes, propostas comerciais e interações da empresa.

INSTRUÇÕES:
- Responda SEMPRE em português brasileiro
- Seja direto, objetivo e útil para o vendedor
- Quando mencionar valores, use o formato brasileiro (R$ X.XXX,XX)
- Destaque informações críticas como status de negociação, riscos de churn e oportunidades
- Se sugerir próximos passos, seja específico e acionável
- Se não encontrar informação no contexto, diga claramente e ofereça o que você sabe

CONTEXTO DISPONÍVEL (dados internos da empresa):
{context}

Responda à pergunta do vendedor com base neste contexto."""

HUMAN_PROMPT = "{question}"


def format_docs(docs: list[Document]) -> str:
    """Formata os documentos recuperados em texto para o prompt."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Fonte desconhecida")
        formatted.append(f"[Fonte {i}: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


@st.cache_resource(show_spinner=False)
def build_rag_chain(_retriever):
    """Constrói a chain RAG (cached no Streamlit)."""
    groq_api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError(
            "❌ GROQ_API_KEY não encontrada. Configure em .streamlit/secrets.toml "
            "ou como variável de ambiente."
        )

    llm = ChatGroq(
        api_key=groq_api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=1500,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])

    chain = (
        {
            "context": _retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


@st.cache_resource(show_spinner=False)
def initialize_rag_system():
    """Inicializa todo o pipeline RAG (cached — roda só uma vez por sessão)."""
    data_dir = "data"
    documents = load_all_documents(data_dir)
    vector_store = build_vector_store(documents)
    retriever = get_retriever(vector_store, k=5)
    chain = build_rag_chain(retriever)
    return chain, retriever, len(documents)
