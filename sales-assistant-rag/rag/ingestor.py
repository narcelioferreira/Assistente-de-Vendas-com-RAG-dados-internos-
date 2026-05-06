"""
Módulo de ingestão de dados: carrega CSV e TXT, faz chunking.
"""
import os
import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_csv_as_documents(filepath: str, source_label: str) -> list[Document]:
    """Converte cada linha de um CSV em um Document do LangChain."""
    df = pd.read_csv(filepath)
    documents = []
    for _, row in df.iterrows():
        # Monta um texto legível com todos os campos da linha
        content = "\n".join(
            f"{col}: {val}" for col, val in row.items() if pd.notna(val)
        )
        doc = Document(
            page_content=content,
            metadata={"source": source_label, "file": os.path.basename(filepath)},
        )
        documents.append(doc)
    return documents


def load_txt_as_documents(filepath: str, source_label: str) -> list[Document]:
    """Carrega um arquivo TXT e divide em chunks semânticos."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n---\n\n", "\n\n", "\n", " "],
    )
    chunks = splitter.create_documents(
        [text],
        metadatas=[{"source": source_label, "file": os.path.basename(filepath)}],
    )
    return chunks


def load_all_documents(data_dir: str = "data") -> list[Document]:
    """Carrega todos os documentos da pasta data/."""
    all_docs = []

    clientes_path = os.path.join(data_dir, "clientes.csv")
    if os.path.exists(clientes_path):
        docs = load_csv_as_documents(clientes_path, "Base de Clientes")
        all_docs.extend(docs)
        print(f"✅ Clientes carregados: {len(docs)} registros")

    propostas_path = os.path.join(data_dir, "propostas.csv")
    if os.path.exists(propostas_path):
        docs = load_csv_as_documents(propostas_path, "Propostas Comerciais")
        all_docs.extend(docs)
        print(f"✅ Propostas carregadas: {len(docs)} registros")

    interacoes_path = os.path.join(data_dir, "interacoes.txt")
    if os.path.exists(interacoes_path):
        docs = load_txt_as_documents(interacoes_path, "Histórico de Interações")
        all_docs.extend(docs)
        print(f"✅ Interações carregadas: {len(docs)} chunks")

    print(f"\n📦 Total de documentos no índice: {len(all_docs)}")
    return all_docs
