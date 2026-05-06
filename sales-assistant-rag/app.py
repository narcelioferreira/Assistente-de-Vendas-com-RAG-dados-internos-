"""
Sales Assistant com RAG — Streamlit App
Assistente de vendas com contexto de dados internos (clientes, propostas, histórico)
"""
import streamlit as st
import time

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Assistant · RAG",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main container */
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* Header */
    .header-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(99, 179, 237, 0.2);
    }
    .header-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #63b3ed;
        margin: 0;
    }
    .header-sub {
        color: #a0aec0;
        font-size: 0.95rem;
        margin-top: 0.3rem;
    }

    /* Metric cards */
    .metric-card {
        background: #1a202c;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #63b3ed; }
    .metric-label { font-size: 0.8rem; color: #718096; margin-top: 2px; }

    /* Chat messages */
    .chat-user {
        background: #1e3a5f;
        border-left: 3px solid #63b3ed;
        border-radius: 0 10px 10px 0;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
    }
    .chat-assistant {
        background: #1a202c;
        border-left: 3px solid #48bb78;
        border-radius: 0 10px 10px 0;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
    }

    /* Sources expander */
    .source-tag {
        display: inline-block;
        background: #2d3748;
        color: #a0aec0;
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 12px;
        margin: 2px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0d1117;
    }

    /* Hide default streamlit elements */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 Sales Assistant RAG")
    st.markdown("---")

    st.markdown("### 📂 Dados carregados")
    st.markdown("""
    - 👥 **Clientes:** 10 registros
    - 📋 **Propostas:** 6 propostas
    - 💬 **Interações:** histórico completo
    """)

    st.markdown("---")
    st.markdown("### 💡 Perguntas de exemplo")

    example_questions = [
        "Qual o status da proposta para a Construtora Horizonte?",
        "Quais clientes estão em risco de churn?",
        "Me dê um briefing sobre o Grupo Meridian antes da reunião",
        "Quais são as oportunidades de upsell mais valiosas?",
        "Como está a negociação com o André da Logística Express?",
        "Quem indicou a Saúde Premium e qual o histórico?",
        "Quais propostas vencem esse mês?",
        "Me fale sobre os prospects do agronegócio",
    ]

    for q in example_questions:
        if st.button(q, use_container_width=True, key=f"ex_{q[:20]}"):
            st.session_state.example_question = q

    st.markdown("---")
    st.markdown("### 🏗️ Stack Técnica")
    st.markdown("""
    - 🧠 **LLM:** Groq (Llama 3.1 8B)
    - 📐 **Embeddings:** HuggingFace MiniLM
    - 🗄️ **Vector DB:** FAISS
    - ⛓️ **Framework:** LangChain
    - 🚀 **Deploy:** Streamlit Cloud
    """)

    st.markdown("---")
    st.caption("Projeto de portfólio · RAG com dados de vendas")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <div class="header-title">🎯 Sales Assistant com RAG</div>
    <div class="header-sub">
        Assistente inteligente que responde perguntas sobre clientes, propostas e histórico de vendas
        usando Retrieval-Augmented Generation (RAG) com dados internos simulados.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Metrics row ───────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
metrics = [
    ("10", "Clientes"),
    ("6", "Propostas"),
    ("R$ 372k", "Pipeline Total"),
    ("1", "Risco de Churn"),
    ("3", "Prospects Ativos"),
]
for col, (val, label) in zip([col1, col2, col3, col4, col5], metrics):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Initialize RAG ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_system():
    from rag.chain import initialize_rag_system
    return initialize_rag_system()


with st.spinner("🔄 Inicializando o sistema RAG (embeddings + índice FAISS)..."):
    try:
        chain, retriever, n_docs = load_system()
        if "rag_ready" not in st.session_state:
            st.session_state.rag_ready = True
    except ValueError as e:
        st.error(str(e))
        st.info("""
        **Como configurar a API Key:**
        1. Acesse [console.groq.com](https://console.groq.com) e crie uma conta gratuita
        2. Gere uma API Key
        3. Crie o arquivo `.streamlit/secrets.toml` com:
        ```toml
        GROQ_API_KEY = "sua_chave_aqui"
        ```
        """)
        st.stop()

# Small success badge
st.success(f"✅ Sistema pronto · {n_docs} documentos indexados · FAISS ativo", icon="🟢")

# ── Chat UI ───────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Handle example question click from sidebar
if "example_question" in st.session_state:
    st.session_state.pending_question = st.session_state.pop("example_question")

# Display chat history
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.write(msg["content"])
                if "sources" in msg:
                    with st.expander("📚 Fontes consultadas", expanded=False):
                        for src in msg["sources"]:
                            st.markdown(f"**[{src['source']}]**")
                            st.caption(src["content"][:300] + "...")
                            st.markdown("---")

# ── Input ─────────────────────────────────────────────────────────────────────
# Handle pending question from sidebar buttons
default_val = st.session_state.pop("pending_question", "")

user_input = st.chat_input(
    "Pergunte sobre clientes, propostas, histórico de negociações...",
)

# Use sidebar button question if no chat input
question = user_input or (default_val if default_val else None)

if question:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user", avatar="👤"):
        st.write(question)

    # Generate response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Consultando base de dados e gerando resposta..."):
            start = time.time()

            # Retrieve relevant docs for display
            relevant_docs = retriever.invoke(question)

            # Run RAG chain
            response = chain.invoke(question)

            elapsed = time.time() - start

        st.write(response)
        st.caption(f"⚡ Resposta em {elapsed:.1f}s · {len(relevant_docs)} trechos consultados")

        # Show sources
        sources = [
            {
                "source": doc.metadata.get("source", "Desconhecida"),
                "content": doc.page_content,
            }
            for doc in relevant_docs
        ]
        with st.expander("📚 Fontes consultadas", expanded=False):
            for src in sources:
                st.markdown(f"**[{src['source']}]**")
                st.caption(src["content"][:300] + "...")
                st.markdown("---")

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources,
    })

# ── Clear button ──────────────────────────────────────────────────────────────
if st.session_state.messages:
    if st.button("🗑️ Limpar conversa", use_container_width=False):
        st.session_state.messages = []
        st.rerun()
