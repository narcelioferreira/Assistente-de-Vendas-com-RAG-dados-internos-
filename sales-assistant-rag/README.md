# 🎯 Sales Assistant com RAG

Assistente de vendas inteligente que usa **Retrieval-Augmented Generation (RAG)** para responder perguntas sobre clientes, propostas e histórico de negociações — com base em dados internos reais (simulados).

> **Projeto de portfólio** demonstrando arquitetura RAG completa com deploy gratuito no Streamlit Cloud.

---

## 🖥️ Demo ao vivo

[**→ Acesse o app aqui**](https://ecxfafgmkmt87fod3uod6u.streamlit.app/)

---

## 💡 Problema que resolve

Vendedores esquecem detalhes críticos sobre clientes:
- Qual era o status da última proposta?
- Quem é o decisor real nessa empresa?
- Por que aquele cliente estava em risco de churn?
- Quais oportunidades de upsell existem?

O **Sales Assistant RAG** responde tudo isso em segundos, consultando os dados internos.

---

## 🧠 Arquitetura RAG

```
Dados (CSV/TXT)
      │
      ▼
  Chunking
(RecursiveTextSplitter)
      │
      ▼
  Embeddings
(HuggingFace MiniLM-L6)
      │
      ▼
  FAISS Index
(armazenamento vetorial)
      │
      ▼
  Retrieval (top-5)
      │
      ▼
  LLM (Groq/Llama 3.1)
      │
      ▼
  Resposta contextualizada
```

---

## 🛠️ Stack Técnica

| Componente | Tecnologia | Custo |
|---|---|---|
| **LLM** | Groq API + Llama 3.1 8B Instant | Grátis |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | Grátis (local) |
| **Vector DB** | FAISS | Grátis (local) |
| **Framework** | LangChain | Open source |
| **UI** | Streamlit | Grátis |
| **Deploy** | Streamlit Cloud | Grátis |

**Custo total: R$ 0,00/mês** 🎉

---

## 📂 Estrutura do Projeto

```
sales-assistant-rag/
├── app.py                    # Interface Streamlit
├── requirements.txt
├── .gitignore
├── README.md
├── .streamlit/
│   └── secrets.toml          # API keys (não commitar!)
├── rag/
│   ├── __init__.py
│   ├── ingestor.py           # Carregamento e chunking de dados
│   ├── vector_store.py       # FAISS + embeddings
│   └── chain.py              # Pipeline RAG completo
└── data/
    ├── clientes.csv           # 10 clientes simulados (B2B)
    ├── propostas.csv          # 6 propostas comerciais
    └── interacoes.txt         # Histórico de reuniões e notas
```

---

## 🚀 Como rodar localmente

### 1. Clone o repositório
```bash
git clone https://github.com/seu-usuario/sales-assistant-rag.git
cd sales-assistant-rag
```

### 2. Instale as dependências
```bash
pip install -r requirements.txt
```

### 3. Configure a API Key (Groq — gratuita)
- Acesse [console.groq.com](https://console.groq.com)
- Crie uma conta e gere uma API Key (gratuita)
- Crie o arquivo `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "sua_chave_aqui"
```

### 4. Rode o app
```bash
streamlit run app.py
```

---

## ☁️ Deploy no Streamlit Cloud (gratuito)

1. **Push para o GitHub** (sem o `secrets.toml`)
2. Acesse [share.streamlit.io](https://share.streamlit.io)
3. Conecte seu repositório
4. Em **Advanced Settings → Secrets**, adicione:
   ```toml
   GROQ_API_KEY = "sua_chave_aqui"
   ```
5. Clique em **Deploy** ✅

---

## 💬 Exemplos de perguntas

- *"Qual o status da proposta para a Construtora Horizonte?"*
- *"Quais clientes estão em risco de churn?"*
- *"Me dê um briefing completo sobre o Grupo Meridian"*
- *"Quais são as maiores oportunidades de upsell?"*
- *"Como está a negociação com o André da Logística Express?"*
- *"Quem indicou a Saúde Premium? Qual o histórico?"*
- *"Quais propostas vencem esse mês?"*
- *"Me fale sobre os prospects do agronegócio"*

---

## 🎯 O que este projeto demonstra

✅ **RAG funcionando de ponta a ponta** (ingestão → embeddings → retrieval → LLM)  
✅ **Engenharia de dados** (parsing de CSV/TXT, chunking inteligente)  
✅ **LangChain** (chains, retrievers, prompts, output parsers)  
✅ **Vector databases** (FAISS com busca por similaridade)  
✅ **Deploy real** em produção (Streamlit Cloud, 100% gratuito)  
✅ **Aplicação com valor de negócio** (problema real de vendas)  
✅ **Código limpo e modular** (separação de responsabilidades)  

---

## 🔧 Possíveis extensões

- [ ] Upload dinâmico de PDFs (novos clientes em tempo real)
- [ ] Autenticação por vendedor (cada um vê só sua carteira)
- [ ] Integração com CRM real (HubSpot, Salesforce via API)
- [ ] Modo de análise: gera relatório semanal automático
- [ ] Alerta proativo de clientes em risco

---

## 👨‍💻 Autor

Feito como projeto de portfólio para demonstrar habilidades em **LLM Engineering, RAG e MLOps**.

*Dados utilizados são 100% fictícios e criados para fins de demonstração.*
