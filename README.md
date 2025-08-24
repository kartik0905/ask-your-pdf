<div align="center">

# 📄 RAG File QA Chatbot  
### Chat with your PDFs using LangChain + Streamlit + OpenAI

</div>

<p align="center">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/LangChain-1E90FF?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain"/>
  <img src="https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white" alt="OpenAI"/>
  <img src="https://img.shields.io/badge/ChromaDB-000000?style=for-the-badge&logo=databricks&logoColor=white" alt="ChromaDB"/>
  <img src="https://img.shields.io/badge/PyMuPDF-008000?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="PyMuPDF"/>
</p>

---

## 📌 Overview

This project is a **Retrieval-Augmented Generation (RAG) chatbot** built with **Streamlit + LangChain + OpenAI**.  
It allows you to **upload PDF files, query them in natural language**, and get **AI-powered answers with sources**.  

- 📄 Upload multiple PDFs  
- ✂️ Split text into smart chunks  
- 🧠 Generate embeddings with OpenAI  
- 💾 Store + retrieve chunks using ChromaDB  
- 🤖 Ask questions & get contextual answers with sources shown  

---

## ✨ Features

- 📂 **Multi-PDF Support** — Upload and query multiple documents  
- 🧩 **Chunking & Embedding** — Splits content for better context retrieval  
- 🔍 **RAG Pipeline** — Retrieval + Context-aware AI answers  
- 🧠 **Powered by OpenAI** — GPT-based conversational interface  
- 📊 **Source Transparency** — Displays top 3 document sources  
- ⚡ **Streamlit UI** — Simple and interactive interface  

---

## 🛠️ Tech Stack

| Layer | Technologies | Purpose |
|-------|--------------|---------|
| **Frontend** | Streamlit | Interactive UI |
| **Backend** | Python, LangChain | RAG pipeline & orchestration |
| **Vector DB** | ChromaDB | Store & retrieve embeddings |
| **Document Loader** | PyMuPDF | Parse PDF files |
| **LLM + Embeddings** | OpenAI (GPT + embeddings) | Contextual QA |

---

## ⚙️ How It Works (RAG Pipeline)

1. 📥 **Upload PDFs** — User uploads documents via Streamlit UI  
2. ✂️ **Text Splitting** — Documents are chunked into smaller passages  
3. 🔑 **Embedding** — Each chunk is embedded using OpenAI embeddings  
4. 💾 **Vector Store** — Chunks + embeddings stored in ChromaDB  
5. ❓ **Query** — User asks a question  
6. 🔍 **Retriever** — Relevant chunks are retrieved  
7. 🤖 **LLM Response** — GPT answers using retrieved context  
8. 📑 **Sources** — Top 3 supporting chunks shown  

---

## 🧪 Local Development

### 🔧 Requirements

- Python **3.9+**  
- OpenAI API Key  

---

## 🏁 Getting Started

### 1. Clone & Setup

```bash
git clone https://github.com/your-username/rag-file-chatbot.git
cd rag-file-chatbot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add API Key

Create a `.env` file (see `.env.example`):

```bash
OPENAI_API_KEY=your_openai_key
```

---

## 🚦 Run the App

```bash
streamlit run app.py
```

The app will run locally at 👉 **http://localhost:8501**

---

## 📁 Folder Structure

```
rag-file-chatbot/
├── app.py              # Main Streamlit app
├── requirements.txt    # Python dependencies
├── .env.example        # Example API keys
├── .gitignore
└── README.md
```

---

## 🙌 Acknowledgments

- [LangChain](https://www.langchain.com/)  
- [Streamlit](https://streamlit.io/)  
- [ChromaDB](https://www.trychroma.com/)  
- [OpenAI](https://openai.com/)  
- [PyMuPDF](https://pymupdf.readthedocs.io/)  

---

<div align="center">
  Built with ❤️ by <a href="https://github.com/kartik0905">Kartik Garg</a>
</div>
