# Pro-RAG-Document-Intelligence

# Pro-RAG Document Intelligence Assistant 🤖📄

An advanced Retrieval-Augmented Generation (RAG) application that allows users to chat with multiple PDF documents in real-time. This project leverages Deep Learning and Large Language Models (LLMs) to provide context-aware answers with source citations.

1. ## 🚀 Features
  * **Multi-PDF Support:** Upload and process multiple PDF files simultaneously.
  * **Real-time Streaming:** Smooth, token-by-token response generation using Groq's high-speed inference.
  * **Source Transparency:** Automated citations that show exactly which part of the PDF was used to generate an answer.
  * **Vector Search:** Uses ChromaDB and HuggingFace Embeddings for semantic document retrieval.

2. ## 🛠️ Tech Stack
  * **Frontend:** Streamlit
  * **LLM:** Llama 3 (via Groq Cloud)
  * **Orchestration:** LangChain (LCEL)
  * **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
  * **Vector Store:** ChromaDB

3. ## 📦 Installation & Setup

   i) Clone the repository:
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)

  ii) Install dependencies:

     Bash
    pip install -r requirements.txt

  iii) Set up your .env file with your GROQ_API_KEY.

  iv) Run the app:

     Bash
    streamlit run app.py

4. ## Live Link
   https://pro-rag-document-intelligence-vxlhsubmuun7cnx8zra3ux.streamlit.app/
