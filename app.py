# 1. NEW CONFIG: install core libraries
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# Paste your API KEY here
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# 2. FEATURE: Multi-File Processing
def process_pdfs(uploaded_files):
    all_docs = []

    # Chunking: 1000 characters per chunk, 200 overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    for uploaded_file in uploaded_files:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = uploaded_file.name

        chunks = text_splitter.split_documents(docs)
        all_docs.extend(chunks)

        if os.path.exists(temp_path):
            os.remove(temp_path)

    return all_docs


# 3. FEATURE: Persistent Chat History (Session State)

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device':'cpu'}
    )

def create_vector_db(chunks):
    embeddings = get_embeddings()
    
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vector_db

# --- USER INTERFACE ---
st.set_page_config(page_title="Enterprise RAG Assistant", layout="wide")
st.title("🤖 RAG-Powered Document and Q&A Assistant")
st.info("Upload your technical PDFs in the sidebar to begin.")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Upload Center")
    uploaded_files = st.file_uploader("Environment is ready Upload any PDF or Company Policies or Legal Docs", type="pdf", accept_multiple_files=True)
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
        
if uploaded_files:
    # Save the file temporarily
    if "vector_db" not in st.session_state:
        with st.spinner("Analyzing document..."):
            chunks = process_pdfs(uploaded_files)
            st.session_state.vector_db = create_vector_db(chunks)
            st.success(f"Indexed Completed! {len(uploaded_files)} files!")

            
    if "vector_db" in st.session_state:
        # Display Chat History
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
        # 4. FEATURE: Chat Input & Citations
        if prompt := st.chat_input("Ask a question about your documents: "):
            # Store and display user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                # step 1. Get the relevant chunks form your PDF first
                retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 3})
                relevant_docs = retriever.get_relevant_documents(prompt)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])

                # step 2. Create a prompt that includes the PDF context
                rag_prompt = f"Context:{context}\n\nQuestion:{prompt}\n\nAnswer using only the context:"

                # step 3. Stream the response
                response_placeholder = st.empty()
                full_response = ""
                
                with st.spinner("Thinking..."):
                    # Initialize Groq Llama-3
                    llm = ChatGroq(
                        temperature=0,
                        model_name="llama-3.3-70b-versatile",
                        api_key = GROQ_API_KEY
                    )

                    for chunk in llm.stream(rag_prompt):
                        if chunk and hasattr(chunk, 'content') and chunk.content is not None:
                            content = chunk.content
                            full_response += content
                            response_placeholder.markdown(full_response + "▌")
                        
                    response_placeholder.markdown(full_response)

                    st.session_state.messages.append({"role":"assistant", "content":full_response})
             
            # 5. Show the citations (The source we found in step 1)
            with st.expander("View Source Citations"):
                for doc in relevant_docs:
                    st.write(f"-{doc.metadata.get('source', 'Unknown PDF')}")
                    st.write(f"📄 **{doc.metadata.get('source')}** - Page {doc.metadata.get('page')}")
                            

