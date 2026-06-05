import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import streamlit as st
import tempfile
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Page Configuration - Must be the first streamlit command called
st.set_page_config(
    page_title="DocuMind RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

# Custom CSS for UI enhancements
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #FF5A5F;
        font-weight: 700 !important;
    }
    .stAlert {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR: Document Management ---
with st.sidebar:
    st.title("📂 Document Control")
    st.write("Upload your reference material here to prime the AI model.")
    
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], label_visibility="collapsed")
    
    st.markdown("---")
    
    # State tracking & processing status indicator
    if uploaded_file:
        if "file_name" not in st.session_state or st.session_state.file_name != uploaded_file.name:
            st.session_state.file_name = uploaded_file.name
            if "rag_chain" in st.session_state:
                del st.session_state.rag_chain
                
        if "rag_chain" not in st.session_state:
            with st.spinner("⏳ Parsing PDF & Generating Embeddings..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_pdf_path = temp_file.name

                try:
                    # 1. Load the Document
                    loader = PyPDFLoader(temp_pdf_path)
                    docs = loader.load()

                    # 2. Split text into chunks
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = splitter.split_documents(docs)

                    # 3. Create embeddings and store in Vector DB
                    embeddings = MistralAIEmbeddings(model="mistral-embed")
                    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)

                    # 4. Create Retriever
                    retriever = vector_store.as_retriever(
                        search_type="mmr",
                        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
                    )

                    # 5. Initialize LLM
                    llm = ChatMistralAI(model="mistral-small-latest")

                    # 6. Create Prompt
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", 'You are a helpful AI assistant.\n\nUse ONLY the provided context to answer the question.\n\nIf the answer is not present in the context,\nsay: "I could not find the answer in the document."\n'),
                        ("human", "Context:\n{context}\n\nQuestion:\n{input}\n")
                    ])

                    # 7. Create chains
                    question_answer_chain = create_stuff_documents_chain(llm, prompt)
                    st.session_state.rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                    st.success("✅ Context Engine Ready!")

                except Exception as e:
                    st.error(f"❌ Processing Error: {e}")
                finally:
                    if os.path.exists(temp_pdf_path):
                        os.remove(temp_pdf_path)
        else:
            st.success("✅ Context Engine Ready!")
    else:
        st.info("💡 Please upload a PDF file to begin.")

# --- MAIN INTERFACE: Q&A Engine ---
st.title("🤖 DocuMind Q&A Dashboard")
st.write("Interact natively with your document context using conversational questions.")

# Split layout into a neat clean workspace
if "rag_chain" in st.session_state:
    
    # Layout with columns to center things nicely
    col1, col2 = st.columns([5, 1])
    
    with col1:
        query = st.text_input("Ask a question about the document:", placeholder="e.g., What is the executive summary stating?", label_visibility="collapsed")
    with col2:
        submit_btn = st.button("Ask AI", use_container_width=True)

    if submit_btn or (query and st.session_state.get('prev_query') != query):
        if not query:
            st.warning("⚠️ Please input a question first.")
        else:
            # Visual feedback utilizing containers
            with st.chat_message("user"):
                st.write(query)
                
            with st.chat_message("assistant"):
                with st.spinner("Analyzing document layers..."):
                    try:
                        response = st.session_state.rag_chain.invoke({"input": query})
                        st.markdown("### Answer")
                        st.write(response["answer"])
                    except Exception as e:
                        st.error(f"An error occurred while retrieving the answer: {e}")
else:
    # Large, clear empty state placeholder
    st.container()
    st.markdown("""
        <div style="background-color: #f0f2f6; padding: 3rem; border-radius: 15px; text-align: center; margin-top: 2rem;">
            <h3 style="color: #31333F;">Waiting for Data Pipeline...</h3>
            <p style="color: #555;">Please upload a PDF document in the left sidebar menu to activate the RAG engine interface.</p>
        </div>
    """, unsafe_allow_html=True)