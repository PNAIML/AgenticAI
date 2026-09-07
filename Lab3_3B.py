import tempfile

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Chat with Your PDF",
    page_icon="📘",
    layout="wide"
)

st.title("📘 Chat with Your PDF")

st.write(
    "Upload a PDF (maximum 2 pages) and ask questions about it."
)

# --------------------------------------------------
# Session State
# --------------------------------------------------

if "db" not in st.session_state:
    st.session_state.db = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("Upload PDF")

    uploaded_file = st.file_uploader(
        "Drag and drop your PDF here",
        type="pdf"
    )

# --------------------------------------------------
# Process PDF
# --------------------------------------------------

if uploaded_file is not None and st.session_state.db is None:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_pdf:

        temp_pdf.write(uploaded_file.read())
        pdf_path = temp_pdf.name

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    if len(documents) > 2:
        st.error(
            "Only PDF files having a maximum of 2 pages are allowed."
        )
        st.stop()

    with st.spinner("Creating Knowledge Base..."):

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = splitter.split_documents(documents)

        embeddings = OllamaEmbeddings(
            model="nomic-embed-text"
        )

        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings
        )

        st.session_state.db = db

    st.success("Knowledge Base Ready!")

# --------------------------------------------------
# Chat History
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# --------------------------------------------------
# Chat
# --------------------------------------------------

if st.session_state.db is not None:

    question = st.chat_input(
        "Ask a question about your PDF..."
    )

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):

            st.markdown(question)

        with st.spinner("Searching PDF..."):

            docs = st.session_state.db.similarity_search(
                question,
                k=3
            )

            context = "\n\n".join(
                doc.page_content
                for doc in docs
            )

            llm = OllamaLLM(model="llama3")

            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template="""
Answer ONLY using the supplied context.

If the answer is unavailable, reply:

I could not find that information in the PDF.

Context:
{context}

Question:
{question}

Answer:
"""
            )

            chain = prompt | llm

            answer = chain.invoke(
                {
                    "context": context,
                    "question": question
                }
            )

        with st.chat_message("assistant"):

            st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

else:

    st.info("Upload a PDF to begin.")