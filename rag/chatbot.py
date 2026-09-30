import os

from groq import Groq
import streamlit as st

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# ----------------------------
# Groq Client
# ----------------------------

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)


# ----------------------------
# Create / Load Vector Database
# ----------------------------

@st.cache_resource
def get_vector_db():

    # Load embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore_path = "vectorstore"

    # ----------------------------
    # If vectorstore already exists
    # ----------------------------

    if os.path.exists(vectorstore_path):

        vector_db = FAISS.load_local(
            vectorstore_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        return vector_db

    # ----------------------------
    # Create vectorstore
    # ----------------------------

    print("Vectorstore not found. Creating it...")

    loader = TextLoader(
        "documents/loan_guide.md"
    )

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = text_splitter.split_documents(documents)

    vector_db = FAISS.from_documents(
        docs,
        embeddings
    )

    vector_db.save_local(
        vectorstore_path
    )

    print("Vectorstore created successfully!")

    return vector_db


# ----------------------------
# Get Vector Database
# ----------------------------

vector_db = get_vector_db()


# ----------------------------
# Chat Function
# ----------------------------

def ask_loan_assistant(question):

    docs = vector_db.similarity_search(
        question,
        k=3
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are a helpful AI loan assistant.

Answer ONLY using the information below.

Context:

{context}

Question:

{question}

If the answer is not available in the context,
reply:

"I couldn't find that information in the loan guide."
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content