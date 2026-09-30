from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --------------------------
# Load the loan guide
# --------------------------

loader = TextLoader("documents/loan_guide.md")

documents = loader.load()

# --------------------------
# Split into smaller chunks
# --------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

print("Number of Chunks:", len(docs))

# --------------------------
# Create Embeddings
# --------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --------------------------
# Create Vector Database
# --------------------------

vector_db = FAISS.from_documents(
    docs,
    embeddings
)

# --------------------------
# Save Vector Database
# --------------------------

vector_db.save_local("vectorstore")

print("Vector Database Created Successfully!")