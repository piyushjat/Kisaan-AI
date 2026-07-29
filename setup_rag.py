"""
Run once before starting app.py:
    python setup_rag.py
"""

import os
from dotenv import load_dotenv
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

FAISS_INDEX_DIR = "./faiss_index"

# 1. Download Dataset from Hugging Face Hub
print("📥 Downloading Agriculture Dataset from Hugging Face...")
hf_dataset = load_dataset("KisanVaani/agriculture-qa-english-only", split="train")

# 2. Convert rows into LangChain Documents
print("🔄 Converting dataset into LangChain format...")
documents = []

# Process the first 2,000 entries for a fast initial setup.
# Remove .select() to load the full dataset (may take longer).
for row in hf_dataset.select(range(2000)):
    text_content = (
        f"Crop Issue/Question: {row['question']}\n"
        f"Solution/Answer: {row['answers']}"
    )
    doc = Document(
        page_content=text_content,
        metadata={"source": "HuggingFace: KisanVaani/agriculture-qa-english-only"},
    )
    documents.append(doc)

print(f"   → {len(documents)} documents loaded.")

# 3. Split Text into Chunks
print("✂️  Splitting text into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
chunks = text_splitter.split_documents(documents)
print(f"   → {len(chunks)} chunks created.")

# 4. Generate Embeddings using a local HuggingFace model (no API key required)
#    'all-MiniLM-L6-v2' is fast, small (~80 MB), and good for semantic search.
print("🧠 Loading local embedding model (all-MiniLM-L6-v2)...")
print("   (First run will download ~80 MB — subsequent runs use local cache)")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 5. Store embeddings in local ChromaDB
print("💾 Storing embeddings in FAISS in disk")
vector_db = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings,
)
vector_db.save_local(FAISS_INDEX_DIR)
print()
print("✅ Success! RAG database is ready.")
print("   Vector store saved to: ./chroma_db")
print("   You can now run:  streamlit run app.py")