from pathlib import Path
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from config import EMBEDDINGS

CHUNK_SIZE = 1000
OVERLAP = 200

# 1. Extract text from a PDF
def extract_text(file) -> str:
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text

# 2. Split into chunks for embedding
def split_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)

# 3. Create and save FAISS vector index
def build_faiss_index(chunks: List[str], index_path: str = "faiss_index") -> None:
    vector_store = FAISS.from_texts(chunks, embedding=EMBEDDINGS)
    vector_store.save_local(index_path)

# 4. Load vector index for retrieval
def load_faiss_index(index_path: str = "faiss_index"):
    if not Path(index_path).exists():
        raise FileNotFoundError("❌ FAISS index not found. Run analysis first.")
    return FAISS.load_local(index_path, EMBEDDINGS, allow_dangerous_deserialization=True)
