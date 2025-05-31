from pathlib import Path
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from config import EMBEDDINGS

CHUNK_SIZE = 10_000
OVERLAP = 1_000

def pdf_to_text(pdf_file) -> str:
    reader = PdfReader(pdf_file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def split_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP
    )
    return splitter.split_text(text)

def build_faiss_index(chunks: List[str], index_path: str = "faiss_index") -> None:
    vector_store = FAISS.from_texts(chunks, embedding=EMBEDDINGS)
    vector_store.save_local(index_path)

def load_faiss_index(index_path: str = "faiss_index"):
    if not Path(index_path).exists():
        raise FileNotFoundError("FAISS index not found; process resume first.")
    return FAISS.load_local(
        index_path, EMBEDDINGS, allow_dangerous_deserialization=True
    )
