import os
from dotenv import load_dotenv
import spacy
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from google.generativeai import configure as google_config
import chromadb
from chromadb.utils import embedding_functions

# ── env & keys ────────────────────────────────────────────────
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("❌ Missing GOOGLE_API_KEY in .env")

google_config(api_key=GOOGLE_API_KEY)

# ── LLM / Embeddings ──────────────────────────────────────────
LLM = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
EMBEDDINGS = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# ── spaCy ─────────────────────────────────────────────────────
try:
    NLP = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError("Run:  python -m spacy download en_core_web_sm")

# ── ChromaDB (feedback store) ─────────────────────────────────
CHROMA_CLIENT = chromadb.PersistentClient(path="./feedback_db")
EMBED_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
FEEDBACK_COL = CHROMA_CLIENT.get_or_create_collection(
    name="career_feedback", embedding_function=EMBED_FN
)
