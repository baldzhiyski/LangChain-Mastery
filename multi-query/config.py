"""
Central config & defaults for Multi-Query RAG.
Flip USE_OPENAI_EMBEDDINGS to run fully free (HuggingFace) vs OpenAI (cheap).
"""

import os
from dotenv import load_dotenv

load_dotenv()  # load .env once here

# --- Embedding / LLM switches ---
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"  # cheapest OpenAI embeddings
OPENAI_CHAT_MODEL = "gpt-4.1-mini"    # deterministic LLM (temperature set in code)

# --- Chunking defaults ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Retrieval defaults ---
TOP_K_PER_QUERY = 4         # how many chunks per sub-query
NUM_QUERY_VARIANTS = 4      # how many paraphrased sub-queries to generate
USE_MMR = False             # can be overridden via CLI
MMR_DIVERSITY = 0.3         # 0..1 (higher => more diversity)

# --- Persistence ---
DEFAULT_PERSIST_DIR = "./stores/chroma_db"

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # required only if using OpenAI
