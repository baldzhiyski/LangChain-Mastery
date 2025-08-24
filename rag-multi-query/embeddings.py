"""
Embedding factory: OpenAI (cheap) vs. free local HF embeddings.
No LangSmith here.
"""

from typing import Optional
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import OPENAI_EMBEDDING_MODEL, OPENAI_API_KEY


def make_embeddings() -> object:
    """
    Return an embeddings instance compatible with LangChain vectorstores.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing, but USE_OPENAI_EMBEDDINGS=True.")
    return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)