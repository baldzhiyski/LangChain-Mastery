"""
Chroma helpers: build & load persisted vector stores.
"""

import bs4
from typing import List, Optional
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from config import CHUNK_SIZE, CHUNK_OVERLAP


def load_web_docs(urls: List[str]):
    """
    Load web pages and keep only relevant parts with SoupStrainer.
    """
    loader = WebBaseLoader(
        web_paths=urls,
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        )
    )
    return loader.load()


def split_docs(docs):
    """
    Split long docs into overlapping chunks to improve retrieval quality.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)


def build_and_persist_chroma(splits, embeddings, persist_directory: str):
    """
    Create a Chroma vector store from splits and persist it.
    """
    vs = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vs.persist()
    return vs


def load_persisted_chroma(embeddings, persist_directory: str):
    """
    Load an existing Chroma vector store (no re-embedding).
    """
    return Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
