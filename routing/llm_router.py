# llm_router_rag.py
"""
LLM-Routed RAG (single file, HTML indexing, no LangSmith/Hub)

What this script does
=====================
1) Index mode:
   - Fetch HTML page(s)
   - Split into overlapping chunks
   - Embed chunks
   - Persist a Chroma vector DB into ./stores/<collection>

2) Ask mode:
   - Auto-discover collections under ./stores
   - Use a tiny LLM "router" prompt to choose the most relevant collection
   - Run Multi-Query Retrieval (generate paraphrases, search, merge & dedupe)
   - Answer with a strict RAG prompt (ONLY from retrieved context)

Why an LLM Router?
==================
You don’t pass a collection when asking. The router LLM picks the best one
(e.g., "python_docs" vs "js_docs") based on your question.

Requirements
============
pip install -U:
  langchain langchain-core langchain-community langchain-openai
  langchain-chroma chromadb openai python-dotenv beautifulsoup4 lxml
  sentence-transformers tiktoken

Environment (.env)
==================
OPENAI_API_KEY=sk-...
# (optional) silence polite UA warning
USER_AGENT=LLMRouterRAG/1.0 (+https://example.com)

Usage
=====
# Index two different sources into separate collections
python llm_router_rag.py index --name python_docs --url https://docs.python.org/3/tutorial/index.html
python llm_router_rag.py index --name js_docs    --url https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide

# Ask — the router chooses the collection automatically
python llm_router_rag.py ask -q "How do I slice a list in Python?"
python llm_router_rag.py ask -q "How do arrow functions work?"

Tip: Index actual topic pages (or crawl a section). Index pages alone (TOCs)
often lack the details needed to answer your question.
"""

import os
import argparse
from typing import List, Set

from dotenv import load_dotenv
import bs4

# Prompts & LLMs
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Free local fallback embeddings (if you set USE_OPENAI_EMBEDDINGS=False)
from langchain_community.embeddings import HuggingFaceEmbeddings

# HTML loader & splitting
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Modern Chroma integration (avoid deprecation warnings)
from langchain_community.vectorstores import Chroma



# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")

# Choose embeddings provider:
# - True  -> OpenAI "text-embedding-3-small" (cheap, good)
# - False -> HuggingFace MiniLM (free, local)
USE_OPENAI_EMBEDDINGS = True
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

BASE_STORES_DIR = "./stores"   # each collection is a subfolder here
CHUNK_SIZE = 1000              # chunking improves retrieval & context
CHUNK_OVERLAP = 200

TOP_K_PER_QUERY = 4            # how many chunks we keep per query variant
NUM_QUERY_VARIANTS = 4         # how many paraphrases we generate


# ──────────────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────────────
# Final answering prompt — strict: answer ONLY from retrieved context
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a helpful assistant. Answer ONLY using the provided context. "
         "If the answer is not in the context, say you don't know."),
        ("human",
         "Context:\n{context}\n\nQuestion: {question}")
    ]
)

# Multi-Query prompt — generate diverse paraphrases of the user question
MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You generate diverse, non-redundant search queries that capture different phrasings."),
        ("human",
         "Question: {question}\n\nGenerate {n} alternative queries (one per line, no numbering).")
    ]
)

# Router prompt — pick ONE collection name from the provided list
ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You route user questions to the most relevant datasource name from the provided list. "
         "Return ONLY the datasource string, nothing else."),
        ("human",
         "Question: {question}\n\nDatasources: {datasources}\n\n"
         "Pick ONE datasource name that best fits the question:")
    ]
)


# ──────────────────────────────────────────────────────────────────────────────
# IO helpers
# ──────────────────────────────────────────────────────────────────────────────
def persist_dir(name: str) -> str:
    """Resolve a collection folder path under ./stores."""
    return os.path.join(BASE_STORES_DIR, name)


def make_embeddings():
    """
    Build the embedding function.

    - OpenAI: high quality, paid
    - HF MiniLM: free local fallback (smaller quality, OK for demos)
    """
    if USE_OPENAI_EMBEDDINGS:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY missing but USE_OPENAI_EMBEDDINGS=True.")
        return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)

    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def load_web_docs(urls: List[str]):
    """
    Fetch HTML for each URL and extract BODY text.
    Note: WebBaseLoader grabs server-rendered HTML (not JS-only content).
    For JS-heavy sites, consider AsyncHtmlLoader or a headless browser later.
    """
    loader = WebBaseLoader(
        web_paths=urls,
        bs_kwargs=dict(parse_only=bs4.SoupStrainer("body")),
    )
    docs = loader.load()

    # Light cleanup: collapse whitespace so chunks are cleaner
    for d in docs:
        d.page_content = " ".join((d.page_content or "").split())

    return docs


def split_docs(docs):
    """Split documents into overlapping chunks for better retrieval & context."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)


def build_and_persist_chroma(splits, embeddings, dirpath: str):
    """
    Create a new Chroma collection for this set of splits and persist to disk.
    Re-running 'index' for the same dir will add more docs to that collection.
    """
    vs = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=dirpath
    )
    vs.persist()  # write to disk
    return vs


def load_persisted_chroma(embeddings, dirpath: str):
    """Open an existing Chroma collection from disk (read/write)."""
    return Chroma(
        embedding_function=embeddings,
        persist_directory=dirpath
    )


def print_retrieved(docs, title="Retrieved", preview=400):
    """Pretty-print retrieved chunk previews + source URL (best-effort)."""
    print(f"\n=== {title} (n={len(docs)}) ===")
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source") or d.metadata.get("url") or "unknown"
        txt = d.page_content or ""
        print(f"\n[{i}] SOURCE: {src}\n{txt[:preview]}{'...' if len(txt)>preview else ''}")


# ──────────────────────────────────────────────────────────────────────────────
# Multi-Query Retrieval (MQR)
# ──────────────────────────────────────────────────────────────────────────────
def generate_query_variants(llm: ChatOpenAI, question: str, n: int) -> List[str]:
    """
    Ask the LLM for 'n' diverse phrasings of the question.
    We then dedupe lines (case-insensitive) and ensure the original question
    is included (at position 0 if not already).
    """
    msgs = MULTI_QUERY_PROMPT.format_messages(question=question, n=n)
    out = llm.invoke(msgs).content or ""
    raw_lines = [ln.strip() for ln in out.split("\n") if ln.strip()]

    seen: Set[str] = set()
    variants: List[str] = []
    for v in raw_lines:
        key = v.lower()
        if key not in seen:
            seen.add(key)
            variants.append(v)

    if question.lower() not in seen:
        variants.insert(0, question)

    return variants


def retrieve_multi(vs: Chroma, question: str, k: int) -> List:
    """
    Classic multi-query retrieval:
      - Create several paraphrases (variants) of the question
      - Similarity search for each variant
      - Merge & de-duplicate results (by source + content hash)
    """
    llm = ChatOpenAI(model_name=OPENAI_CHAT_MODEL, temperature=0)
    variants = generate_query_variants(llm, question, NUM_QUERY_VARIANTS)

    merged, seen = [], set()
    for q in variants:
        docs = vs.similarity_search(q, k=k)
        for d in docs:
            sig = (d.metadata.get("source", ""), hash(d.page_content))
            if sig not in seen:
                seen.add(sig)
                merged.append(d)
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# Router
# ──────────────────────────────────────────────────────────────────────────────
def route_to_collection(question: str, known: List[str]) -> str:
    """
    Use a tiny LLM prompt to pick ONE collection from 'known'.
    If OPENAI_API_KEY is missing, fall back to a keyword heuristic.

    NOTE: For a more robust router you can:
      - Return top-K collections and fuse results across them (RRF)
      - Add semantic routing via collection profiles (embed & cosine sim)
    """
    if not known:
        raise RuntimeError("No collections found under ./stores")

    # Cheap fallback if no LLM available
    if not OPENAI_API_KEY:
        q = question.lower()
        if "python" in q:
            return "python_docs" if "python_docs" in known else known[0]
        if "javascript" in q or "js " in q:
            return "js_docs" if "js_docs" in known else known[0]
        return known[0]

    llm = ChatOpenAI(model_name=OPENAI_CHAT_MODEL, temperature=0)
    msgs = ROUTER_PROMPT.format_messages(
        question=question,
        datasources=", ".join(known)
    )
    raw = llm.invoke(msgs).content or ""
    choice = raw.strip()
    # Sanitize: ensure the router's choice actually exists
    return choice if choice in known else known[0]


# ──────────────────────────────────────────────────────────────────────────────
# Commands
# ──────────────────────────────────────────────────────────────────────────────
def cmd_index(name: str, urls: List[str]) -> None:
    """
    Index HTML pages into a Chroma collection named 'name'.
    Safe to call multiple times to add more pages into the same collection.
    """
    dirpath = persist_dir(name)
    os.makedirs(dirpath, exist_ok=True)

    docs = load_web_docs(urls)
    splits = split_docs(docs)
    embeddings = make_embeddings()

    build_and_persist_chroma(splits, embeddings, dirpath)
    print(f"✅ Indexed into: {dirpath}")
    print("Tip: Prefer indexing topic pages (or crawl a section) over TOC/landing pages.")


def cmd_ask(question: str) -> None:
    """
    Ask a question:
      - Discover available collections under ./stores
      - Route to the best collection name
      - Run multi-query retrieval
      - Answer with a strict RAG prompt
    """
    if not os.path.isdir(BASE_STORES_DIR):
        raise SystemExit("No collections found. Run 'index' first.")

    known = [d for d in os.listdir(BASE_STORES_DIR)
             if os.path.isdir(persist_dir(d))]
    if not known:
        raise SystemExit("No collections found under ./stores. Run 'index' first.")

    chosen = route_to_collection(question, known)
    print(f"\n🔀 Router chose collection: {chosen}")

    embeddings = make_embeddings()
    vs = load_persisted_chroma(embeddings, persist_dir(chosen))

    # Retrieve
    retrieved = retrieve_multi(vs, question, TOP_K_PER_QUERY)
    print_retrieved(retrieved, title=f"Retrieved from '{chosen}'")

    # If nothing retrieved, be explicit (prevents empty context hallucinations)
    if not retrieved:
        print("\n=== Answer ===\n I don't know (no relevant context was retrieved).")
        return

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY required to generate the final answer.")

    # Answer
    llm = ChatOpenAI(model_name=OPENAI_CHAT_MODEL, temperature=0)
    chain = (RAG_PROMPT | llm | StrOutputParser())
    ctx = "\n\n".join(d.page_content for d in retrieved)
    ans = chain.invoke({"context": ctx, "question": question})
    print("\n=== Answer ===\n", ans)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LLM-Routed RAG")
    sub = parser.add_subparsers(dest="cmd")

    p_index = sub.add_parser("index", help="Index: HTML -> split -> embed -> persist")
    p_index.add_argument("--name", required=True, help="Collection name (folder under ./stores)")
    p_index.add_argument("--url", action="append", required=True, help="Repeatable URL to index")

    p_ask = sub.add_parser("ask", help="Ask a question; router selects the collection")
    p_ask.add_argument("-q", "--question", required=True, help="User question")

    args = parser.parse_args()

    if args.cmd == "index":
        return cmd_index(args.name, args.url)
    if args.cmd == "ask":
        return cmd_ask(args.question)

    parser.print_help()


if __name__ == "__main__":
    main()
