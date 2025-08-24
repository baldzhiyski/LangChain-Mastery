# semantic_router_rag.py
"""
Semantic-Routed RAG (single file, HTML indexing, no LangSmith/Hub)

What this script does
=====================
1) Index mode:
   - Fetch HTML page(s)
   - Split into overlapping chunks
   - Embed chunks
   - Persist a Chroma vector DB into ./stores/<collection>

2) Ask mode:
   - Auto-discover collections under ./stores
   - Route to the best collection via **semantic similarity** between:
        - the user's question embedding, and
        - a short text **profile** for each collection (stored in profile.txt)
   - Run Multi-Query Retrieval (generate paraphrases, search, merge & dedupe)
   - Answer with a strict RAG prompt (ONLY from retrieved context)

Why semantic routing?
=====================
Instead of asking an LLM to pick a collection, we do a **cheap, deterministic**
embedding similarity between the question and each collection's **profile**.
It’s fast, transparent, and doesn’t require LLM tokens.

Requirements
============
pip install -U:
  langchain langchain-core langchain-community langchain-openai
  langchain-chroma chromadb openai python-dotenv beautifulsoup4 lxml
  sentence-transformers tiktoken

Environment (.env)
==================
# Required if using OpenAI embeddings/LLM
OPENAI_API_KEY=sk-...

# Optional: silence polite UA warning
USER_AGENT=SemanticRouterRAG/1.0 (+https://example.com)

Usage
=====
# Index collections (you can run multiple times to add more URLs)
python semantic_router_rag.py index --name python_docs --url https://docs.python.org/3/tutorial/index.html
python semantic_router_rag.py index --name js_docs     --url https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide

# (Optional) Edit ./stores/<name>/profile.txt to make routing smarter
#   e.g., python_docs/profile.txt:
#     "Official Python docs: tutorial, lists, slicing, functions, classes, stdlib"
#         js_docs/profile.txt:
#     "MDN JavaScript guide: syntax, arrow functions, promises, prototypes"

# Ask — the semantic router chooses the collection automatically
python semantic_router_rag.py ask -q "How do I slice a list in Python?"
python semantic_router_rag.py ask -q "How do arrow functions work?"

Tip: Index actual topic pages (or crawl a section). Index pages alone (TOCs)
often lack the details needed to answer your question.
"""

import os
import argparse
from typing import List, Set

from dotenv import load_dotenv
import bs4
import numpy as np

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
os.environ.setdefault("USER_AGENT", "SemanticRouterRAG/1.0 (+https://example.com)")

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
# Semantic router
# ──────────────────────────────────────────────────────────────────────────────
def read_or_create_profile(name: str) -> str:
    """
    Ensure a short routing profile exists for a collection.
    The profile is a 1–2 sentence description stored at ./stores/<name>/profile.txt
    You can edit it manually to improve routing quality.
    """
    path = os.path.join(persist_dir(name), "profile.txt")
    if os.path.exists(path):
        return open(path, "r", encoding="utf-8").read().strip()

    default = f"{name}: general documentation and guides."
    with open(path, "w", encoding="utf-8") as f:
        f.write(default)
    return default


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between 1-D vectors a and b.
    (No external deps; avoids sklearn.)
    """
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def route_semantic(question: str, collections: List[str]) -> str:
    """
    Embed the question and each collection's profile; pick the collection
    with the highest cosine similarity.
    """
    # Build embeddings (OpenAI or HF)
    embs = make_embeddings()

    # Embed helper: returns list[list[float]]
    def embed_texts(texts: List[str]) -> List[List[float]]:
        # Both OpenAIEmbeddings and HuggingFaceEmbeddings expose embed_documents / embed_query
        return embs.embed_documents(texts)

    # Prepare profiles
    profiles = [read_or_create_profile(c) for c in collections]
    prof_vecs = np.array(embed_texts(profiles), dtype=np.float32)

    # Query embedding
    q_vec = np.array(embs.embed_query(question), dtype=np.float32)

    # Cosine similarity against each profile
    sims = np.array([_cosine_sim(v, q_vec) for v in prof_vecs], dtype=np.float32)

    # Report scores for transparency (optional)
    print("\n=== Semantic routing scores ===")
    for c, s in zip(collections, sims):
        print(f"{c:>20} : {s:.4f}")

    # Pick best
    idx = int(np.argmax(sims))
    return collections[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Commands
# ──────────────────────────────────────────────────────────────────────────────
def cmd_index(name: str, urls: List[str]) -> None:
    """
    Index HTML pages into a Chroma collection named 'name'.
    Safe to call multiple times to add more pages into the same collection.
    Also ensures a simple profile.txt exists for semantic routing.
    """
    dirpath = persist_dir(name)
    os.makedirs(dirpath, exist_ok=True)

    docs = load_web_docs(urls)
    splits = split_docs(docs)
    embeddings = make_embeddings()

    build_and_persist_chroma(splits, embeddings, dirpath)

    # Ensure a profile exists (you can edit it later)
    _ = read_or_create_profile(name)

    print(f"✅ Indexed into: {dirpath}")
    print("Tip: Prefer indexing topic pages (or crawl a section) over TOC/landing pages.")
    print(f"Tip: Edit the routing profile at: {os.path.join(dirpath, 'profile.txt')}")


def cmd_ask(question: str) -> None:
    """
    Ask a question:
      - Discover available collections under ./stores
      - Route via semantic similarity to the best collection
      - Run multi-query retrieval
      - Answer with a strict RAG prompt
    """
    if not os.path.isdir(BASE_STORES_DIR):
        raise SystemExit("No collections found. Run 'index' first.")

    known = [d for d in os.listdir(BASE_STORES_DIR)
             if os.path.isdir(persist_dir(d))]
    if not known:
        raise SystemExit("No collections found under ./stores. Run 'index' first.")

    chosen = route_semantic(question, known)
    print(f"\n🔀 Semantic router chose: {chosen}")

    embeddings = make_embeddings()
    vs = load_persisted_chroma(embeddings, persist_dir(chosen))

    # Retrieve
    retrieved = retrieve_multi(vs, question, TOP_K_PER_QUERY)
    print_retrieved(retrieved, title=f"Retrieved from '{chosen}'")

    # If nothing retrieved, be explicit (prevents empty context hallucinations)
    if not retrieved:
        print("\n=== Answer ===\n I don't know (no relevant context was retrieved).")
        return

    if not OPENAI_API_KEY and USE_OPENAI_EMBEDDINGS:
        raise RuntimeError("OPENAI_API_KEY required to generate the final answer with OpenAI Chat.")

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
    parser = argparse.ArgumentParser(description="Semantic-Routed RAG")
    sub = parser.add_subparsers(dest="cmd")

    p_index = sub.add_parser("index", help="Index: HTML -> split -> embed -> persist")
    p_index.add_argument("--name", required=True, help="Collection name (folder under ./stores)")
    p_index.add_argument("--url", action="append", required=True, help="Repeatable URL to index")

    p_ask = sub.add_parser("ask", help="Ask a question; semantic router selects the collection")
    p_ask.add_argument("-q", "--question", required=True, help="User question")

    args = parser.parse_args()

    if args.cmd == "index":
        return cmd_index(args.name, args.url)
    if args.cmd == "ask":
        return cmd_ask(args.question)

    parser.print_help()


if __name__ == "__main__":
    main()