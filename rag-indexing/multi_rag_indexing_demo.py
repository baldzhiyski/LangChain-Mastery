"""
Multi-representation Indexing — minimal demo

Goal
----
Index *two representations* of each document:
1) A SHORT, dense **summary** (good for matching queries)
2) The FULL original document (what you actually want to return)

We index only the **summaries** into the vector store (for fast/accurate retrieval),
and we keep the **full docs** in a side store. At query time:
- Search over the summaries (child reps) to find candidates
- Then fetch the corresponding full parent docs for the final answer

Why this helps
--------------
Queries often match a concise summary better than a long raw page. This increases recall
and reduces noise, while still letting you show the full original content.

Requirements
------------
pip install -U langchain langchain-core langchain-community langchain-openai \
               langchain-chroma chromadb python-dotenv beautifulsoup4 lxml

.env
----
OPENAI_API_KEY=sk-...
(optional) USER_AGENT=MultiRepDemo/1.0 (+https://example.com)

Run
---
python multi_rep_indexing_demo.py
"""

import os
import uuid
from typing import List

from dotenv import load_dotenv

# Load HTML pages
from langchain_community.document_loaders import WebBaseLoader

# LLM + prompts
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Vector store + multi-vector retriever
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever

# Store for parent documents (we’ll keep it simple: in-memory)
from langchain.storage import InMemoryByteStore

# Basic LangChain document class
from langchain_core.documents import Document


# ----------------------------
# Config (kept very minimal)
# ----------------------------
load_dotenv()
os.environ.setdefault("USER_AGENT", "MultiRepDemo/1.0 (+https://example.com)")

OPENAI_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
EMBED_MODEL = "text-embedding-3-small"

# Persist Chroma so you can reuse without re-indexing (optional, but handy)
PERSIST_DIR = "./stores/multi_rep_demo"
COLLECTION_NAME = "summaries"
ID_KEY = "doc_id"  # the metadata field that links child->parent


def load_docs() -> List[Document]:
    """Download two blog posts (as examples)."""
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-02-05-human-data-quality/",
    ]
    docs: List[Document] = []
    for u in urls:
        loader = WebBaseLoader(u)  # extracts server-rendered HTML text
        docs.extend(loader.load())
    # Light cleanup (optional): trim whitespace
    for d in docs:
        d.page_content = " ".join((d.page_content or "").split())
    return docs


def build_summarizer_chain():
    """A tiny chain: {doc} -> 'Summarize...' -> LLM -> string."""
    prompt = ChatPromptTemplate.from_template(
        "Summarize the following document in 4–6 sentences, focusing on key ideas:\n\n{doc}"
    )
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    chain = (
        {"doc": lambda x: x.page_content}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def make_vectorstore():
    """Create (or reopen) a Chroma collection for child reps (summaries)."""
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vs = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    return vs


def make_retriever(vectorstore: Chroma) -> MultiVectorRetriever:
    """
    MultiVectorRetriever glues together:
      - vectorstore for CHILD reps (summaries)
      - a doc store for PARENT docs (full originals)
      - an id_key to link child->parent
    """
    parent_store = InMemoryByteStore()  # simple in-memory store
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=parent_store,
        id_key=ID_KEY,
    )
    return retriever


def index(docs: List[Document]) -> MultiVectorRetriever:
    """
    1) Summarize each full doc (child rep)
    2) Add summaries to the vector store (with parent id in metadata)
    3) Put full parent docs into the byte store (id -> Document)
    """
    print("🔧 Summarizing documents...")
    summarize = build_summarizer_chain()
    summaries: List[str] = summarize.batch(docs, {"max_concurrency": 4})

    # Create unique ids to link child summaries back to parent docs
    doc_ids = [str(uuid.uuid4()) for _ in docs]

    # Build vector store + retriever
    vectorstore = make_vectorstore()
    retriever = make_retriever(vectorstore)

    # Create child docs (summaries) that point to their parent id
    summary_docs = []
    for i, s in enumerate(summaries):
        # Carry forward the original source if present
        src = docs[i].metadata.get("source", "")
        summary_docs.append(
            Document(
                page_content=s,
                metadata={ID_KEY: doc_ids[i], "source": src, "rep": "summary"},
            )
        )

    print("💾 Adding summaries to vector store...")
    retriever.vectorstore.add_documents(summary_docs)
    retriever.vectorstore.persist()  # write to disk (optional but useful)

    print("📦 Storing full parent docs (in-memory)...")
    # Map id -> raw bytes of the Document (handled by ByteStore)
    retriever.docstore.mset(list(zip(doc_ids, docs)))

    print("✅ Indexing complete.")
    return retriever


def demo_query(retriever: MultiVectorRetriever, query: str):
    """
    Run a query:
      - The retriever searches the SUMMARY vectors
      - Then it returns the FULL original parent documents
    """
    print(f"\n🔎 Query: {query}\n")
    # Get the full parent docs via the retriever
    parent_docs = retriever.get_relevant_documents(query)

    if not parent_docs:
        print("No results.")
        return

    # Show top result (parent), plus (optionally) the top matching summary
    top = parent_docs[0]
    print("=== TOP PARENT DOC ===")
    print("Source:", top.metadata.get("source", "unknown"))
    print(top.page_content[:600], "..." if len(top.page_content) > 600 else "")

    # (Optional) Peek at which summary matched (child side)
    child_hits = retriever.vectorstore.similarity_search(query, k=1)
    if child_hits:
        ch = child_hits[0]
        print("\n--- Matching SUMMARY (child rep) ---")
        print("Child source:", ch.metadata.get("source", "unknown"))
        print(ch.page_content[:350], "..." if len(ch.page_content) > 350 else "")


def main():
    # 1) Load docs
    docs = load_docs()

    # 2) Index with multi-representation (summary child -> parent full doc)
    retriever = index(docs)

    # 3) Ask something that should match a concept in the posts
    demo_query(retriever, "Memory in agents")    # matches agent post
    demo_query(retriever, "data quality for human feedback datasets")  # matches data-quality post


if __name__ == "__main__":
    main()
