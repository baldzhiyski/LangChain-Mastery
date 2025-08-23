"""
Small helpers to inspect what was retrieved (chunks, sources).
"""

from typing import List
from langchain_core.documents import Document


def print_retrieved(docs: List[Document], title: str = "Retrieved Chunks"):
    print(f"\n=== {title} (n={len(docs)}) ===")
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        source = meta.get("source") or meta.get("url") or "unknown"
        print(f"\n[{i}] SOURCE: {source}")
        print(d.page_content[:500] + ("..." if len(d.page_content) > 500 else ""))
