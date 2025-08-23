"""
Query structuring for metadata filters (teaching demo)

This demo turns a natural-language question (e.g., "short videos on chat langchain from 2023")
into a typed, structured query object you can pass to your search layer:
- content_search (semantic transcript query)
- title_search   (short title keywords)
- optional metadata filters (view_count, publish_date, length)

Why this pattern?
- Precision & speed (apply metadata filters with the vector search)
- Safety (only set filters if the user asked for them)
- Explainability (log/inspect the structured query)
- Backend-agnostic (works with Chroma/PGVector/ES/SQL/etc.)

Requirements:
  pip install -U langchain langchain-core langchain-community langchain-openai python-dotenv
  # optional (for YouTube metadata peek):
  pip install -U youtube-transcript-api pytube

.env:
  OPENAI_API_KEY=sk-...
"""

import os
import datetime
from typing import Optional, Dict, Any

from dotenv import load_dotenv

# LangChain core/LLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# ✅ Pydantic v2 imports
from pydantic import BaseModel, Field

# (Optional) YouTube metadata peek — helps you design realistic filters
try:
    from langchain_community.document_loaders import YoutubeLoader
    HAS_YT = True
except Exception:
    HAS_YT = False


# ──────────────────────────────────────────────────────────────────────────────
# 1) (Optional) Peek at YouTube metadata so you know what you can filter on
# ──────────────────────────────────────────────────────────────────────────────
def demo_youtube_metadata() -> None:
    """Fetch a sample YouTube doc (with add_video_info) and print common fields."""
    if not HAS_YT:
        print("ℹ️  Skipping YouTube metadata demo (install youtube-transcript-api + pytube to enable).")
        return
    try:
        docs = YoutubeLoader.from_youtube_url(
            "https://www.youtube.com/watch?v=pbAd8O1Lvm4",
            add_video_info=True,  # title / view_count / publish_date / length
        ).load()
        md = docs[0].metadata if docs else {}
        print("\n🔎 Sample YouTube metadata keys:", sorted(list(md.keys()))[:20], "...")
        print("   (example) title        :", md.get("title"))
        print("   (example) view_count   :", md.get("view_count"))     # int
        print("   (example) publish_date :", md.get("publish_date"))   # date/datetime
        print("   (example) length       :", md.get("length"))         # seconds
        print("   (note) transcript text lives in document.page_content")
    except Exception as e:
        # Network/HTTP issues are fine for the demo; just continue.
        print(f"⚠️  Could not load YouTube metadata: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# 2) Structured query schema (what your backend expects)
# ──────────────────────────────────────────────────────────────────────────────
class TutorialSearch(BaseModel):
    """A typed query you can pass into your retrieval layer."""

    # Text queries
    content_search: str = Field(
        ...,
        description="Similarity search query applied to video transcripts (rich natural language).",
    )
    title_search: str = Field(
        ...,
        description="Short keyword version for video titles; avoid fluff.",
    )

    # Optional numeric/date filters — ONLY set when user explicitly asked
    min_view_count: Optional[int] = Field(None, description="Minimum view count (inclusive).")
    max_view_count: Optional[int] = Field(None, description="Maximum view count (exclusive).")
    earliest_publish_date: Optional[datetime.date] = Field(None, description="Earliest date (inclusive).")
    latest_publish_date: Optional[datetime.date] = Field(None, description="Latest date (exclusive).")
    min_length_sec: Optional[int] = Field(None, description="Minimum length in seconds (inclusive).")
    max_length_sec: Optional[int] = Field(None, description="Maximum length in seconds (exclusive).")

    def pretty_print(self) -> None:
        """Human-friendly logging for debugging/UI chips (Pydantic v2-safe)."""
        print("\n🧩 Structured query:")
        # In v2, use model_fields; just print non-None values (simplest & robust)
        for name in self.model_fields:
            val = getattr(self, name)
            if val is not None:
                print(f"  - {name}: {val}")


# ──────────────────────────────────────────────────────────────────────────────
# 3) LLM that outputs the above schema (structured output / function calling)
# ──────────────────────────────────────────────────────────────────────────────
def build_query_analyzer():
    """
    Returns a Runnable that takes {"question": "..."} and outputs a TutorialSearch.
    Keep temp=0 for stable, deterministic structure.
    """
    system = (
        "You convert user questions into database queries.\n"
        "The database contains tutorial videos about a software library for building LLM apps.\n"
        "Given a question, return a query optimized to retrieve the most relevant results.\n\n"
        "If there are acronyms or unknown words, DO NOT rephrase them.\n\n"
        "Rules:\n"
        "- Always set content_search.\n"
        "- Always set title_search (short keywords).\n"
        "- ONLY set numeric/date filters when the user explicitly constrains them "
        "(e.g., 'under 5 minutes' => max_length_sec=300; 'in 2023' => a date range).\n"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )

    # Small, capable model; 0 temp for consistent schema
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    # Ask the model to emit exactly the TutorialSearch schema
    structured_llm = llm.with_structured_output(TutorialSearch)
    return prompt | structured_llm


# ──────────────────────────────────────────────────────────────────────────────
# 4) Example: translate TutorialSearch → backend (e.g., Chroma) filters
# ──────────────────────────────────────────────────────────────────────────────
def to_chroma_filter(q: TutorialSearch) -> Dict[str, Any]:
    """
    Convert our structured query into a Chroma-compatible filter dict.
    Adjust field names to match how you stored metadata in your DB.
    """
    filt: Dict[str, Any] = {}

    # view_count range
    if q.min_view_count is not None or q.max_view_count is not None:
        sub = {}
        if q.min_view_count is not None:
            sub["$gte"] = q.min_view_count
        if q.max_view_count is not None:
            sub["$lt"] = q.max_view_count
        filt["view_count"] = sub

    # publish_date range (store/compare as ISO dates for simplicity)
    def _iso(d: Optional[datetime.date]) -> Optional[str]:
        return d.isoformat() if isinstance(d, datetime.date) else None

    if q.earliest_publish_date is not None or q.latest_publish_date is not None:
        sub = {}
        if q.earliest_publish_date is not None:
            sub["$gte"] = _iso(q.earliest_publish_date)
        if q.latest_publish_date is not None:
            sub["$lt"] = _iso(q.latest_publish_date)
        filt["publish_date"] = sub

    # length range (seconds)
    if q.min_length_sec is not None or q.max_length_sec is not None:
        sub = {}
        if q.min_length_sec is not None:
            sub["$gte"] = q.min_length_sec
        if q.max_length_sec is not None:
            sub["$lt"] = q.max_length_sec
        filt["length"] = sub

    return filt


# ──────────────────────────────────────────────────────────────────────────────
# 5) Tiny demo runner
# ──────────────────────────────────────────────────────────────────────────────
def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing. Add it to your environment or a .env file.")

    # Optional: peek at real metadata fields (safe to fail; demo continues)
    demo_youtube_metadata()

    query_analyzer = build_query_analyzer()

    examples = [
        "rag from scratch",
        "videos on chat langchain published in 2023",
        "videos that are focused on the topic of chat langchain that are published before 2024",
        "how to use multi-modal models in an agent, only videos under 5 minutes",
    ]

    for q in examples:
        print("\n" + "=" * 80)
        print("User question:", q)
        try:
            structured: TutorialSearch = query_analyzer.invoke({"question": q})
            structured.pretty_print()
            filt = to_chroma_filter(structured)
            print("\n🧮 Backend filter (example for Chroma):", filt)
        except Exception as e:
            print("⚠️  Could not structure query:", e)


if __name__ == "__main__":
    main()
