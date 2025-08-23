"""
Multi-Query RAG (no LangSmith/Hub).
- Index web docs -> split -> embed -> persist in Chroma
- Generate multiple query variants via LLM
- Retrieve per variant, merge & dedupe, optional MMR
- Feed context to LLM with local prompt
- Print retrieved chunks & answer

Usage examples:
python main.py --index --url https://lilianweng.github.io/posts/2023-06-23-agent/ --persist ./stores/chroma_db
python main.py --ask "What is Task Decomposition?" --persist ./stores/chroma_db --mmr

"""

import os
import argparse
from typing import List, Set

from dotenv import load_dotenv

# LLM client used for both: (a) generating query variants, (b) answering
from langchain_openai import ChatOpenAI

# Turn a ChatPromptTemplate output into a final string
from langchain_core.output_parsers import StrOutputParser

# Your local modules (NOTE: fixed typos in the import names)
from embeddings import make_embeddings                       # picks OpenAI or HF embeddings
from prompts import RAG_PROMPT, MULTI_QUERY_PROMPT          # local prompt templates
from vectorestore import (                                   # loader/splitter/chroma helpers
    load_web_docs,
    split_docs,
    build_and_persist_chroma,
    load_persisted_chroma,
)
from utils import print_retrieved                     # pretty-print retrieved chunks
from config import (                                        # central config flags / defaults
    OPENAI_API_KEY,
    OPENAI_CHAT_MODEL,
    TOP_K_PER_QUERY,
    NUM_QUERY_VARIANTS,
    MMR_DIVERSITY,
    DEFAULT_PERSIST_DIR,
)


def generate_query_variants(llm: ChatOpenAI, question: str, n: int) -> List[str]:
    """
    Use the LLM to produce n diverse paraphrases of the user's question.
    Why? Multi-Query Retrieval broadens recall: different phrasings can surface
    different chunks from your vector DB.

    Steps:
      1) Format the multi-query prompt with the original question + n desired variants.
      2) Ask the model (temperature=0 keeps it concise/consistent).
      3) Split lines and de-duplicate (case-insensitive).
      4) Ensure the *original* question is included as the first variant.
    """
    prompt = MULTI_QUERY_PROMPT.format_messages(question=question, n=n)
    raw = llm.invoke(prompt)                       # ChatOpenAI returns a Message
    text = raw.content.strip()
    variants = [line.strip() for line in text.split("\n") if line.strip()]

    # Deduplicate while preserving order
    seen: Set[str] = set()
    uniq: List[str] = []
    for v in variants:
        key = v.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(v)

    # Always include the original question at the front if missing
    if question.lower() not in seen:
        uniq.insert(0, question)

    return uniq


def retrieve_multi(vectorstore, embeddings, question: str, k: int, use_mmr: bool, mmr_lambda: float) -> List:
    """
    Retrieve context with multi-query logic and (optionally) MMR.

    What happens here:
      - We turn the vector store into a retriever (top-k nearest neighbors).
      - We ask an LLM to generate `NUM_QUERY_VARIANTS` paraphrases of the question.
      - For each variant, we do a vector search:
          * If use_mmr=True, try Max Marginal Relevance (diversity-aware).
          * Else, do classic similarity search.
      - We merge all docs and de-duplicate them using a (source, content-hash) signature.

    Notes:
      - `embeddings` param is unused here (kept for API symmetry / future tweaks).
      - If your installed vectorstore doesn't have MMR, we fall back to similarity search.
    """
    # Make a small, deterministic LLM to generate the query variants
    variants_llm = ChatOpenAI(model_name=OPENAI_CHAT_MODEL, temperature=0)
    sub_queries = generate_query_variants(variants_llm, question, NUM_QUERY_VARIANTS)

    merged = []
    seen_ids = set()

    for q in sub_queries:
        if use_mmr:
            # MMR tries to balance relevance (to q) with diversity (among selected docs)
            try:
                docs = vectorstore.max_marginal_relevance_search(
                    q,
                    k=k,
                    fetch_k=max(32, k * 3),   # over-fetch then diversify
                    lambda_mult=mmr_lambda,    # 0..1 (higher => more diversity)
                )
            except AttributeError:
                # If not available in your version, fall back to standard similarity
                docs = vectorstore.similarity_search(q, k=k)
        else:
            docs = vectorstore.similarity_search(q, k=k)

        # Merge-dedupe: avoid repeating the same chunk multiple times
        for d in docs:
            sig = (d.metadata.get("source", ""), hash(d.page_content))
            if sig not in seen_ids:
                seen_ids.add(sig)
                merged.append(d)

    return merged


def build_rag_chain(vectorstore, llm: ChatOpenAI):
    """
    Build the *answering* chain.

    Why not pass a retriever directly?
    - We want to control *which* docs go in (multi-query union), so we inject
      context manually and keep the chain itself simple:
        {context, question} -> RAG_PROMPT -> LLM -> string
    """
    def format_docs(docs):
        # Take the top retrieved chunks and join as a single context block
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        RAG_PROMPT     # prompt template requiring {context} and {question}
        | llm          # LLM to generate the final answer
        | StrOutputParser()  # convert the LLM's message to plain text
    )
    return chain, format_docs


def index_urls(urls: List[str], persist_dir: str):
    """
    Full indexing pipeline:
      1) Download + parse the pages (only keep relevant parts via SoupStrainer).
      2) Split into overlapping chunks (better retrieval & context continuity).
      3) Embed with your chosen embedding model (OpenAI or HuggingFace).
      4) Save the embeddings to a persistent Chroma DB (so you pay/compute once).
    """
    docs = load_web_docs(urls)
    splits = split_docs(docs)
    embeddings = make_embeddings()
    _ = build_and_persist_chroma(splits, embeddings, persist_dir)
    print(f"✅ Indexed & persisted to: {persist_dir}")


def answer_question(question: str, persist_dir: str, use_mmr: bool):
    """
    Question-answering pipeline:
      1) Load the persisted Chroma DB (no re-embedding).
      2) Multi-query retrieval (paraphrase question -> search -> merge results).
      3) Print which chunks were used (source + snippet) for transparency.
      4) Build RAG prompt with the retrieved context and ask the LLM.
      5) Print the final grounded answer.
    """
    embeddings = make_embeddings()
    vs = load_persisted_chroma(embeddings, persist_dir)

    # Do the actual multi-query retrieval (optionally with MMR)
    retrieved = retrieve_multi(
        vectorstore=vs,
        embeddings=embeddings,    # not used in this function but kept for API symmetry
        question=question,
        k=TOP_K_PER_QUERY,
        use_mmr=use_mmr,
        mmr_lambda=MMR_DIVERSITY
    )

    # Show what context we fed to the LLM (helps you debug/learn)
    print_retrieved(retrieved, title="Retrieved Context")

    # You need an OpenAI key only for the *answering* LLM in this script
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required to use ChatOpenAI for answering.")
    llm = ChatOpenAI(model_name=OPENAI_CHAT_MODEL, temperature=0)

    # Build the simple answering chain and inject the merged context
    chain, format_docs = build_rag_chain(vs, llm)
    context_text = format_docs(retrieved)

    # Fire the chain: {context, question} -> prompt -> LLM -> string
    answer = chain.invoke({"context": context_text, "question": question})

    print("\n=== Answer ===\n")
    print(answer)


def main():
    """
    CLI entrypoint:
      --index  : run the indexing pipeline (download/split/embed/save)
      --ask    : run the QA flow using the saved Chroma DB
      --url    : one or more pages to index
      --persist: where to store/load the Chroma DB
      --mmr    : enable diversity-aware retrieval (recommended when chunks are redundant)
    """
    load_dotenv()  # load .env once here (e.g., OPENAI_API_KEY, USER_AGENT)
    parser = argparse.ArgumentParser(description="Multi-Query RAG (no LangSmith)")
    parser.add_argument("--index", action="store_true", help="Run indexing pipeline")
    parser.add_argument("--ask", type=str, help="Ask a question using multi-query RAG")
    parser.add_argument("--url", action="append", help="One or more URLs to index", default=[])
    parser.add_argument("--persist", type=str, default=DEFAULT_PERSIST_DIR, help="Chroma persist directory")
    parser.add_argument("--mmr", action="store_true", help="Enable Max Marginal Relevance re-ranking")

    args = parser.parse_args()

    # Ensure persist directory exists so Chroma can write files
    os.makedirs(args.persist, exist_ok=True)

    # Run indexing if requested
    if args.index:
        if not args.url:
            raise SystemExit("Please provide at least one --url to index.")
        index_urls(args.url, args.persist)

    # Run QA if requested
    if args.ask:
        answer_question(args.ask, args.persist, use_mmr=args.mmr)

    # Show help if no action flags were provided
    if not args.index and not args.ask:
        parser.print_help()


if __name__ == "__main__":
    main()
