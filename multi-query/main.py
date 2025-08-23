import os
import argparse
from typing import List, Set, Dict, Tuple

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Local modules — make sure your filenames match these:
from embeddings import make_embeddings
from prompts import COMBINE_PROMPT, DECOMPOSE_PROMPT, RAG_PROMPT, MULTI_QUERY_PROMPT
from vectorestore import (   # NOTE: was 'vectorestore' before — fixed
    load_web_docs,
    split_docs,
    build_and_persist_chroma,
    load_persisted_chroma,
)
from utils import print_retrieved  # NOTE: was 'utils' before — fixed
from config import (
    OPENAI_API_KEY,
    OPENAI_CHAT_MODEL,
    TOP_K_PER_QUERY,
    NUM_QUERY_VARIANTS,
    MMR_DIVERSITY,
    DEFAULT_PERSIST_DIR,
    RRF_K,
    PER_QUERY_K_FUSION,
)

# -------------------------------
# Multi-Query: generate variants
# -------------------------------
def generate_query_variants(llm: ChatOpenAI, question: str, n: int) -> List[str]:
    prompt = MULTI_QUERY_PROMPT.format_messages(question=question, n=n)
    raw = llm.invoke(prompt)
    text = (raw.content or "").strip()
    variants = [line.strip() for line in text.split("\n") if line.strip()]

    # de-dup while preserving order; ensure original is included
    seen: Set[str] = set()
    uniq: List[str] = []
    for v in variants:
        key = v.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(v)
    if question.lower() not in seen:
        uniq.insert(0, question)
    return uniq

# ----------------------------------
# Multi-Query retrieval (+ optional MMR)
# ----------------------------------
def retrieve_multi(vectorstore, embeddings, question: str, k: int, use_mmr: bool, mmr_lambda: float) -> List:
    variants_llm = ChatOpenAI(model_name=OPENAI_CHAT_MODEL, temperature=0)
    sub_queries = generate_query_variants(variants_llm, question, NUM_QUERY_VARIANTS)

    merged = []
    seen_ids = set()
    for q in sub_queries:
        if use_mmr:
            try:
                docs = vectorstore.max_marginal_relevance_search(
                    q, k=k, fetch_k=max(32, k * 3), lambda_mult=mmr_lambda
                )
            except AttributeError:
                docs = vectorstore.similarity_search(q, k=k)
        else:
            docs = vectorstore.similarity_search(q, k=k)

        for d in docs:
            sig = (d.metadata.get("source", "") or d.metadata.get("url", ""), hash(d.page_content))
            if sig not in seen_ids:
                seen_ids.add(sig)
                merged.append(d)
    return merged

# -------------------------------
# RAG Fusion (Reciprocal Rank Fusion)
# -------------------------------
def rag_fusion_rrf(
    vectorstore,
    sub_queries: List[str],
    per_query_k: int = PER_QUERY_K_FUSION,
    rrf_k: int = RRF_K,
    final_k: int = TOP_K_PER_QUERY,
):
    scores: Dict[Tuple[str, int], float] = {}
    keep_one_doc_obj: Dict[Tuple[str, int], object] = {}

    for q in sub_queries:
        docs = vectorstore.similarity_search(q, k=per_query_k)
        for rank, d in enumerate(docs, start=1):
            sig = (d.metadata.get("source", "") or d.metadata.get("url", ""), hash(d.page_content))
            scores[sig] = scores.get(sig, 0.0) + 1.0 / (rrf_k + rank)
            keep_one_doc_obj.setdefault(sig, d)

    top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:final_k]
    return [keep_one_doc_obj[sig] for sig, _ in top]

# -------------------------------
# RAG answer chain
# -------------------------------
def build_rag_chain(llm: ChatOpenAI):
    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (RAG_PROMPT | llm | StrOutputParser())
    return chain, format_docs

# -------------------------------
# Indexing pipeline
# -------------------------------
def index_urls(urls: List[str], persist_dir: str):
    docs = load_web_docs(urls)
    splits = split_docs(docs)
    embeddings = make_embeddings()
    _ = build_and_persist_chroma(splits, embeddings, persist_dir)
    print(f"✅ Indexed & persisted to: {persist_dir}")

# -------------------------------
# Ask: Multi-Query (+MMR or Fusion)
# -------------------------------
def answer_question(question: str, persist_dir: str, use_mmr: bool, use_fusion: bool = False, show_variants: bool = False):
    embeddings = make_embeddings()
    vs = load_persisted_chroma(embeddings, persist_dir)

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required to use ChatOpenAI for answering.")
    variants_llm = ChatOpenAI(model_name=OPENAI_CHAT_MODEL, temperature=0)
    answer_llm = ChatOpenAI(model_name=OPENAI_CHAT_MODEL, temperature=0)

    sub_queries = generate_query_variants(variants_llm, question, NUM_QUERY_VARIANTS)
    if show_variants:
        print("\n=== Query Variants ===")
        for i, qv in enumerate(sub_queries, 1):
            print(f"{i:>2}. {qv}")

    if use_fusion:
        retrieved = rag_fusion_rrf(
            vectorstore=vs,
            sub_queries=sub_queries,
            per_query_k=PER_QUERY_K_FUSION,
            rrf_k=RRF_K,
            final_k=TOP_K_PER_QUERY
        )
    else:
        retrieved = retrieve_multi(
            vectorstore=vs,
            embeddings=embeddings,
            question=question,
            k=TOP_K_PER_QUERY,
            use_mmr=use_mmr,
            mmr_lambda=MMR_DIVERSITY
        )

    print_retrieved(retrieved, title="Retrieved Context")

    chain, format_docs = build_rag_chain(answer_llm)
    context_text = format_docs(retrieved)
    answer = chain.invoke({"context": context_text, "question": question})

    print("\n=== Answer ===\n")
    print(answer)

# -------------------------------
# Decomposition helpers
# -------------------------------
def generate_subquestions(llm: ChatOpenAI, question: str, n: int) -> List[str]:
    msgs = DECOMPOSE_PROMPT.format_messages(question=question, n=n)
    resp = llm.invoke(msgs)
    text = (resp.content or "").strip()
    subs = [line.strip() for line in text.split("\n") if line.strip()]
    seen = set(); uniq = []
    for s in subs:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(s)
    return uniq

def answer_with_context(llm: ChatOpenAI, docs, question: str) -> str:
    chain, format_docs = build_rag_chain(llm)
    ctx = format_docs(docs)
    return chain.invoke({"context": ctx, "question": question})

def answer_decomposed(
    question: str,
    persist_dir: str,
    num_subqs: int = 3,
    use_mmr: bool = True,
    show_subqs: bool = True,
    show_subanswers: bool = False,  # <— NEW: print each sub-answer
):
    embeddings = make_embeddings()
    vs = load_persisted_chroma(embeddings, persist_dir)

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required to use ChatOpenAI.")
    planner_llm = ChatOpenAI(model_name=OPENAI_CHAT_MODEL, temperature=0)
    answer_llm = ChatOpenAI(model_name=OPENAI_CHAT_MODEL, temperature=0)
    combine_llm = ChatOpenAI(model_name=OPENAI_CHAT_MODEL, temperature=0)

    # 1) Decompose
    subqs = generate_subquestions(planner_llm, question, num_subqs)
    if show_subqs:
        print("\n=== Sub-questions ===")
        for i, s in enumerate(subqs, 1):
            print(f"{i:>2}. {s}")

    # 2) Retrieve & answer each sub-question
    sub_pairs = []
    for idx, sq in enumerate(subqs, 1):
        try:
            docs = vs.max_marginal_relevance_search(
                sq, k=TOP_K_PER_QUERY, fetch_k=max(32, TOP_K_PER_QUERY*3), lambda_mult=MMR_DIVERSITY
            ) if use_mmr else vs.similarity_search(sq, k=TOP_K_PER_QUERY)
        except AttributeError:
            docs = vs.similarity_search(sq, k=TOP_K_PER_QUERY)

        print_retrieved(docs, title=f"Retrieved for: {sq}")

        sub_answer = answer_with_context(answer_llm, docs, sq)
        sub_pairs.append((sq, sub_answer))

        # <— NEW: print each sub-answer as we go
        if show_subanswers:
            print(f"\n--- Sub-answer {idx} ---")
            print(f"Q: {sq}\nA: {sub_answer}\n")

    # 3) Synthesize final answer
    lines = [f"Q: {q} | A: {a}" for (q, a) in sub_pairs]
    msgs = COMBINE_PROMPT.format_messages(
        question=question,
        sub_answers="\n".join(lines)
    )
    final = combine_llm.invoke(msgs).content

    print("\n=== Final Answer (Decomposed) ===\n")
    print(final)

# -------------------------------
# CLI
# -------------------------------
def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Multi-Query RAG (no LangSmith)")

    # Core modes
    parser.add_argument("--index", action="store_true", help="Run indexing pipeline")
    parser.add_argument("--ask", type=str, help="Ask a question using multi-query RAG")

    # Inputs & storage
    parser.add_argument("--url", action="append", default=[], help="One or more URLs to index")
    parser.add_argument("--persist", type=str, default=DEFAULT_PERSIST_DIR, help="Chroma persist directory")

    # Retrieval controls
    parser.add_argument("--mmr", action="store_true", help="Enable Max Marginal Relevance re-ranking")
    parser.add_argument("--fusion", action="store_true", help="Use RAG-Fusion (RRF) across variants")
    parser.add_argument("--show-variants", action="store_true", help="Print the generated query variants")

    # Decomposition
    parser.add_argument("--decompose", action="store_true", help="Use query decomposition pipeline")
    parser.add_argument("--subqs", type=int, default=3, help="Number of sub-questions to generate")
    parser.add_argument("--show-subqs", action="store_true", help="Print generated sub-questions")
    parser.add_argument("--show-subanswers", action="store_true", help="Print each sub-answer before the final")

    args = parser.parse_args()
    os.makedirs(args.persist, exist_ok=True)

    # Indexing
    if args.index:
        if not args.url:
            raise SystemExit("Please provide at least one --url to index.")
        index_urls(args.url, args.persist)

    # Ask: choose normal / fusion / decomposition paths
    if args.ask and args.decompose:
        return answer_decomposed(
            args.ask,
            args.persist,
            num_subqs=args.subqs,
            use_mmr=args.mmr,
            show_subqs=args.show_subqs,
            show_subanswers=args.show_subanswers,
        )

    if args.ask:
        return answer_question(
            args.ask,
            args.persist,
            use_mmr=args.mmr,
            use_fusion=args.fusion,
            show_variants=args.show_variants,
        )

    if not args.index and not args.ask:
        parser.print_help()

if __name__ == "__main__":
    main()
