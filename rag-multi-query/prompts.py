"""
Prompt templates (local; no Hub).
- system prompt for RAG answering
- prompt to generate multiple query variants
"""

from langchain_core.prompts import ChatPromptTemplate

# RAG answer prompt: strictly rely on context
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a helpful assistant. Answer the user using ONLY the provided context.\n"
         "If the answer is not in the context, say you don't know."),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ]
)

# Multi-query expansion prompt: produce distinct paraphrases
MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a query generation engine. Given a user question, create diverse, "
         "semantically different search queries that could retrieve relevant passages. "
         "Be concise and avoid redundancy."),
        ("human",
         "Question: {question}\n\n"
         "Generate {n} alternative search queries (one per line). "
         "Do NOT number them; just return each query on a new line.")
    ]
)


DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a question decomposition engine. Break the user's question into a small set of "
         "non-overlapping, concrete sub-questions that, if answered, would collectively answer the original question. "
         "Avoid redundancy. Focus on distinct facets or steps."),
        ("human",
         "Question: {question}\n\n"
         "Generate {n} sub-questions. Output one per line, no numbering.")
    ]
)

COMBINE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a careful synthesizer. Using ONLY the provided sub-answers, produce a concise final answer. "
         "If information is missing, say you don't know. Do not invent details."),
        ("human",
         "Original Question:\n{question}\n\n"
         "Sub-answers (one per line as 'Q: ... | A: ...'):\n{sub_answers}\n\n"
         "Final answer:")
    ]
)


HYDE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a domain expert. Draft a concise, factual answer to the question using general knowledge. "
         "This is a hypothetical draft for retrieval; do not hedge or say you lack context. 120–200 words."),
        ("human", "Question: {question}\n\nHypothetical answer:")
    ]
)