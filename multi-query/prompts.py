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
