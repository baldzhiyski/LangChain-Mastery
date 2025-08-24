# =========================
# RAG: Web -> Chunks -> Embeddings -> Chroma -> Retrieval -> LLM Answer
# No LangSmith/Hub required. Uses cheapest OpenAI embeddings.
# =========================

import os
import bs4
from dotenv import load_dotenv  # loads OPENAI_API_KEY from .env (optional but recommended)

# LangChain core & community pieces
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma

# Prompt / chain utilities (replace LangChain Hub prompt with a local template)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# OpenAI LLM + Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# --------- 0) Load environment (.env) ---------
# Make sure you have a `.env` file with:
# OPENAI_API_KEY=your_openai_api_key_here
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY missing. Add it to your .env or environment.")


# --------- 1) INDEXING: Load -> Split -> Embed -> Store ---------

# Load Documents (from Lilian Weng's agent post)
# - SoupStrainer keeps only relevant parts of the HTML
# - headers/User-Agent helps avoid warnings and is polite to servers
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

docs = loader.load()

# Split long docs into overlapping chunks (helps retrieval & LLM context)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Embed with the CHEAPEST OpenAI embedding model (text-embedding-3-small)
# NOTE: This calls the OpenAI API and incurs a tiny cost.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create a local vector store (Chroma) in-memory
# (If you want persistence, add persist_directory="./chroma_db" and call .persist())
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()  # turns the store into a retriever


# --------- 2) RETRIEVAL + GENERATION: Prompt -> LLM ---------

# Local prompt template (replaces hub.pull("rlm/rag-prompt"))
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a helpful assistant. Answer the user using ONLY the provided context. "
         "If the answer is not in the context, say you don't know."),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ]
)

# LLM (deterministic output with temperature=0)
llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)

# Helper to format retrieved docs into a single string for the prompt
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain:
# - Takes the user's question
# - Uses the retriever to get relevant chunks and formats them as "context"
# - Fills the prompt -> sends to LLM -> returns a plain string
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# --------- 3) ASK A QUESTION ---------
# Triggers: retrieval -> prompt formatting -> LLM answer
answer = rag_chain.invoke("What is Task Decomposition?")
print("\nAnswer:\n", answer)
