# How to Run (Ways to Start) — Multi-Query RAG

> Run these from the folder containing `main.py`. Replace the URL or question as you like. The `--persist` directory can be any path you choose.

---

## 1) Index (creates embeddings once)

**What it does:** Downloads the page(s) → splits into chunks → computes embeddings → saves a Chroma DB to disk (so you don’t re-embed next time).

```bash
python main.py --index --url https://lilianweng.github.io/posts/2023-06-23-agent/ --persist ./stores/chroma_db
```

## 2) Ask (basic multi‑query)

**What it does:** Loads the saved DB → generates multiple query variants → retrieves top‑k per variant → merges results → answers with a RAG prompt.

```bash
python main.py --ask "What is Task Decomposition?" --persist ./stores/chroma_db
```

## 3) Ask + MMR (diversify results)

**What it does:** Same as basic ask, but uses **Max‑Marginal‑Relevance** per variant to avoid redundant chunks (more diverse context).

```bash
python main.py --ask "What is Task Decomposition?" --persist ./stores/chroma_db --mmr
```

## 4) Ask + Show Variants (see the queries)

**What it does:** Prints the LLM‑generated query variants before retrieval, then proceeds like the chosen mode (basic, MMR, or Fusion).

```bash
python main.py --ask "What is Task Decomposition?" --persist ./stores/chroma_db --show-variants
```

## 5) Ask with **RAG‑Fusion** (RRF across variants)

**What it does:** Runs a search for each variant, then **fuses** the ranked lists with **Reciprocal Rank Fusion (RRF)** to get a consensus top‑k.

```bash
python main.py --ask "What is Task Decomposition?" --persist ./stores/chroma_db --fusion
```

_(Optionally also see the variants):_

```bash
python main.py --ask "What is Task Decomposition?" --persist ./stores/chroma_db --fusion --show-variants
```

## 6) Decompose

**What it does:** Decompose into 3 sub-questions, MMR per sub-q, show sub-qs

```bash

python main.py --index --url https://lilianweng.github.io/posts/2023-06-23-agent/ --url https://arxiv.org/abs/2210.03629 --url https://arxiv.org/abs/2305.10601 --url https://arxiv.org/abs/2305.04091 --persist ./stores/chroma_db --ask "Explain how agents plan tasks across multiple steps" --decompose --subqs 3 --mmr --show-subqs --show-subanswers


```

## 7) Index multiple URLs

**What it does:** Embeds more sources into the same DB (adds to existing).

```bash
python main.py --index   --url https://lilianweng.github.io/posts/2023-06-23-agent/   --url https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/   --persist ./stores/chroma_db
```

## 8) Clean rebuild (start fresh)

**What it does:** Deletes the existing Chroma DB so a new `--index` starts from scratch.

- Windows PowerShell:

```powershell
Remove-Item -Recurse -Force .\stores\chroma_db
```

- macOS/Linux:

```bash
rm -rf ./stores/chroma_db
```

## 9) One‑liner: index then ask

**What it does:** Embeds the source, then immediately asks a question using the new DB.

- Windows PowerShell:

```powershell
python main.py --index --url https://lilianweng.github.io/posts/2023-06-23-agent/ --persist ./stores/chroma_db; python main.py --ask "What is Task Decomposition?" --persist ./stores/chroma_db --mmr
```

---

## Differences at a Glance

| Mode                   | What it focuses on                                                             | When to use                                                 |
| ---------------------- | ------------------------------------------------------------------------------ | ----------------------------------------------------------- |
| **Ask (basic)**        | Multi‑query recall + simple merge                                              | Fast, simple baseline                                       |
| **Ask + MMR**          | **Diversity** within each variant’s results (reduces duplicates)               | Overlapping chunks / repetitive retrieval                   |
| **Ask + Fusion (RRF)** | **Consensus** across variants (stable top‑k)                                   | Ambiguous queries, synonym‑heavy corpora                    |
| **Show Variants**      | Transparency                                                                   | Debug/learning—see what the model searched for              |
| **Ask + Decompose**    | Multi-hop coverage: split into sub-questions, retrieve/answer each, synthesize | Complex/broad questions needing multiple pieces of evidence |

Ask + Decompose
Multi-hop coverage: split into sub-questions, retrieve/answer each, synthesize
Complex/broad questions needing multiple pieces of evidence

> Tip: You can combine **Fusion + Show Variants**, or use **MMR** without Fusion, depending on whether you want **diversity** (MMR) or **consensus** (Fusion).
