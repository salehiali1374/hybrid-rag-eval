# Design

## Goal

Measure, stage by stage, how much each retrieval technique improves a Persian RAG system, and serve the best configuration behind an API. Every claim in the README must come from a number this repository produces.

## Data

**PersianQA** (Persian Wikipedia, SQuAD format), turned into a retrieval benchmark:

| Part | Source | Use |
| --- | --- | --- |
| Corpus | all unique paragraphs of train + test | documents to search |
| Test queries (answerable) | test split | final retrieval and answer metrics |
| Train queries | train split | tuning choices (fusion constant, top-k), never the test set |
| Unanswerable queries | `is_impossible` questions | does the system refuse instead of inventing an answer? |

Known limitation: each question has exactly one relevant paragraph, so recall@k equals hit rate@k.

## Pipeline

```
query ─┬─> BM25 (sparse) ─┐
       └─> dense (e5) ────┴─> RRF fusion ─> cross-encoder rerank ─> top-k context ─> LLM ─> answer + citations
```

## Hardware constraints

CPU only, about 10 GB RAM. Therefore: small multilingual embedding model (e.g. `intfloat/multilingual-e5-small`), reranking only the top 20–50 candidates, and LLM calls through an OpenAI-compatible endpoint (OpenRouter, or a small local model via Ollama). Retrieval evaluation needs no LLM at all.

## Milestones

| # | Milestone | Done when |
| --- | --- | --- |
| 0 | Skeleton: package, config, data loader, API health, Docker, CI | tests pass, container starts |
| 1 | Metrics: recall@k, MRR, nDCG, `evaluate_run` | `tests/test_metrics.py` passes |
| 2 | BM25 from scratch + Persian text normalization | first baseline numbers on test |
| 3 | Dense retrieval with embedding cache | dense numbers on test |
| 4 | Reciprocal rank fusion, constant tuned on train | ablation table: BM25 / dense / hybrid |
| 5 | Cross-encoder reranking | ablation row + latency cost |
| 6 | Generation with citations and refusal | answers cite paragraph ids |
| 7 | Answer evaluation: LLM judge checked against ~50 human labels | judge agreement reported |
| 8 | Retrieval API, Docker Compose, tracing, CI eval gate | `docker compose up` serves queries |
| 9 | README with results, limitations, how to reproduce | |

## Out of scope

Chat UI, authentication, multi-tenancy, Kubernetes, fine-tuning an LLM, supporting several vector databases.
