"""Retrieval metrics.

Conventions:
- `retrieved` is a ranked list of doc ids, best first. Rank 1 is `retrieved[0]`.
- A document is relevant when its relevance grade is > 0.
- Only the first `k` retrieved documents count for "@k" metrics.
"""

import math

from hybrid_rag.schema import Qrels, Run


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the relevant documents that appear in the top k.

    Raise ValueError if `relevant` is empty (recall is undefined).
    """
    if not relevant:
        raise ValueError("relevant cannot be empty")

    top_k = retrieved[:k]
    retrieved_relevant = set(top_k) & relevant
    return len(retrieved_relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str], k: int | None = None) -> float:
    """1 / rank of the first relevant document, or 0.0 if none is found.

    If `k` is given, only the top k documents are searched (this is "MRR@k").
    """
    document = retrieved if k is None else retrieved[:k]

    for rank, doc in enumerate(document, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevance: dict[str, int], k: int) -> float:
    """Normalized Discounted Cumulative Gain with linear gain.

    DCG@k  = sum over ranks i = 1..k of  grade(doc at rank i) / log2(i + 1)
    IDCG@k = the same sum for the ideal ordering (all grades sorted descending)
    nDCG@k = DCG@k / IDCG@k, or 0.0 when IDCG@k is 0.
    Documents missing from `relevance` have grade 0.
    """
    top_k = retrieved[:k]

    # DCG of the actual ranking
    dcg = 0.0

    for rank, doc in enumerate(top_k, start=1):
        grade = relevance.get(doc, 0)
        dcg += grade / math.log2(rank + 1)

    # IDCG: best possible ranking
    ideal_grade = sorted(relevance.values(), reverse=True)[:k]

    idcg = 0.0

    for rank, grade in enumerate(ideal_grade, start=1):
        idcg += grade / math.log2(rank + 1)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def evaluate_run(
    run: Run, qrels: Qrels, k_values: tuple[int, ...] = (1, 5, 10)
) -> dict[str, float]:
    """Average every metric over the queries in `qrels` that have a relevant doc.

    Returns keys like "recall@5", "mrr@10", "ndcg@10" for each k in `k_values`.
    Queries whose qrels contain no grade > 0 are skipped (same convention as BEIR).
    If no query can be evaluated, every metric is 0.0.
    A query that is in `qrels` but missing from `run` scores 0 on every metric
    (it must still count in the average: a system that skips queries should not
    look better). Queries in `run` but not in `qrels` are ignored.
    """
    results: dict[str, float] = {}

    for k in k_values:
        recall_total = 0.0
        mrr_total = 0.0
        ndcg_total = 0.0

        evaluated_queries = 0

        for query_id, relevance in qrels.items():
            retrieved = run.get(query_id, [])
            relevant = {doc for doc, grade in relevance.items() if grade > 0}

            if not relevant:
                continue

            recall_total += recall_at_k(retrieved, relevant, k)

            mrr_total += reciprocal_rank(retrieved, relevant, k)

            ndcg_total += ndcg_at_k(retrieved, relevance, k)

            evaluated_queries += 1

        if evaluated_queries == 0:
            results[f"recall@{k}"] = 0.0
            results[f"mrr@{k}"] = 0.0
            results[f"ndcg@{k}"] = 0.0
        else:
            results[f"recall@{k}"] = recall_total / evaluated_queries
            results[f"mrr@{k}"] = mrr_total / evaluated_queries
            results[f"ndcg@{k}"] = ndcg_total / evaluated_queries
    return results
