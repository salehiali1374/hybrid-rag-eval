"""Retrieval metrics.

Conventions:
- `retrieved` is a ranked list of doc ids, best first. Rank 1 is `retrieved[0]`.
- A document is relevant when its relevance grade is > 0.
- Only the first `k` retrieved documents count for "@k" metrics.
"""

from hybrid_rag.schema import Qrels, Run


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the relevant documents that appear in the top k.

    Raise ValueError if `relevant` is empty (recall is undefined).
    """
    raise NotImplementedError


def reciprocal_rank(retrieved: list[str], relevant: set[str], k: int | None = None) -> float:
    """1 / rank of the first relevant document, or 0.0 if none is found.

    If `k` is given, only the top k documents are searched (this is "MRR@k").
    """
    raise NotImplementedError


def ndcg_at_k(retrieved: list[str], relevance: dict[str, int], k: int) -> float:
    """Normalized Discounted Cumulative Gain with linear gain.

    DCG@k  = sum over ranks i = 1..k of  grade(doc at rank i) / log2(i + 1)
    IDCG@k = the same sum for the ideal ordering (all grades sorted descending)
    nDCG@k = DCG@k / IDCG@k, or 0.0 when IDCG@k is 0.
    Documents missing from `relevance` have grade 0.
    """
    raise NotImplementedError


def evaluate_run(
    run: Run, qrels: Qrels, k_values: tuple[int, ...] = (1, 5, 10)
) -> dict[str, float]:
    """Average every metric over all queries in `qrels`.

    Returns keys like "recall@5", "mrr@10", "ndcg@10" for each k in `k_values`.
    A query that is in `qrels` but missing from `run` scores 0 on every metric
    (it must still count in the average: a system that skips queries should not
    look better). Queries in `run` but not in `qrels` are ignored.
    """
    raise NotImplementedError
