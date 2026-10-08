"""Reciprocal Rank Fusion (RRF).

BM25 scores and cosine similarities live on different scales, so they cannot
be added. RRF ignores the scores and combines the *ranks* instead
(Cormack, Clarke and Buettcher, SIGIR 2009):

    score(d) = sum over the rankings i that contain d of  w_i / (k + rank_i(d))

where rank_i(d) starts at 1 for the best document of ranking i and w_i is the
weight of that ranking (1.0 when no weights are given). A document that is
missing from a ranking simply gets no contribution from it.

`k` controls how much a top position is worth compared with agreement between
rankings: with a small k, rank 1 in a single ranking can beat a document that
is ranked moderately well everywhere; with a large k, agreement wins. The
paper used k = 60.
"""


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
    weights: list[float] | None = None,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """Fuse several ranked lists of doc ids into one list of (doc_id, score).

    The result is sorted by score, highest first; ties are broken by doc_id in
    ascending order. If `top_k` is given, only the best `top_k` are returned.
    Raise ValueError if `weights` is given and its length differs from the
    number of rankings.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    elif len(weights) != len(rankings):
        raise ValueError(f"{len(rankings)} rankings but {len(weights)} weights")

    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)

    # Highest score first; equal scores fall back to ascending doc_id.
    fused = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return fused if top_k is None else fused[:top_k]
