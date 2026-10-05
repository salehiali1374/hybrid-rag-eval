"""Evaluate hybrid retrieval (BM25 + dense, fused with RRF) on PersianQA.

Two modes:

    # 1. choose the RRF constant and the dense weight on the TRAIN split
    python -m hybrid_rag.experiments.hybrid --split train --tune

    # 2. report BM25 vs dense vs hybrid with the chosen values
    python -m hybrid_rag.experiments.hybrid --split test --rrf-k 60 --dense-weight 1.0

Tuning on the test split is refused on purpose: parameters picked on the data
you report on make the reported number too optimistic.
"""

import argparse
import json
from pathlib import Path

from hybrid_rag.config import get_settings
from hybrid_rag.data.persianqa import load_persianqa
from hybrid_rag.eval.metrics import evaluate_run
from hybrid_rag.retrieval.bm25 import BM25
from hybrid_rag.retrieval.dense import DenseIndex, Encoder
from hybrid_rag.retrieval.embedding_cache import load_or_compute
from hybrid_rag.retrieval.fusion import reciprocal_rank_fusion
from hybrid_rag.schema import Document, Query, Run
from hybrid_rag.text.normalize import tokenize

K_VALUES = (1, 5, 10, 20)
TUNE_RRF_K = (1, 5, 10, 20, 40, 60, 100)
TUNE_DENSE_WEIGHT = (1.0, 1.5, 2.0, 3.0)
SELECTION_METRIC = "mrr@10"


def bm25_run(docs: list[Document], queries: list[Query], depth: int) -> Run:
    bm25 = BM25({doc.id: tokenize(doc.text) for doc in docs})
    return {
        query.id: [doc_id for doc_id, _ in bm25.search(tokenize(query.text), top_k=depth)]
        for query in queries
    }


def dense_run(docs: list[Document], queries: list[Query], depth: int, model_name: str) -> Run:
    from sentence_transformers import SentenceTransformer  # heavy import, only needed here

    cache_dir = get_settings().data_dir / "embeddings"
    encoder = Encoder(SentenceTransformer(model_name, device="cpu"))
    doc_embeddings = load_or_compute(
        cache_dir, model_name, "passage", [d.text for d in docs], encoder.encode_passages
    )
    query_embeddings = load_or_compute(
        cache_dir, model_name, "query", [q.text for q in queries], encoder.encode_queries
    )
    hits = DenseIndex([d.id for d in docs], doc_embeddings).search_batch(query_embeddings, depth)
    return {q.id: [doc_id for doc_id, _ in row] for q, row in zip(queries, hits, strict=True)}


def fuse(bm25: Run, dense: Run, rrf_k: int, dense_weight: float) -> Run:
    return {
        query_id: [
            doc_id
            for doc_id, _ in reciprocal_rank_fusion(
                [bm25[query_id], dense[query_id]],
                k=rrf_k,
                weights=[1.0, dense_weight],
                top_k=max(K_VALUES),
            )
        ]
        for query_id in bm25
    }


def rounded(metrics: dict[str, float]) -> dict[str, float]:
    return {name: round(value, 4) for name, value in metrics.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--tune", action="store_true", help="grid search (train split only)")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--depth", type=int, default=100, help="candidates per retriever")
    parser.add_argument("--model", default=get_settings().embedding_model)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.tune and args.split != "train":
        parser.error("--tune is only allowed with --split train")

    dataset = load_persianqa(get_settings().data_dir)
    query_set = dataset.splits[args.split]
    docs = list(dataset.corpus.values())
    queries = list(query_set.queries.values())

    bm25 = bm25_run(docs, queries, args.depth)
    dense = dense_run(docs, queries, args.depth, args.model)
    args.out.mkdir(parents=True, exist_ok=True)
    common = {
        "dataset": dataset.name,
        "split": args.split,
        "num_docs": len(docs),
        "num_queries": len(queries),
        "depth": args.depth,
        "model": args.model,
    }

    if args.tune:
        grid = []
        for rrf_k in TUNE_RRF_K:
            for dense_weight in TUNE_DENSE_WEIGHT:
                metrics = evaluate_run(fuse(bm25, dense, rrf_k, dense_weight), query_set.qrels)
                grid.append(
                    {"rrf_k": rrf_k, "dense_weight": dense_weight, "metrics": rounded(metrics)}
                )
        grid.sort(key=lambda row: row["metrics"][SELECTION_METRIC], reverse=True)
        report = {**common, "selection_metric": SELECTION_METRIC, "best": grid[0], "grid": grid}
        out_file = args.out / "hybrid_tune_train.json"
        out_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

        print(f"{'rrf_k':>6} {'dense_w':>8} {'recall@1':>9} {'mrr@10':>8} {'recall@10':>10}")
        for row in grid:
            m = row["metrics"]
            print(
                f"{row['rrf_k']:>6} {row['dense_weight']:>8} "
                f"{m['recall@1']:>9.4f} {m['mrr@10']:>8.4f} {m['recall@10']:>10.4f}"
            )
        print(f"\nbest by {SELECTION_METRIC}: {grid[0]}\nsaved {out_file}")
        return

    hybrid = fuse(bm25, dense, args.rrf_k, args.dense_weight)
    methods = {
        "bm25": evaluate_run(bm25, query_set.qrels, K_VALUES),
        "dense": evaluate_run(dense, query_set.qrels, K_VALUES),
        "hybrid_rrf": evaluate_run(hybrid, query_set.qrels, K_VALUES),
    }
    report = {
        **common,
        "params": {"rrf_k": args.rrf_k, "dense_weight": args.dense_weight},
        "methods": {name: rounded(metrics) for name, metrics in methods.items()},
    }
    out_file = args.out / f"hybrid_{args.split}_k-{args.rrf_k}_w-{args.dense_weight}.json"
    out_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"{'method':<12} {'recall@1':>9} {'mrr@10':>8} {'recall@10':>10} {'recall@20':>10}")
    for name, metrics in methods.items():
        print(
            f"{name:<12} {metrics['recall@1']:>9.4f} {metrics['mrr@10']:>8.4f} "
            f"{metrics['recall@10']:>10.4f} {metrics['recall@20']:>10.4f}"
        )
    print(f"\nsaved {out_file}")


if __name__ == "__main__":
    main()
