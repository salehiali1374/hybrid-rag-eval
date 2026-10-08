"""Evaluate cross-encoder reranking on top of hybrid retrieval on PersianQA.

Pipeline: BM25 + dense -> RRF (constants tuned on train, see experiments.hybrid)
-> the cross-encoder rescores the best `--depth` candidates.

Usage:
    # quick latency check on 30 queries
    python -m hybrid_rag.experiments.rerank --split test --limit 30

    # full run
    python -m hybrid_rag.experiments.rerank --split test --depth 20

The reranker has no tunable constant besides `--depth`. It is fixed up front
(20, the low end of the 20-50 planned in docs/design.md) instead of being
picked on the test split.
"""

import argparse
import json
import time
from pathlib import Path

from hybrid_rag.config import get_settings
from hybrid_rag.data.persianqa import load_persianqa
from hybrid_rag.eval.metrics import evaluate_run
from hybrid_rag.experiments.hybrid import bm25_run, dense_run, fuse, rounded
from hybrid_rag.retrieval.rerank import Reranker
from hybrid_rag.schema import Run

K_VALUES = (1, 5, 10, 20)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--depth", type=int, default=20, help="candidates sent to the reranker")
    parser.add_argument("--rrf-k", type=int, default=1)
    parser.add_argument("--dense-weight", type=float, default=1.5)
    parser.add_argument("--retrieval-depth", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None, help="only the first N queries")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--embedding-model", default=get_settings().embedding_model)
    parser.add_argument("--reranker-model", default=get_settings().reranker_model)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.depth > max(K_VALUES):
        parser.error(f"--depth must be <= {max(K_VALUES)}, the largest evaluated k")

    dataset = load_persianqa(get_settings().data_dir)
    query_set = dataset.splits[args.split]
    docs = list(dataset.corpus.values())
    queries = list(query_set.queries.values())[: args.limit]
    qrels = {q.id: query_set.qrels[q.id] for q in queries}

    bm25 = bm25_run(docs, queries, args.retrieval_depth)
    dense = dense_run(docs, queries, args.retrieval_depth, args.embedding_model)
    hybrid = fuse(bm25, dense, args.rrf_k, args.dense_weight)

    from sentence_transformers import CrossEncoder  # heavy import, only needed here

    reranker = Reranker(
        CrossEncoder(args.reranker_model, device="cpu", max_length=512),
        batch_size=args.batch_size,
    )

    reranked: Run = {}
    start = time.perf_counter()
    for i, query in enumerate(queries, start=1):
        candidates = [
            (doc_id, dataset.corpus[doc_id].text) for doc_id in hybrid[query.id][: args.depth]
        ]
        top = [doc_id for doc_id, _ in reranker.rerank(query.text, candidates)]
        # Ranks below `depth` keep their hybrid order, so recall@k stays comparable for k > depth.
        reranked[query.id] = top + hybrid[query.id][args.depth : max(K_VALUES)]
        if i % 25 == 0 or i == len(queries):
            elapsed = time.perf_counter() - start
            print(f"reranked {i}/{len(queries)} queries, {elapsed / i:.2f} s/query", flush=True)
    rerank_seconds = time.perf_counter() - start

    methods = {
        "bm25": evaluate_run(bm25, qrels, K_VALUES),
        "dense": evaluate_run(dense, qrels, K_VALUES),
        "hybrid_rrf": evaluate_run(hybrid, qrels, K_VALUES),
        "hybrid_rrf+rerank": evaluate_run(reranked, qrels, K_VALUES),
    }
    report = {
        "dataset": dataset.name,
        "split": args.split,
        "num_docs": len(docs),
        "num_queries": len(queries),
        "params": {
            "embedding_model": args.embedding_model,
            "reranker_model": args.reranker_model,
            "rrf_k": args.rrf_k,
            "dense_weight": args.dense_weight,
            "rerank_depth": args.depth,
        },
        "methods": {name: rounded(metrics) for name, metrics in methods.items()},
        "rerank_seconds_per_query": round(rerank_seconds / len(queries), 3),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    suffix = f"_limit{args.limit}" if args.limit else ""
    name = f"rerank_{args.split}_{args.reranker_model.split('/')[-1]}_d{args.depth}{suffix}.json"
    out_file = args.out / name
    out_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'method':<20} {'recall@1':>9} {'mrr@10':>8} {'recall@10':>10} {'recall@20':>10}")
    for method, metrics in methods.items():
        print(
            f"{method:<20} {metrics['recall@1']:>9.4f} {metrics['mrr@10']:>8.4f} "
            f"{metrics['recall@10']:>10.4f} {metrics['recall@20']:>10.4f}"
        )
    print(f"\nrerank cost: {report['rerank_seconds_per_query']} s/query\nsaved {out_file}")


if __name__ == "__main__":
    main()
