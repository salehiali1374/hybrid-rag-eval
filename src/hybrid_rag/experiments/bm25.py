"""Evaluate BM25 on PersianQA.

Usage:
    python -m hybrid_rag.experiments.bm25 --split test
    python -m hybrid_rag.experiments.bm25 --split train --k1 0.9 --b 0.4
"""

import argparse
import json
import time
from pathlib import Path

from hybrid_rag.config import get_settings
from hybrid_rag.data.persianqa import load_persianqa
from hybrid_rag.eval.metrics import evaluate_run
from hybrid_rag.retrieval.bm25 import BM25
from hybrid_rag.text.normalize import tokenize

K_VALUES = (1, 5, 10, 20)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    dataset = load_persianqa(get_settings().data_dir)
    query_set = dataset.splits[args.split]

    start = time.perf_counter()
    bm25 = BM25({doc.id: tokenize(doc.text) for doc in dataset.corpus.values()}, args.k1, args.b)
    index_seconds = time.perf_counter() - start

    start = time.perf_counter()
    run = {
        query.id: [doc_id for doc_id, _ in bm25.search(tokenize(query.text), top_k=max(K_VALUES))]
        for query in query_set.queries.values()
    }
    search_seconds = time.perf_counter() - start

    metrics = evaluate_run(run, query_set.qrels, K_VALUES)
    report = {
        "method": "bm25",
        "dataset": dataset.name,
        "split": args.split,
        "params": {"k1": args.k1, "b": args.b},
        "num_docs": len(dataset.corpus),
        "num_queries": len(query_set.queries),
        "metrics": {name: round(value, 4) for name, value in metrics.items()},
        "index_seconds": round(index_seconds, 2),
        "ms_per_query": round(1000 * search_seconds / len(query_set.queries), 2),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / f"bm25_{args.split}_k1-{args.k1}_b-{args.b}.json"
    out_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"{'metric':<10} value")
    for name, value in report["metrics"].items():
        print(f"{name:<10} {value:.4f}")
    print(f"\nsaved {out_file}  ({report['ms_per_query']} ms/query)")


if __name__ == "__main__":
    main()
