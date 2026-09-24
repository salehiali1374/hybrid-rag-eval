"""Evaluate dense retrieval on PersianQA.

Needs the `dense` extra (sentence-transformers). The first run downloads the
model and embeds the corpus (about 2 minutes on a laptop CPU); later runs read
the embeddings from data/embeddings/.

Usage:
    python -m hybrid_rag.experiments.dense --split test
"""

import argparse
import json
import time
from pathlib import Path

from hybrid_rag.config import get_settings
from hybrid_rag.data.persianqa import load_persianqa
from hybrid_rag.eval.metrics import evaluate_run
from hybrid_rag.retrieval.dense import DenseIndex, Encoder
from hybrid_rag.retrieval.embedding_cache import load_or_compute

K_VALUES = (1, 5, 10, 20)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--model", default=get_settings().embedding_model)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer  # heavy import, only needed here

    settings = get_settings()
    cache_dir = settings.data_dir / "embeddings"
    dataset = load_persianqa(settings.data_dir)
    query_set = dataset.splits[args.split]

    encoder = Encoder(SentenceTransformer(args.model, device="cpu"))

    docs = list(dataset.corpus.values())
    doc_embeddings = load_or_compute(
        cache_dir, args.model, "passage", [d.text for d in docs], encoder.encode_passages
    )
    index = DenseIndex([d.id for d in docs], doc_embeddings)

    queries = list(query_set.queries.values())
    start = time.perf_counter()
    query_embeddings = encoder.encode_queries([q.text for q in queries])
    encode_seconds = time.perf_counter() - start

    start = time.perf_counter()
    results = index.search_batch(query_embeddings, top_k=max(K_VALUES))
    search_seconds = time.perf_counter() - start

    run = {q.id: [doc_id for doc_id, _ in hits] for q, hits in zip(queries, results, strict=True)}
    metrics = evaluate_run(run, query_set.qrels, K_VALUES)
    report = {
        "method": "dense",
        "dataset": dataset.name,
        "split": args.split,
        "params": {"model": args.model},
        "num_docs": len(docs),
        "num_queries": len(queries),
        "metrics": {name: round(value, 4) for name, value in metrics.items()},
        "query_encode_ms_per_query": round(1000 * encode_seconds / len(queries), 2),
        "search_ms_per_query": round(1000 * search_seconds / len(queries), 3),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / f"dense_{args.split}_{args.model.split('/')[-1]}.json"
    out_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"{'metric':<10} value")
    for name, value in report["metrics"].items():
        print(f"{name:<10} {value:.4f}")
    print(f"\nsaved {out_file}")


if __name__ == "__main__":
    main()
