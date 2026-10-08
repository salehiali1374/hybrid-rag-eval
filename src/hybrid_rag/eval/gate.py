"""CI quality gate: fail if BM25 retrieval quality drops below the thresholds in eval_gate.json.

Usage:
    python -m hybrid_rag.eval.gate
"""

import argparse
import json
import sys
from pathlib import Path

from hybrid_rag.config import get_settings
from hybrid_rag.data.persianqa import load_persianqa
from hybrid_rag.eval.metrics import evaluate_run
from hybrid_rag.retrieval.bm25 import BM25
from hybrid_rag.text.normalize import tokenize


def check_gate(metrics: dict[str, float], minimums: dict[str, float]) -> list[str]:
    """One message per metric that is missing or below its minimum; empty if the gate passes."""
    failures = []
    for name, minimum in minimums.items():
        if name not in metrics:
            failures.append(f"{name}: not computed")
        elif metrics[name] < minimum:
            failures.append(f"{name}: {metrics[name]:.4f} is below the minimum {minimum:.4f}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("eval_gate.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))

    dataset = load_persianqa(get_settings().data_dir)
    query_set = dataset.splits[config["split"]]
    bm25 = BM25({doc.id: tokenize(doc.text) for doc in dataset.corpus.values()})
    run = {
        query.id: [doc_id for doc_id, _ in bm25.search(tokenize(query.text), top_k=10)]
        for query in query_set.queries.values()
    }
    metrics = evaluate_run(run, query_set.qrels, (1, 10))

    print(f"{'metric':<10} {'value':>8} {'minimum':>8}")
    for name, minimum in config["min"].items():
        print(f"{name:<10} {metrics.get(name, float('nan')):>8.4f} {minimum:>8.4f}")
    failures = check_gate(metrics, config["min"])
    if failures:
        print("\nGATE FAILED:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("\ngate passed")


if __name__ == "__main__":
    main()
