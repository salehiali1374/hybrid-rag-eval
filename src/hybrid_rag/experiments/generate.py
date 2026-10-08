"""Check citations and refusal of the answer generator on PersianQA.

Takes a random sample of answerable and unanswerable test questions, retrieves the
top passages with the hybrid retriever (BM25 + dense + RRF), asks an LLM to answer
from them, and reports:

- answerable questions: how often it answers, cites the gold passage, cites nothing
  or cites a number that does not exist
- unanswerable questions: how often it correctly refuses

This does NOT grade whether the answer text is correct; that is milestone 7.

The LLM endpoint comes from the environment, never from the code:

    export HRE_LLM_BASE_URL=http://<host>:<port>/v1
    export HRE_LLM_MODEL=<model name as listed by the server>
    python -m hybrid_rag.experiments.generate --n 20

Calls are sequential on purpose: a shared GPU should not get a burst of requests.
"""

import argparse
import json
import random
import time
from pathlib import Path

from hybrid_rag.config import get_settings
from hybrid_rag.data.persianqa import load_persianqa
from hybrid_rag.experiments.hybrid import bm25_run, dense_run, fuse
from hybrid_rag.generation.answer import generate_answer
from hybrid_rag.generation.client import OpenAICompatibleClient


def rate(count: int, total: int) -> float | None:
    return round(count / total, 4) if total else None


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--n", type=int, default=20, help="questions of EACH kind")
    parser.add_argument("--context-k", type=int, default=5, help="passages shown to the LLM")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="", help="suffix of the result file name")
    parser.add_argument("--rrf-k", type=int, default=1)
    parser.add_argument("--dense-weight", type=float, default=1.5)
    parser.add_argument("--retrieval-depth", type=int, default=100)
    parser.add_argument("--embedding-model", default=settings.embedding_model)
    parser.add_argument("--model", default=settings.llm_model)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    if not args.model:
        parser.error("set HRE_LLM_MODEL or pass --model")

    dataset = load_persianqa(settings.data_dir)
    query_set = dataset.splits[args.split]
    docs = list(dataset.corpus.values())
    rng = random.Random(args.seed)
    answerable = rng.sample(list(query_set.queries.values()), args.n)
    unanswerable = rng.sample(
        list(query_set.unanswerable.values()), min(args.n, len(query_set.unanswerable))
    )
    kinds = {q.id: "answerable" for q in answerable} | {q.id: "unanswerable" for q in unanswerable}
    queries = answerable + unanswerable

    bm25 = bm25_run(docs, queries, args.retrieval_depth)
    dense = dense_run(docs, queries, args.retrieval_depth, args.embedding_model)
    hybrid = fuse(bm25, dense, args.rrf_k, args.dense_weight)

    client = OpenAICompatibleClient(settings.llm_base_url, args.model, settings.llm_api_key)
    rows = []
    for i, query in enumerate(queries, start=1):
        doc_ids = hybrid[query.id][: args.context_k]
        passages = [(doc_id, dataset.corpus[doc_id].text) for doc_id in doc_ids]
        start = time.perf_counter()
        answer = generate_answer(client, query.text, passages)
        seconds = time.perf_counter() - start
        gold = set(query_set.qrels.get(query.id, {}))
        rows.append(
            {
                "id": query.id,
                "kind": kinds[query.id],
                "question": query.text,
                "answer": answer.text,
                "refused": answer.refused,
                "citations": answer.citations,
                "invalid_citations": answer.invalid_citations,
                "gold_in_context": bool(gold & set(doc_ids)),
                "cites_gold": bool(gold & set(answer.citations)),
                "seconds": round(seconds, 2),
            }
        )
        print(f"{i}/{len(queries)} {kinds[query.id]:<12} refused={answer.refused}", flush=True)

    ans = [r for r in rows if r["kind"] == "answerable"]
    answered = [r for r in ans if not r["refused"]]
    unans = [r for r in rows if r["kind"] == "unanswerable"]
    summary = {
        "answerable": {
            "n": len(ans),
            "gold_in_context": rate(sum(r["gold_in_context"] for r in ans), len(ans)),
            "refused": rate(sum(r["refused"] for r in ans), len(ans)),
            "answered_cites_gold": rate(sum(r["cites_gold"] for r in answered), len(answered)),
            "answered_without_citation": rate(
                sum(not r["citations"] for r in answered), len(answered)
            ),
            "answered_with_invalid_citation": rate(
                sum(bool(r["invalid_citations"]) for r in answered), len(answered)
            ),
        },
        "unanswerable": {
            "n": len(unans),
            "refused": rate(sum(r["refused"] for r in unans), len(unans)),
            "answered": rate(sum(not r["refused"] for r in unans), len(unans)),
        },
        "seconds_per_question": round(sum(r["seconds"] for r in rows) / len(rows), 2),
    }
    report = {
        "dataset": dataset.name,
        "split": args.split,
        "params": {
            "llm_model": args.model,
            "context_k": args.context_k,
            "seed": args.seed,
            "rrf_k": args.rrf_k,
            "dense_weight": args.dense_weight,
        },
        "summary": summary,
        "rows": rows,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / f"generate_{args.split}_n{args.n}_seed{args.seed}{args.tag}.json"
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nsaved {out_file}")


if __name__ == "__main__":
    main()
