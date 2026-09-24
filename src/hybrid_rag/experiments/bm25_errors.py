"""List the queries BM25 fails on, for manual error analysis.

A query "fails" when its relevant paragraph is not in the top `--k` results.
For each failure the report shows the query, the rank of the relevant
paragraph (if it is in the top 100), the relevant paragraph and BM25's top
result, plus which query tokens each of them contains.

Usage:
    python -m hybrid_rag.experiments.bm25_errors --split test --k 10
"""

import argparse
from pathlib import Path

from hybrid_rag.config import get_settings
from hybrid_rag.data.persianqa import load_persianqa
from hybrid_rag.retrieval.bm25 import BM25
from hybrid_rag.text.normalize import tokenize

SNIPPET_CHARS = 400


def snippet(text: str) -> str:
    return text if len(text) <= SNIPPET_CHARS else text[:SNIPPET_CHARS] + " …"


def mark_overlap(query_tokens: list[str], doc_tokens: list[str]) -> str:
    """Query tokens, with the ones missing from the document struck through."""
    present = set(doc_tokens)
    return " ".join(t if t in present else f"~~{t}~~" for t in query_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("results/analysis"))
    args = parser.parse_args()

    dataset = load_persianqa(get_settings().data_dir)
    query_set = dataset.splits[args.split]
    doc_tokens = {doc.id: tokenize(doc.text) for doc in dataset.corpus.values()}
    bm25 = BM25(doc_tokens)

    sections = []
    for query in query_set.queries.values():
        (gold_id,) = query_set.qrels[query.id]
        ranking = [doc_id for doc_id, _ in bm25.search(tokenize(query.text), top_k=100)]
        if gold_id in ranking[: args.k]:
            continue

        gold_rank = ranking.index(gold_id) + 1 if gold_id in ranking else None
        top_id = ranking[0] if ranking else None
        query_tokens = tokenize(query.text)

        lines = [
            f"## {query.id}",
            "",
            f"**Query:** {query.text}",
            "",
            f"**Rank of relevant paragraph:** {gold_rank or '> 100'}",
            "",
            f"**Relevant paragraph** `{gold_id}` ({dataset.corpus[gold_id].title}) "
            f"— query tokens found: {mark_overlap(query_tokens, doc_tokens[gold_id])}",
            "",
            f"> {snippet(dataset.corpus[gold_id].text)}",
            "",
        ]
        if top_id:
            lines += [
                f"**BM25 top result** `{top_id}` ({dataset.corpus[top_id].title}) "
                f"— query tokens found: {mark_overlap(query_tokens, doc_tokens[top_id])}",
                "",
                f"> {snippet(dataset.corpus[top_id].text)}",
                "",
            ]
        lines += ["**Category:** _TODO_", ""]
        sections.append("\n".join(lines))

    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / f"bm25_{args.split}_failures_at_{args.k}.md"
    header = (
        f"# BM25 failures on PersianQA {args.split} (relevant paragraph not in top {args.k})\n\n"
        f"{len(sections)} of {len(query_set.queries)} queries. "
        "Struck-through tokens (~~like this~~) do not occur in that paragraph.\n\n"
    )
    out_file.write_text(header + "\n".join(sections), encoding="utf-8")
    print(f"{len(sections)} failures -> {out_file}")


if __name__ == "__main__":
    main()
