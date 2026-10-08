"""Build the set of answers that a human labels to validate the LLM judge.

Samples answerable test questions, generates an answer for each with the real
pipeline (hybrid retrieval + LLM), keeps the first `--n` that were not refused, and writes

- results/labeling/answers.json   the answers (committed together with the human labels)
- data/labeling/label.html        a self-contained Persian page for labelling them
                                   (generated, not committed; it hides any judge output)

Usage (endpoint and model come from the environment, see experiments.generate):
    python -m hybrid_rag.experiments.make_labeling_set --n 50
"""

import argparse
import json
import random
from pathlib import Path

from hybrid_rag.config import get_settings
from hybrid_rag.data.persianqa import load_persianqa
from hybrid_rag.experiments.hybrid import bm25_run, dense_run, fuse
from hybrid_rag.generation.answer import generate_answer, strip_citations
from hybrid_rag.generation.client import OpenAICompatibleClient

PAGE = Path(__file__).with_name("label_page.html").read_text(encoding="utf-8")


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50, help="answers to label")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-k", type=int, default=5)
    parser.add_argument("--rrf-k", type=int, default=1)
    parser.add_argument("--dense-weight", type=float, default=1.5)
    parser.add_argument("--retrieval-depth", type=int, default=100)
    parser.add_argument("--embedding-model", default=settings.embedding_model)
    parser.add_argument("--model", default=settings.llm_model)
    parser.add_argument("--out", type=Path, default=Path("results/labeling"))
    parser.add_argument("--page", type=Path, default=settings.data_dir / "labeling" / "label.html")
    args = parser.parse_args()
    if not args.model:
        parser.error("set HRE_LLM_MODEL or pass --model")

    dataset = load_persianqa(settings.data_dir)
    docs = list(dataset.corpus.values())
    candidates = list(dataset.splits["test"].queries.values())
    sample = random.Random(args.seed).sample(candidates, min(len(candidates), round(args.n * 1.3)))

    bm25 = bm25_run(docs, sample, args.retrieval_depth)
    dense = dense_run(docs, sample, args.retrieval_depth, args.embedding_model)
    hybrid = fuse(bm25, dense, args.rrf_k, args.dense_weight)

    client = OpenAICompatibleClient(settings.llm_base_url, args.model, settings.llm_api_key)
    items, refused = [], 0
    for query in sample:
        if len(items) == args.n:
            break
        passages = [(d, dataset.corpus[d].text) for d in hybrid[query.id][: args.context_k]]
        answer = generate_answer(client, query.text, passages)
        if answer.refused:
            refused += 1
            continue
        items.append(
            {
                "id": query.id,
                "question": query.text,
                "reference_answers": list(query.answers),
                "answer": answer.text,
                "answer_shown": strip_citations(answer.text),
                "citations": answer.citations,
            }
        )
        print(f"{len(items)}/{args.n}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    answers_file = args.out / "answers.json"
    payload = {
        "llm_model": args.model,
        "seed": args.seed,
        "refused_and_skipped": refused,
        "items": items,
    }
    answers_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    args.page.parent.mkdir(parents=True, exist_ok=True)
    shown = [
        {k: x[k] for k in ("id", "question", "reference_answers", "answer_shown")} for x in items
    ]
    args.page.write_text(
        PAGE.replace("__ITEMS__", json.dumps(shown, ensure_ascii=False)), encoding="utf-8"
    )
    print(f"saved {answers_file} ({len(items)} answers, {refused} refusals skipped)")
    print(f"label page: {args.page.resolve()}")


if __name__ == "__main__":
    main()
