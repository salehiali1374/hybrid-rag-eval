"""Does the LLM judge reject wrong answers?

The human-labelled answers are almost all correct, so they say little about whether the judge
catches wrong ones. Here wrong answers are made by construction: a question is shown together
with the answer to a *different* question. These are easy negatives, so a high rejection rate
is necessary but not sufficient evidence that the judge is reliable.

Usage (endpoint and model come from the environment, see experiments.generate):
    python -m hybrid_rag.experiments.judge_negatives --n 25
"""

import argparse
import json
import random
from pathlib import Path

from hybrid_rag.config import get_settings
from hybrid_rag.generation.client import OpenAICompatibleClient
from hybrid_rag.generation.judge import judge_answer


def make_negatives(items: list[dict], correct_ids: set[str], n: int, seed: int) -> list[dict]:
    """Up to `n` (question, wrong answer) pairs built from the items whose id is in `correct_ids`.

    The answer comes from another item. Donors whose answer contains one of the question's
    reference answers are skipped, so every pair is wrong by construction.
    """
    rng = random.Random(seed)
    pool = [item for item in items if item["id"] in correct_ids]
    negatives = []
    for item in rng.sample(pool, min(n, len(pool))):
        donors = [
            d
            for d in pool
            if d["id"] != item["id"]
            and not any(ref in d["answer"] for ref in item["reference_answers"])
        ]
        if not donors:
            continue
        donor = rng.choice(donors)
        negatives.append(
            {
                "id": f"{item['id']}~{donor['id']}",
                "question": item["question"],
                "reference_answers": item["reference_answers"],
                "answer": donor["answer"],
            }
        )
    return negatives


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, default=Path("results/labeling/answers.json"))
    parser.add_argument("--labels", type=Path, default=Path("results/labeling/human_labels.json"))
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=settings.llm_model)
    parser.add_argument("--out", type=Path, default=Path("results/judge_negatives.json"))
    args = parser.parse_args()
    if not args.model:
        parser.error("set HRE_LLM_MODEL or pass --model")

    items = json.loads(args.answers.read_text(encoding="utf-8"))["items"]
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    correct_ids = {i for i, label in labels.items() if label == "correct"}
    negatives = make_negatives(items, correct_ids, args.n, args.seed)

    client = OpenAICompatibleClient(settings.llm_base_url, args.model, settings.llm_api_key)
    rejected, wrongly_accepted, unclear = 0, [], 0
    for pair in negatives:
        verdict = judge_answer(client, pair["question"], pair["reference_answers"], pair["answer"])
        if verdict is None:
            unclear += 1
        elif verdict:
            wrongly_accepted.append(pair)
        else:
            rejected += 1

    report = {
        "judge_model": args.model,
        "kind": "synthetic: the answer to a different question",
        "n": len(negatives),
        "rejected": rejected,
        "wrongly_accepted": len(wrongly_accepted),
        "judge_unclear": unclear,
        "rejection_rate": round(rejected / len(negatives), 4),
        "wrongly_accepted_pairs": wrongly_accepted,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "wrongly_accepted_pairs"}, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
