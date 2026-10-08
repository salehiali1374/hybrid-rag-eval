"""How well does the LLM judge agree with the human labels?

Runs the judge on the answers in results/labeling/answers.json and compares its
verdicts with results/labeling/human_labels.json (the file downloaded from the
labelling page). Answers the human marked "unsure" and replies the judge did not
answer clearly are left out of the comparison and counted separately.

Usage (endpoint and model come from the environment, see experiments.generate):
    python -m hybrid_rag.experiments.judge_agreement
"""

import argparse
import json
from pathlib import Path

from hybrid_rag.config import get_settings
from hybrid_rag.eval.agreement import agreement
from hybrid_rag.generation.client import OpenAICompatibleClient
from hybrid_rag.generation.judge import judge_answer


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, default=Path("results/labeling/answers.json"))
    parser.add_argument("--labels", type=Path, default=Path("results/labeling/human_labels.json"))
    parser.add_argument("--model", default=settings.llm_model)
    parser.add_argument("--out", type=Path, default=Path("results/judge_agreement.json"))
    args = parser.parse_args()
    if not args.model:
        parser.error("set HRE_LLM_MODEL or pass --model")

    items = json.loads(args.answers.read_text(encoding="utf-8"))["items"]
    human = json.loads(args.labels.read_text(encoding="utf-8"))
    client = OpenAICompatibleClient(settings.llm_base_url, args.model, settings.llm_api_key)

    rows, unsure, unlabeled, unclear = [], 0, 0, 0
    for item in items:
        label = human.get(item["id"])
        if label is None:
            unlabeled += 1
            continue
        if label == "unsure":
            unsure += 1
            continue
        verdict = judge_answer(client, item["question"], item["reference_answers"], item["answer"])
        if verdict is None:
            unclear += 1
            continue
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "reference_answers": item["reference_answers"],
                "answer": item["answer"],
                "human_correct": label == "correct",
                "judge_correct": verdict,
            }
        )

    result = agreement([r["human_correct"] for r in rows], [r["judge_correct"] for r in rows])
    disagreements = [r for r in rows if r["human_correct"] != r["judge_correct"]]
    report = {
        "judge_model": args.model,
        "compared": result,
        "human_correct_rate": round(sum(r["human_correct"] for r in rows) / len(rows), 4),
        "left_out": {"unsure": unsure, "unlabeled": unlabeled, "judge_unclear": unclear},
        "disagreements": disagreements,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({k: v for k, v in report.items() if k != "disagreements"}, indent=2))
    for r in disagreements:
        who = (
            "judge says correct, human says incorrect"
            if r["judge_correct"]
            else "judge says incorrect, human says correct"
        )
        print(f"\n[{who}]\nQ: {r['question']}\nref: {r['reference_answers']}\nA: {r['answer']}")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
