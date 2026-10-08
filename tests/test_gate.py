import json
from pathlib import Path

from hybrid_rag.eval.gate import check_gate


def test_passes_when_every_metric_meets_its_minimum():
    assert check_gate({"recall@1": 0.91, "mrr@10": 0.93}, {"recall@1": 0.9, "mrr@10": 0.93}) == []


def test_reports_each_metric_below_its_minimum():
    failures = check_gate(
        {"recall@1": 0.80, "recall@10": 0.99}, {"recall@1": 0.9, "recall@10": 0.97}
    )
    assert len(failures) == 1 and failures[0].startswith("recall@1:")


def test_a_missing_metric_fails():
    assert check_gate({}, {"recall@10": 0.97}) == ["recall@10: not computed"]


def test_committed_thresholds_are_below_the_committed_baseline():
    root = Path(__file__).parent.parent
    config = json.loads((root / "eval_gate.json").read_text(encoding="utf-8"))
    baseline = json.loads(
        (root / "results" / "bm25_test_k1-1.2_b-0.75.json").read_text(encoding="utf-8")
    )["metrics"]
    assert check_gate(baseline, config["min"]) == []
    assert config["split"] == "test"
