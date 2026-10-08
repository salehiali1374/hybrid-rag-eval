"""Agreement between two binary raters, for checking an LLM judge against humans.

"Positive" means the answer was rated correct.
"""

from typing import TypedDict


class Agreement(TypedDict):
    n: int
    accuracy: float
    kappa: float | None
    tp: int  # both say correct
    fp: int  # judge says correct, human says incorrect
    fn: int  # judge says incorrect, human says correct
    tn: int  # both say incorrect


def agreement(human: list[bool], judge: list[bool]) -> Agreement:
    """Accuracy, Cohen's kappa and the confusion counts of `judge` against `human`.

    Kappa is (po - pe) / (1 - pe): po is the observed agreement (= accuracy) and pe the
    agreement expected if both raters labelled at random with their own label frequencies.
    It is None when pe == 1, i.e. both raters used a single, identical label throughout.
    Raise ValueError if the lists differ in length or are empty.
    """
    if len(human) != len(judge):
        raise ValueError(f"{len(human)} human labels but {len(judge)} judge labels")
    if not human:
        raise ValueError("no labels to compare")

    n = len(human)
    tp = sum(h and j for h, j in zip(human, judge, strict=True))
    fp = sum(not h and j for h, j in zip(human, judge, strict=True))
    fn = sum(h and not j for h, j in zip(human, judge, strict=True))
    tn = n - tp - fp - fn

    po = (tp + tn) / n
    human_pos, judge_pos = (tp + fn) / n, (tp + fp) / n
    pe = human_pos * judge_pos + (1 - human_pos) * (1 - judge_pos)
    kappa = None if pe == 1 else (po - pe) / (1 - pe)
    return {
        "n": n,
        "accuracy": round(po, 4),
        "kappa": None if kappa is None else round(kappa, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
