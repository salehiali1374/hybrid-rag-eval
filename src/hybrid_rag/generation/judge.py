"""LLM judge for answer correctness.

The judge sees the question, the reference answers from the dataset and the
system's answer (without its [n] citation markers) and replies with one word.
How far it can be trusted is measured separately, against human labels (see
`hybrid_rag.eval.agreement` and `experiments.judge_agreement`).

Only answers that were actually given are judged. Refusals are scored by
looking at the refusal marker, not by a model.
"""

import re

from hybrid_rag.generation.answer import LLMClient, strip_citations

SYSTEM_PROMPT = """You grade answers to questions about Persian text.
You are given a question, one or more reference answers, and a candidate answer.
The candidate is CORRECT if it states the same fact as at least one reference answer.
Different wording, a full sentence instead of a short phrase, or extra correct detail is fine.
The candidate is INCORRECT if it states a different fact, contradicts the references, leaves out
the fact that was asked for, or only says that the information is missing.
Judge against the reference answers only; do not use outside knowledge to overrule them.
Reply with exactly one word: CORRECT or INCORRECT."""

_INCORRECT = re.compile(r"\bINCORRECT\b", re.IGNORECASE)
_CORRECT = re.compile(r"\bCORRECT\b", re.IGNORECASE)


def build_messages(
    question: str, reference_answers: tuple[str, ...] | list[str], answer: str
) -> list[dict[str, str]]:
    references = " | ".join(reference_answers)
    user = (
        f"Question: {question}\n"
        f"Reference answer(s): {references}\n"
        f"Candidate answer: {strip_citations(answer)}"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def parse_verdict(raw: str) -> bool | None:
    """True for CORRECT, False for INCORRECT, None if the reply is neither.

    INCORRECT is tested first because the word CORRECT does not match inside it
    (word boundaries), but a reply naming both must not count as correct.
    """
    if _INCORRECT.search(raw):
        return False
    if _CORRECT.search(raw):
        return True
    return None


def judge_answer(
    client: LLMClient, question: str, reference_answers: tuple[str, ...] | list[str], answer: str
) -> bool | None:
    """Ask `client` whether `answer` is correct; None if its reply cannot be understood."""
    return parse_verdict(client.complete(build_messages(question, reference_answers, answer)))
