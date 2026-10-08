"""Answer generation with citations and refusal.

The LLM sees the retrieved passages numbered [1], [2], ... and must answer only
from them, citing the passage numbers it used. Short numbers are used in the
prompt instead of document ids because models copy them far more reliably; this
module maps them back to the real ids afterwards and rejects numbers that do not
exist (a hallucinated citation).

If the passages do not contain the answer, the model must reply with the single
marker `NO_ANSWER`, which `parse_answer` turns into `Answer.refused`.

As with `Encoder` and `Reranker`, the LLM is injected: any object with a
`complete(messages) -> str` method works, so the tests need no network.
"""

import re
from dataclasses import dataclass, field
from typing import Protocol

REFUSAL_MARKER = "NO_ANSWER"

SYSTEM_PROMPT = f"""You answer questions about Persian text using ONLY the numbered passages given.
Rules:
1. Answer in Persian, briefly, using only facts stated in the passages.
2. After every claim, cite the passage number(s) you used, like [1] or [1][3].
3. Never cite a number that is not in the list of passages.
4. Answer only if a passage states the answer directly and explicitly. If the passages merely
   mention the topic, are related, or give only part of what was asked (for example a name is
   missing, or a list is not given), reply with exactly {REFUSAL_MARKER} and nothing else.
5. Never write a sentence saying the information is missing: use {REFUSAL_MARKER} instead.
Do not use any outside knowledge."""

# "[1]", "[12]", also Persian/Arabic-Indic digits: "[۱]". Several numbers in one pair of
# brackets ("[1, 3]" or "[1،3]") are accepted too.
_CITATION = re.compile(r"\[\s*(\d+(?:\s*[,،]\s*\d+)*)\s*\]")
_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


class LLMClient(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str: ...


@dataclass(frozen=True)
class Answer:
    text: str
    refused: bool = False
    citations: list[str] = field(default_factory=list)  # real doc ids, in order of first use
    invalid_citations: list[int] = field(default_factory=list)  # numbers outside 1..n


def build_messages(question: str, passages: list[tuple[str, str]]) -> list[dict[str, str]]:
    """Chat messages for `passages`, a list of (doc_id, text) in ranked order."""
    numbered = "\n\n".join(f"[{i}] {text}" for i, (_, text) in enumerate(passages, start=1))
    user = f"Passages:\n\n{numbered}\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def parse_answer(raw: str, doc_ids: list[str]) -> Answer:
    """Turn the model's raw reply into an `Answer`; `doc_ids[i - 1]` is passage [i]."""
    text = raw.strip()
    if REFUSAL_MARKER in text:
        return Answer(text="", refused=True)

    citations: list[str] = []
    invalid: list[int] = []
    for match in _CITATION.finditer(text.translate(_DIGITS)):
        for number in (int(part) for part in re.split(r"[,،]", match.group(1))):
            if not 1 <= number <= len(doc_ids):
                if number not in invalid:
                    invalid.append(number)
            elif doc_ids[number - 1] not in citations:
                citations.append(doc_ids[number - 1])
    return Answer(text=text, citations=citations, invalid_citations=invalid)


def strip_citations(text: str) -> str:
    """`text` without its [n] markers, e.g. for showing the answer to a judge."""
    return re.sub(r"\s*" + _CITATION.pattern, "", text.translate(_DIGITS)).strip()


def generate_answer(client: LLMClient, question: str, passages: list[tuple[str, str]]) -> Answer:
    """Ask `client` to answer `question` from `passages` and check its reply."""
    reply = client.complete(build_messages(question, passages))
    return parse_answer(reply, [doc_id for doc_id, _ in passages])
