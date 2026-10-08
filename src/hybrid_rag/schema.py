"""Core data types shared by every stage of the pipeline.

Conventions (same as TREC / BEIR):
- A *run* maps each query id to its ranked list of document ids (best first).
- *Qrels* map each query id to {doc_id: graded relevance}; relevance > 0 means relevant.
"""

from dataclasses import dataclass, field

Run = dict[str, list[str]]
Qrels = dict[str, dict[str, int]]


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    title: str = ""


@dataclass(frozen=True)
class Query:
    id: str
    text: str
    answers: tuple[str, ...] = ()  # reference answers (distinct, in file order); () if unknown


@dataclass
class QuerySet:
    """Queries of one split.

    `qrels` covers only answerable queries. `unanswerable` holds queries whose
    answer is not in the corpus: they have no relevant documents, so they are
    left out of retrieval metrics and used later to test refusal behaviour.
    """

    queries: dict[str, Query] = field(default_factory=dict)
    qrels: Qrels = field(default_factory=dict)
    unanswerable: dict[str, Query] = field(default_factory=dict)


@dataclass
class RetrievalDataset:
    name: str
    corpus: dict[str, Document]
    splits: dict[str, QuerySet]
