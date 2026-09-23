"""PersianQA loader, turned into a retrieval benchmark.

PersianQA (https://github.com/sajjjadayobi/PersianQA) is a SQuAD-style reading
comprehension dataset built on Persian Wikipedia. Each question belongs to one
paragraph. We reuse it for retrieval:

- corpus  = every unique paragraph from the train and test files
- qrels   = question -> the paragraph it was written for (relevance 1)
- unanswerable questions (`is_impossible`) are kept separately for refusal tests
"""

import json
from pathlib import Path

import httpx

from hybrid_rag.schema import Document, Query, QuerySet, RetrievalDataset

BASE_URL = "https://raw.githubusercontent.com/sajjjadayobi/PersianQA/main/dataset/"
FILES = {"train": "pqa_train.json", "test": "pqa_test.json"}


def download(data_dir: Path) -> dict[str, Path]:
    """Download the raw JSON files once and return their local paths."""
    raw_dir = data_dir / "raw" / "persianqa"
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, name in FILES.items():
        path = raw_dir / name
        if not path.exists():
            response = httpx.get(BASE_URL + name, timeout=60, follow_redirects=True)
            response.raise_for_status()
            path.write_bytes(response.content)
        paths[split] = path
    return paths


def parse(raw_by_split: dict[str, dict]) -> RetrievalDataset:
    """Build the retrieval dataset from already-loaded SQuAD-style JSON objects."""
    corpus: dict[str, Document] = {}
    doc_id_by_text: dict[str, str] = {}
    splits: dict[str, QuerySet] = {}

    for split, raw in raw_by_split.items():
        query_set = QuerySet()
        for article_idx, article in enumerate(raw["data"]):
            for para_idx, paragraph in enumerate(article["paragraphs"]):
                context = paragraph["context"].strip()
                doc_id = doc_id_by_text.get(context)
                if doc_id is None:
                    doc_id = f"{split}-{article_idx}-{para_idx}"
                    doc_id_by_text[context] = doc_id
                    corpus[doc_id] = Document(id=doc_id, text=context, title=article["title"])

                for qa in paragraph["qas"]:
                    query = Query(id=f"{split}-{qa['id']}", text=qa["question"].strip())
                    if qa.get("is_impossible", False):
                        query_set.unanswerable[query.id] = query
                    else:
                        query_set.queries[query.id] = query
                        query_set.qrels[query.id] = {doc_id: 1}
        splits[split] = query_set

    return RetrievalDataset(name="persianqa", corpus=corpus, splits=splits)


def load_persianqa(data_dir: Path) -> RetrievalDataset:
    paths = download(data_dir)
    raw = {split: json.loads(path.read_text(encoding="utf-8")) for split, path in paths.items()}
    return parse(raw)
