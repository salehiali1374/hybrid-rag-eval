import pytest

from hybrid_rag.data.persianqa import load_persianqa, parse

RAW = {
    "test": {
        "data": [
            {
                "title": "رئال مادرید",
                "paragraphs": [
                    {
                        "context": "باشگاه فوتبال رئال مادرید در مادرید قرار دارد.",
                        "qas": [
                            {"id": 1, "question": "پایتخت اسپانیا کجاست؟", "is_impossible": False},
                            {"id": 2, "question": "مربی تیم کیست؟", "is_impossible": True},
                        ],
                    }
                ],
            }
        ]
    },
    "train": {
        "data": [
            {
                "title": "تکراری",
                "paragraphs": [
                    {
                        "context": "باشگاه فوتبال رئال مادرید در مادرید قرار دارد.",
                        "qas": [{"id": 3, "question": "رئال کجاست؟", "is_impossible": False}],
                    }
                ],
            }
        ]
    },
}


def test_parse_builds_corpus_qrels_and_unanswerable():
    ds = parse(RAW)
    assert len(ds.corpus) == 1  # identical paragraphs across splits are stored once
    doc_id = next(iter(ds.corpus))
    assert ds.splits["test"].qrels == {"test-1": {doc_id: 1}}
    assert set(ds.splits["test"].unanswerable) == {"test-2"}
    assert ds.splits["train"].qrels == {"train-3": {doc_id: 1}}


@pytest.mark.network
def test_real_download(tmp_path):
    ds = load_persianqa(tmp_path)
    assert len(ds.splits["test"].queries) > 500
    assert len(ds.splits["test"].unanswerable) > 100
