import numpy as np

from hybrid_rag.retrieval.embedding_cache import cache_key, load_or_compute


def test_computes_once_then_reads_from_disk(tmp_path):
    calls = []

    def compute(texts):
        calls.append(texts)
        return np.arange(len(texts) * 2, dtype=np.float32).reshape(len(texts), 2)

    first = load_or_compute(tmp_path, "org/model", "passage", ["a", "b"], compute)
    second = load_or_compute(tmp_path, "org/model", "passage", ["a", "b"], compute)
    assert len(calls) == 1
    np.testing.assert_array_equal(first, second)


def test_key_changes_with_model_kind_and_texts():
    base = cache_key("m", "passage", ["a", "b"])
    assert cache_key("m2", "passage", ["a", "b"]) != base
    assert cache_key("m", "query", ["a", "b"]) != base
    assert cache_key("m", "passage", ["a", "c"]) != base
    assert cache_key("m", "passage", ["ab"]) != cache_key("m", "passage", ["a", "b"])
