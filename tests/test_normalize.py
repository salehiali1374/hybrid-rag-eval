"""Specification for hybrid_rag.text.normalize."""

from hybrid_rag.text.normalize import normalize, tokenize


class TestNormalize:
    def test_arabic_letters_become_persian(self):
        assert normalize("كتاب") == "کتاب"
        assert normalize("علي") == "علی"
        assert normalize("مصطفى") == "مصطفی"

    def test_digits_become_ascii(self):
        assert normalize("سال ۱۴۰۳") == "سال 1403"
        assert normalize("٢٠٢٤") == "2024"

    def test_diacritics_and_tatweel_removed(self):
        assert normalize("کِتابٌ") == "کتاب"
        assert normalize("کـــتاب") == "کتاب"

    def test_zwnj_becomes_space(self):
        assert normalize("کتاب‌ها") == "کتاب ها"

    def test_lowercase_and_whitespace(self):
        assert normalize("  BM25   و  RAG \n") == "bm25 و rag"

    def test_idempotent(self):
        once = normalize("كتاب‌هاي ۱۲")
        assert normalize(once) == once

    def test_alef_madda_becomes_alef(self):
        assert normalize("آشیل") == "اشیل"
        assert tokenize("آشیل") == tokenize("اشیل")


class TestTokenize:
    def test_drops_punctuation(self):
        assert tokenize("سلام، دنیا! «کتاب‌ها» چیست؟") == ["سلام", "دنیا", "کتاب", "ها", "چیست"]

    def test_normalizes_first(self):
        assert tokenize("كتاب ۲") == ["کتاب", "2"]

    def test_empty(self):
        assert tokenize("  ،؟ ") == []
