"""Persian text normalization and tokenization for lexical retrieval.

The same function must be applied to documents and queries: if the corpus
writes "كتاب" (Arabic kaf) and the user types "کتاب" (Persian kaf), BM25 sees
two unrelated words unless both sides are normalized identically.
"""


def normalize(text: str) -> str:
    """Return a canonical form of `text`.

    Rules, applied in this order of importance:
    1. Arabic letters -> Persian: "ك" -> "ک", "ي" and "ى" -> "ی".
    2. Persian (۰-۹) and Arabic-Indic (٠-٩) digits -> ASCII 0-9.
    3. Remove Arabic diacritics (U+064B to U+0652) and tatweel "ـ" (U+0640).
    4. Replace the zero-width non-joiner (U+200C) with a space, so that
       "کتاب‌ها" becomes "کتاب ها" and matches queries containing "کتاب".
    5. Lowercase (for Latin text) and collapse runs of whitespace into one space,
       stripping leading/trailing whitespace.
    """
    raise NotImplementedError


def tokenize(text: str) -> list[str]:
    """Normalize `text`, then split it into word tokens.

    A token is a maximal run of Unicode word characters (letters, digits,
    underscore), so punctuation such as "،" "؟" "!" "." "«" "»" is dropped.
    """
    raise NotImplementedError
