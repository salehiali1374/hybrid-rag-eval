"""Persian text normalization and tokenization for lexical retrieval.

The same function must be applied to documents and queries: if the corpus
writes "كتاب" (Arabic kaf) and the user types "کتاب" (Persian kaf), BM25 sees
two unrelated words unless both sides are normalized identically.
"""

import re

_TRANSLATION_TABLE = str.maketrans(
    {
        "ك": "ک",
        "ي": "ی",
        "ى": "ی",
        "۰": "0",
        "۱": "1",
        "۲": "2",
        "۳": "3",
        "۴": "4",
        "۵": "5",
        "۶": "6",
        "۷": "7",
        "۸": "8",
        "۹": "9",
        "٠": "0",
        "١": "1",
        "٢": "2",
        "٣": "3",
        "٤": "4",
        "٥": "5",
        "٦": "6",
        "٧": "7",
        "٨": "8",
        "٩": "9",
        "ـ": None,
        "\u200c": " ",
    }
)


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

    text = text.translate(_TRANSLATION_TABLE)

    text = re.sub(r"[\u064B-\u0652]", "", text)

    text = re.sub(r"\s+", " ", text.lower()).strip()

    return text


def tokenize(text: str) -> list[str]:
    """Normalize `text`, then split it into word tokens.

    A token is a maximal run of Unicode word characters (letters, digits,
    underscore), so punctuation such as "،" "؟" "!" "." "«" "»" is dropped.
    """
    text = normalize(text)

    return re.findall(r"\w+", text, flags=re.UNICODE)
