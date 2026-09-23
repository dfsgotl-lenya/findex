from findex.tokenize import tokenize


def test_mixed_case_is_casefolded() -> None:
    assert list(tokenize("Python PYTHON PyThOn")) == ["python", "python", "python"]


def test_cyrillic_text_is_supported() -> None:
    assert list(tokenize("Привіт, Україно!")) == ["привіт", "україно"]


def test_combining_mark_is_nfc_normalized() -> None:
    assert list(tokenize("cafe\u0301")) == ["café"]


def test_internal_apostrophe_is_preserved() -> None:
    assert list(tokenize("don't п'ять")) == ["don't", "п'ять"]


def test_hyphen_is_a_separator() -> None:
    assert list(tokenize("state-of-the-art")) == ["state", "of", "the", "art"]


def test_digits_are_kept() -> None:
    assert list(tokenize("Python3 2026")) == ["python3", "2026"]


def test_punctuation_is_removed() -> None:
    assert list(tokenize("Hello, world!!!")) == ["hello", "world"]


def test_empty_text_yields_nothing() -> None:
    assert list(tokenize("")) == []
