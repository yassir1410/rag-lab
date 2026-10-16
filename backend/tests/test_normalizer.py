from app.rag.ingestion.cleaning.normalizer import TextNormalizer


def test_normalizer_whitespace_and_newlines():
    normalizer = TextNormalizer()
    raw = "Line 1   with    multiple    spaces.\r\n\r\n\r\n\r\nLine 2 after excessive breaks.\t\n"
    normalized = normalizer.normalize(raw)

    assert "Line 1 with multiple spaces." in normalized
    assert "Line 2 after excessive breaks." in normalized
    assert "\r" not in normalized
    # Excessive breaks collapsed to 2 newlines
    assert "\n\n\n" not in normalized


def test_normalizer_preserves_code_indentation():
    normalizer = TextNormalizer()
    raw = """Here is python code:

```python
def foo():
    x = 10
    if x > 5:
        return True
```
"""
    normalized = normalizer.normalize(raw)
    assert "    x = 10" in normalized
    assert "        return True" in normalized


def test_normalizer_comparison_stats():
    normalizer = TextNormalizer()
    raw = "Multiple      spaces   reduce       character count."
    norm, stats = normalizer.compare(raw)

    assert stats["raw_character_count"] > stats["normalized_character_count"]
    assert stats["raw_word_count"] == 5
    assert stats["normalized_word_count"] == 5
    assert stats["character_reduction_pct"] > 0
