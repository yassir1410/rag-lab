# System Architecture Guide

The RAG Engineering Lab is constructed around a clean separation of concerns.

## Pipeline Components

The ingestion pipeline transforms raw input into structured entities:

1. **File Validation**: Validates magic bytes, file extensions, and file size limits.
2. **Cryptographic Hashing**: Computes SHA-256 for database duplicate detection.
3. **Format Parser**: Executes the dedicated parser (PDF, DOCX, TXT, MD).
4. **Text Normalization**: Cleans whitespace without mutating source data.

## Code Example

```python
def check_duplicate(sha256_hash: str, db_session) -> bool:
    """Returns True if the content hash exists in the database."""
    return db_session.query(Document).filter_by(sha256_hash=sha256_hash).count() > 0
```

## Supported Parsers

| Format | Library | Page Attribution | Table Support |
| --- | --- | --- | --- |
| PDF | PyMuPDF (fitz) | Yes (1-indexed) | Bounding boxes |
| DOCX | python-docx | Flow-based | Markdown tables |
| TXT | Python standard | Flow-based | N/A |
| Markdown | Python standard | Flow-based | Native syntax |
