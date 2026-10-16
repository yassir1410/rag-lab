import pytest
from app.rag.ingestion.validation.file_validator import FileValidator
from app.rag.ingestion.exceptions import (
    FileTooLargeError,
    FileValidationError,
    UnsupportedFileTypeError,
)


def test_validator_sanitize_filename():
    validator = FileValidator()
    # Path traversal attempt
    sanitized = validator.sanitize_filename("../../../etc/passwd.txt")
    assert "/" not in sanitized
    assert ".." not in sanitized
    assert sanitized == "passwd.txt"

    # Weird characters
    sanitized = validator.sanitize_filename("my report; rm -rf /.pdf")
    assert ";" not in sanitized
    assert "/" not in sanitized
    assert sanitized.endswith(".pdf")


def test_validator_unsupported_extension():
    validator = FileValidator()
    with pytest.raises(UnsupportedFileTypeError):
        validator.validate(b"binary data", "malicious.exe", "application/x-msdownload")


def test_validator_size_limit():
    # Set limit to 1 KB for test
    validator = FileValidator(max_size_bytes=1024)
    oversized = b"x" * 2048
    with pytest.raises(FileTooLargeError):
        validator.validate(oversized, "big.txt", "text/plain")


def test_validator_pdf_magic_bytes():
    validator = FileValidator()
    # Faked PDF extension with text content
    with pytest.raises(FileValidationError):
        validator.validate(b"This is not a real PDF", "fake.pdf", "application/pdf")

    # Valid signature
    mime, name = validator.validate(b"%PDF-1.4\nvalid pdf content", "real.pdf", "application/pdf")
    assert mime == "application/pdf"
    assert name == "real.pdf"


def test_validator_null_bytes_in_text():
    validator = FileValidator()
    with pytest.raises(FileValidationError):
        validator.validate(b"text with \x00 null bytes", "file.txt", "text/plain")
