import hashlib
import io
import re
import uuid
import zipfile
from pathlib import Path
from typing import Tuple
from app.rag.ingestion.exceptions import (
    FileTooLargeError,
    FileValidationError,
    UnsupportedFileTypeError,
)


class FileValidator:
    """Validates uploaded files for size, integrity, signature (magic bytes), and path safety."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown"}
    SUPPORTED_MIMES = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
        "text/markdown",
        "text/x-markdown",
        "application/octet-stream",  # often sent by browsers for .md or .docx
    }

    def __init__(self, max_size_bytes: int = 15 * 1024 * 1024):
        self.max_size_bytes = max_size_bytes

    def sanitize_filename(self, filename: str) -> str:
        """Strip path traversal, directory separators, null bytes, and unsafe characters."""
        if not filename:
            return "document"
        # Extract basename only to prevent directory traversal
        clean_name = Path(filename).name
        # Remove null bytes
        clean_name = clean_name.replace("\x00", "")
        # Remove non-alphanumeric except dots, dashes, underscores
        clean_name = re.sub(r"[^\w\.\- ]", "_", clean_name).strip()
        if not clean_name:
            clean_name = "document"
        return clean_name

    def generate_storage_filename(self, sanitized_filename: str) -> str:
        """Generate a collision-free UUID storage name while preserving sanitized extension."""
        ext = Path(sanitized_filename).suffix.lower()
        unique_id = uuid.uuid4().hex
        return f"{unique_id}{ext}"

    def compute_sha256(self, file_bytes: bytes) -> str:
        """Compute SHA-256 cryptographic hash of file content."""
        hasher = hashlib.sha256()
        hasher.update(file_bytes)
        return hasher.hexdigest()

    def validate(
        self, file_bytes: bytes, filename: str, reported_mime: str
    ) -> Tuple[str, str]:
        """
        Validate file size, extension, and content signatures.
        Returns:
            Tuple[detected_mime_type, sanitized_filename]
        """
        # 1. Size check
        if len(file_bytes) > self.max_size_bytes:
            mb = self.max_size_bytes / (1024 * 1024)
            raise FileTooLargeError(
                f"File size ({len(file_bytes) / (1024 * 1024):.2f} MB) exceeds maximum allowed limit of {mb:.0f} MB"
            )

        # 2. Extension check
        sanitized_name = self.sanitize_filename(filename)
        ext = Path(sanitized_name).suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"Unsupported file extension '{ext}'. Supported formats: PDF, DOCX, TXT, MD"
            )

        # 3. Magic bytes / signature check
        detected_mime = self._verify_content_signature(file_bytes, ext, reported_mime)

        return detected_mime, sanitized_name

    def _verify_content_signature(
        self, file_bytes: bytes, ext: str, reported_mime: str
    ) -> str:
        """Inspect byte header and structure to verify file authenticity."""
        if ext == ".pdf":
            # PDF header magic: %PDF-
            if not file_bytes.startswith(b"%PDF-"):
                raise FileValidationError(
                    "Invalid PDF: Missing '%PDF-' header signature"
                )
            return "application/pdf"

        elif ext == ".docx":
            # DOCX is a zip archive starting with PK\x03\x04
            if not file_bytes.startswith(b"PK\x03\x04"):
                raise FileValidationError(
                    "Invalid DOCX: Missing standard ZIP signature 'PK\\x03\\x04'"
                )
            try:
                with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
                    names = z.namelist()
                    if "[Content_Types].xml" not in names:
                        raise FileValidationError(
                            "Invalid DOCX structure: Missing '[Content_Types].xml'"
                        )
            except zipfile.BadZipFile as e:
                raise FileValidationError(f"Invalid DOCX archive: {str(e)}")
            return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

        elif ext in {".txt", ".md", ".markdown"}:
            # Text formats: should not contain excessive null bytes or invalid non-text signatures
            sample = file_bytes[:1024]
            if b"\x00" in sample:
                raise FileValidationError(
                    f"Binary content detected in {ext} text file (null byte encountered)"
                )
            # Try decoding text sample
            try:
                sample.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    sample.decode("latin-1")
                except Exception:
                    raise FileValidationError("Unrecognized text encoding")

            if ext in {".md", ".markdown"}:
                return "text/markdown"
            return "text/plain"

        return reported_mime or "application/octet-stream"
