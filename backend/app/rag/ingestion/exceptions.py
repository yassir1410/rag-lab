class IngestionError(Exception):
    """Base exception for all document ingestion errors."""
    pass


class UnsupportedFileTypeError(IngestionError):
    """Raised when an uploaded file format is not supported."""
    pass


class FileValidationError(IngestionError):
    """Raised when file validation fails (magic bytes, malformed headers, etc.)."""
    pass


class FileTooLargeError(FileValidationError):
    """Raised when an uploaded file exceeds the configured size limit."""
    pass


class DuplicateDocumentError(IngestionError):
    """Raised when a document with identical SHA-256 hash already exists."""
    def __init__(self, message: str, existing_document_id: str, existing_filename: str):
        super().__init__(message)
        self.existing_document_id = existing_document_id
        self.existing_filename = existing_filename


class CorruptedFileError(IngestionError):
    """Raised when a document is corrupted and cannot be parsed."""
    pass


class EncryptedFileError(IngestionError):
    """Raised when a document is encrypted or password-protected."""
    pass
