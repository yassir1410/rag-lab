from typing import List, Optional
from app.rag.ingestion.base import DocumentParser
from app.rag.ingestion.exceptions import UnsupportedFileTypeError
from app.rag.ingestion.parsers.pdf_parser import PdfParser
from app.rag.ingestion.parsers.docx_parser import DocxParser
from app.rag.ingestion.parsers.txt_parser import TxtParser
from app.rag.ingestion.parsers.markdown_parser import MarkdownParser


class ParserRegistry:
    """
    Central registry for format parsers.
    Enables dynamic registration and resolution of document parsers by MIME type or file extension.
    """

    def __init__(self):
        self._parsers: List[DocumentParser] = []
        self._register_default_parsers()

    def _register_default_parsers(self):
        """Register the core standard format parsers."""
        self.register(PdfParser())
        self.register(DocxParser())
        self.register(TxtParser())
        self.register(MarkdownParser())

    def register(self, parser: DocumentParser):
        """Add a parser instance to the registry."""
        self._parsers.append(parser)

    def get_parser(self, mime_type: str, filename: str) -> DocumentParser:
        """
        Locate the first matching parser capable of handling the MIME type or filename.
        Raises UnsupportedFileTypeError if no registered parser matches.
        """
        for parser in self._parsers:
            if parser.can_handle(mime_type, filename):
                return parser

        raise UnsupportedFileTypeError(
            f"No registered parser available for file '{filename}' with MIME type '{mime_type}'"
        )


# Global singleton registry instance
default_parser_registry = ParserRegistry()
