import pytest
from app.rag.ingestion.parsers.pdf_parser import PdfParser
from app.rag.ingestion.parsers.docx_parser import DocxParser
from app.rag.ingestion.parsers.txt_parser import TxtParser
from app.rag.ingestion.parsers.markdown_parser import MarkdownParser
from app.rag.ingestion.exceptions import CorruptedFileError


def test_pdf_multipage_extraction(synthetic_pdf_bytes):
    """Test PDF extraction preserves page numbers and text blocks."""
    parser = PdfParser()
    doc = parser.parse(synthetic_pdf_bytes, "test_document.pdf")

    assert doc.original_filename == "test_document.pdf"
    assert doc.page_count == 2
    assert doc.extraction_method == "pymupdf"
    assert len(doc.sections) >= 2

    # Check page numbers are 1-indexed
    page_numbers = {s.page_number for s in doc.sections if s.page_number is not None}
    assert 1 in page_numbers
    assert 2 in page_numbers
    assert "Principles of Retrieval" in doc.raw_text
    assert "Context Synthesis" in doc.raw_text


def test_pdf_empty_page_warning(synthetic_empty_pdf_bytes):
    """Test PDF with blank page produces warning without failing."""
    parser = PdfParser()
    doc = parser.parse(synthetic_empty_pdf_bytes, "empty.pdf")

    assert doc.page_count == 1
    assert len(doc.warnings) >= 1
    assert any("Blank page" in w for w in doc.warnings)


def test_pdf_corrupted_file():
    """Corrupted PDF bytes should raise CorruptedFileError."""
    parser = PdfParser()
    with pytest.raises(CorruptedFileError):
        parser.parse(b"%PDF-invalid-binary-content-corrupted", "corrupted.pdf")


def test_docx_parser(synthetic_docx_bytes):
    """Test DOCX parser extracts headings, paragraphs, and formatted tables."""
    parser = DocxParser()
    doc = parser.parse(synthetic_docx_bytes, "test.docx")

    assert doc.original_filename == "test.docx"
    assert doc.extraction_method == "python-docx"
    assert len(doc.sections) >= 3

    # Check heading extraction
    headings = [s.heading for s in doc.sections if s.heading]
    assert "Deep Learning Foundations" in headings or "Model Parameters" in headings

    # Check table extraction as Markdown table
    tables = [s for s in doc.sections if s.content_type == "table"]
    assert len(tables) >= 1
    table_content = tables[0].content
    assert "| Parameter | Value |" in table_content
    assert "| Batch Size | 32 |" in table_content


def test_docx_corrupted():
    """Corrupted DOCX raises CorruptedFileError."""
    parser = DocxParser()
    with pytest.raises(CorruptedFileError):
        parser.parse(b"PK\x03\x04not-a-valid-docx-zip-content", "corrupt.docx")


def test_txt_parser_utf8(synthetic_txt_bytes):
    """Test TXT parser extracts paragraphs separated by blank lines."""
    parser = TxtParser()
    doc = parser.parse(synthetic_txt_bytes, "sample.txt")

    assert doc.extraction_method == "txt_parser"
    assert len(doc.sections) == 3
    assert "Introduction to Retrieval-Augmented Generation" in doc.sections[0].content
    assert "ensures answers are grounded" in doc.sections[2].content


def test_txt_parser_bom(synthetic_txt_bom_bytes):
    """Test TXT parser transparently decodes UTF-8 BOM."""
    parser = TxtParser()
    doc = parser.parse(synthetic_txt_bom_bytes, "sample_bom.txt")

    assert "Document with Byte Order Mark." in doc.raw_text
    assert len(doc.sections) == 2


def test_markdown_parser(synthetic_md_bytes):
    """Test Markdown parser preserves headings, code fences, and tables."""
    parser = MarkdownParser()
    doc = parser.parse(synthetic_md_bytes, "architecture.md")

    assert doc.extraction_method == "markdown_parser"
    assert len(doc.sections) >= 3

    # Code block preservation check
    code_sections = [s for s in doc.sections if s.content_type == "code" or "```" in s.content]
    assert len(code_sections) >= 1
    assert "def compute_similarity(v1, v2):" in code_sections[0].content

    # Table section check
    table_sections = [s for s in doc.sections if s.content_type == "table" or "| Metric |" in s.content]
    assert len(table_sections) >= 1
