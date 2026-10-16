# Laboratory Experiments — Phase 02: Document Ingestion & Parsing

This guide details hands-on, reproducible experiments to observe the inner workings of multi-format document parsing and normalization.

---

## Experiment A: TXT vs PDF Extraction Dynamics

### Goal
Observe how text extraction mechanisms differ between stream-oriented text (`.txt`) and visual coordinate-oriented documents (`.pdf`).

### Setup
Use `sample_documents/sample.txt` and generate or upload a standard 2-page PDF.

### Steps
1. Navigate to the **Documents** tab in the web dashboard.
2. Upload `sample.txt`.
3. Upload `sample.pdf`.
4. Inspect the extracted sections and the page numbers attributed to each.

### Expected Observations
- The TXT document segments cleanly into paragraphs without fixed page coordinates (`page_number = null`).
- The PDF document segments by spatial bounding boxes and explicitly tags sections with their 1-indexed page numbers (`Page 1`, `Page 2`).

### Explanation
Plain text files contain natural newline flow controls, while PDFs define glyph placements. PyMuPDF reconstructs blocks using geometric heuristics.

### Limitations
Complex multi-column or overlapping text in PDFs may occasionally require OCR or layout-aware vision models to discern reading order.

---

## Experiment B: DOCX Headings and Table Preservation

### Goal
Verify that Word document tables are converted to Markdown without destroying cell boundaries, and that document headings are detected.

### Setup
Create a simple `.docx` document containing an `Heading 1` and a 2x2 table.

### Steps
1. Upload the `.docx` file in the dashboard.
2. Click the eye icon to view extracted sections.
3. Observe the `content_type` of each section.

### Expected Observations
- Headings are labeled with `content_type = "heading"` and assigned as metadata to subsequent sections.
- Tables are transformed into Markdown tables:
  ```markdown
  | Header 1 | Header 2 |
  | --- | --- |
  | Cell 1 | Cell 2 |
  ```

### Explanation
By traversing `doc.element.body` in true document order, `python-docx` captures the sequential flow of both text and tables.

### Limitations
Nested or merged table cells can become complex in plain Markdown notation; flat tables remain the most reliable representation for LLM context windows.

---

## Experiment C: Raw vs Normalized Text Comparison

### Goal
Quantify whitespace reduction and clean-up performance across documents.

### Setup
A text or PDF file containing irregular spacing, excessive blank lines, and Windows CRLF line endings (`\r\n`).

### Steps
1. Upload the file and open the **Learning & Comparison** tab in the document drawer.
2. Inspect the **Raw vs Normalized Chars** metric and the side-by-side view.

### Expected Observations
- Normalized character count is typically 2% to 15% lower than raw character count.
- Indentation inside Markdown code blocks (` ``` `) remains intact.
- Excess blank lines are collapsed to single paragraph breaks.

### Explanation
`TextNormalizer` applies Unicode NFKC normalization and whitespace reduction while detecting code fences to prevent syntax corruption.

---

## Experiment D: Duplicate Uploads & Collision Policy

### Goal
Test how the ingestion engine handles duplicate files with identical contents versus different filenames.

### Setup
A sample document `test.txt`.

### Steps
1. Upload `test.txt`. Notice successful 201 response.
2. Make a copy of `test.txt` named `test_renamed.txt`.
3. Upload `test_renamed.txt`.

### Expected Observations
- The upload is rejected with **HTTP 409 Conflict**.
- The API response points directly to the existing document ID and its original filename.

### Explanation
Content uniqueness is computed using cryptographic SHA-256 hashing. Regardless of the filename, identical binary content yields an identical hash, enforcing duplicate rejection at the database level.

---

## Experiment E: Scanned / Empty PDF Behavior

### Goal
Examine how the system responds to image-only or empty pages without making false claims about OCR.

### Setup
A PDF consisting of an image or a blank page.

### Steps
1. Upload the image-based PDF.
2. Check the document warnings in the table or details drawer.

### Expected Observations
- The parser does not fail with an unhandled exception.
- A prominent amber badge displays: *"Page X: Scanned or image-only page detected. OCR is required to extract text (OCR not performed in Phase 02)."*

### Explanation
The parser inspects `page.get_images()` and character density. Rather than hallucinating or silently dropping content, it explicitly alerts the user that an OCR engine is required.
