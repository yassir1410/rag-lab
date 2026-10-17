# Phase 12 Experiment Report: Real Gemini Vision & Table-Aware Evidence Retrieval Validation

**Date:** 2026-10-17  
**Document Evaluated:** `rag_multimodal_benchmark_solar_system.pdf` (`7f01947d-3c01-4b00-9992-fe87ad0f05fb`)  
**Target Benchmark Query:** *"According to the planetary data table on page 1, what is Jupiter's surface gravity? Cite the source."*  
**Configured Vision Model:** `gemini-3.8-flash`  
**Execution Mode:** Monitored Single Live Upstream API Call + Full Local Regression Suite  

---

## 1. Executive Summary

During Phase 12 multimodal retrieval and generation evaluation, two primary failure modes were diagnosed:
1. **Gemini Vision Upstream Failures:** The legacy implementation used a hardcoded, deprecated model identifier (`gemini-1.5-flash`), placed API keys insecurely in URL query parameters, omitted actual base64 image payload transmission, and silently swallowed upstream API errors into plain text answer fields with hallucinated citations.
2. **Table Evidence Retrieval Bottleneck:** Lexical and hybrid retrieval truncated database elements prematurely via an unranked `LIMIT 8`, lacked possessive and punctuation normalization (failing to match `"Jupiter's"` and `"gravity?"`), lacked table intent prioritization, and leaked container internal filesystem paths (`/app/data/uploads/...`) to API consumers.

Following approved architectural corrections, a single monitored live validation call was conducted against Google's Gemini Vision API with real base64 image bytes from page 1 attached.

### Key Results

| Metric / Requirement | Target Expectation | Verified Empirical Result | Status |
|---|---|---|---|
| **Upstream HTTP Status** | `200 OK` | `200 OK` | **PASSED** |
| **Model Used** | `gemini-3.8-flash` | `gemini-3.8-flash` | **PASSED** |
| **Vision Enabled** | `True` (real image inference) | `True` (`is_vision_enabled: True`) | **PASSED** |
| **Grounded Answer** | Contains `23.1 m/s²` | `"According to the planetary data table on page 1, Jupiter's gravity is **23.1 m/s²** [TABLE_P1]."` | **PASSED** |
| **Primary Evidence Tag** | `[TABLE_P1]` (Page 1) | `[TABLE_P1]` (Score: 1.0, Type: table) | **PASSED** |
| **Valid Citations** | `['[TABLE_P1]']` | `['[TABLE_P1]']` | **PASSED** |
| **Hallucinated Citations** | `[]` (0 hallucinations) | `[]` (0 hallucinations) | **PASSED** |
| **Public Asset Security** | No raw disk paths in API | Public URLs: `/api/v1/multimodal/documents/{id}/pages/1/image` | **PASSED** |
| **Server-Side Latency** | $< 5000\text{ ms}$ | $3459.07\text{ ms}$ | **PASSED** |
| **Mock Provider Invocation** | Zero fallback | Strictly disabled when `provider="gemini"` | **PASSED** |

---

## 2. Upstream Gemini API Model Availability Findings

When probing Google's `v1beta` REST endpoint for the configured API key:
- Requesting `gemini-1.5-flash` returned `HTTP 404: models/gemini-1.5-flash is not found for API version v1beta`.
- Requesting `gemini-2.5-flash` returned:
  ```json
  HTTP 404 Not Found
  {
    "detail": "Gemini model 'gemini-2.5-flash' not found for API version v1beta: This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.8-flash for the latest features and improvements."
  }
  ```
- Requesting `gemini-3.8-flash` returned `HTTP 200 OK`, supporting full multimodal base64 image parts via `inline_data`.

**Decision:** `GEMINI_MODEL=gemini-3.8-flash` is established as the active, verified default. Model selection is configurable via environment settings and visible in all response payloads.

---

## 3. Implemented Architectural Corrections

### A. Authentication & Multimodal Transmission (`vision_provider.py`)
- Removed API keys from URL query strings. Transmitted securely via header:
  ```http
  x-goog-api-key: [REDACTED]
  Content-Type: application/json
  ```
- Constructed compliant `inline_data` base64 parts:
  ```python
  parts.append({
      "inline_data": {
          "mime_type": mime,
          "data": base64_image_bytes,
      }
  })
  ```
- Enforced strict filesystem boundaries: paths must reside within `/app/data/uploads`, files must exist, size must be $\le 15\text{ MB}$, and maximum 2 images are transmitted.
- Defined `VisionProviderError(message, status_code)`. Strictly classified HTTP errors (401, 404, 429, 502, 504) without converting upstream failures into synthetic answers.

### B. Table-Aware Evidence Retrieval (`text_retriever.py`)
- Implemented `normalize_query_terms`:
  - Token extraction using `\b[a-zA-Z0-9_\u00c0-\u017f]+\b`.
  - Stripped English possessives (`jupiter's` $\to$ `jupiter`).
  - Removed standard question and citation stopwords (`"what"`, `"cite"`, `"source"`, `"according"`, `"the"`).
- Removed arbitrary SQL limit. Fetched matching candidates before ranking.
- Added table intent detection: queries containing `"table"`, `"tableau"`, `"tabular"`, or `"matrix"` apply a $1.35\times$ relevance multiplier to `element_type == "table"`.
- Resolved local asset paths to zero-padded page images (`page_{page_number:03d}.png`) for seamless base64 encoding.

### C. Context Injection & Public Asset Isolation (`multimodal_context.py` & `service.py`)
- Extracted Markdown representations from `structured_data["markdown"]` and prepended structured evidence headers (`--- EVIDENCE [TABLE_P1] (Page 1, Type: table) ---`).
- Replaced container file paths with authorized HTTP endpoints (`/api/v1/multimodal/documents/{id}/pages/{page}/image`) in all client-facing models.
- Completely removed automatic fallback to `MockVisionProvider` when `provider == "gemini"` was requested.

---

## 4. Single Monitored Validation Execution Log

### Pre-flight Verification
1. **FastAPI Health:** Verified `GET /api/v1/health` $\to$ `HTTP 200 OK`.
2. **Page Image Asset:** Confirmed `/app/data/uploads/multimodal/7f01947d-3c01-4b00-9992-fe87ad0f05fb/page_001.png` exists ($318,020\text{ bytes}$).
3. **Pydantic Schema:** Confirmed `VisionQAResponse.citations` accommodates `List[Union[str, Dict[str, Any]]]`.
4. **Process Health:** Confirmed zero orphaned Python client tasks in container `/proc`.
5. **Client Timeout:** Configured client-side timeout at $60\text{ seconds}$.

### Request Payload
```json
{
  "document_id": "7f01947d-3c01-4b00-9992-fe87ad0f05fb",
  "question": "According to the planetary data table on page 1, what is Jupiter's surface gravity? Cite the source.",
  "page_number": 1,
  "provider": "gemini"
}
```

### Empirical Response (Redacted)
```json
{
  "question": "According to the planetary data table on page 1, what is Jupiter's surface gravity? Cite the source.",
  "answer": "According to the planetary data table on page 1, Jupiter's gravity is **23.1 m/s²** [TABLE_P1].",
  "citations": [
    "[TABLE_P1]"
  ],
  "evidence_items": [
    {
      "citation_tag": "[TABLE_P1]",
      "element_type": "table",
      "page_number": 1,
      "score": 1.0,
      "image_path": "/api/v1/multimodal/documents/7f01947d-3c01-4b00-9992-fe87ad0f05fb/pages/1/image"
    },
    {
      "citation_tag": "[IMG_P1]",
      "element_type": "figure",
      "page_number": 1,
      "score": 1.0,
      "image_path": "/api/v1/multimodal/documents/7f01947d-3c01-4b00-9992-fe87ad0f05fb/pages/1/image"
    },
    {
      "citation_tag": "[TITLE_P1]",
      "element_type": "title",
      "page_number": 1,
      "score": 0.9839,
      "image_path": "/api/v1/multimodal/documents/7f01947d-3c01-4b00-9992-fe87ad0f05fb/pages/1/image"
    },
    {
      "citation_tag": "[PARAGRAPH_P1]",
      "element_type": "paragraph",
      "page_number": 1,
      "score": 0.9683,
      "image_path": "/api/v1/multimodal/documents/7f01947d-3c01-4b00-9992-fe87ad0f05fb/pages/1/image"
    }
  ],
  "model_used": "gemini-3.8-flash",
  "is_vision_enabled": true,
  "warnings": [],
  "latency_ms": 3459.07
}
```

---

## 5. Verification Checklist

- [x] **Actual Image Bytes Transmitted:** Validated base64 payload from `page_001.png` ($318\text{ KB}$) passed through `inline_data`.
- [x] **Upstream HTTP 200:** Confirmed Google API responded with HTTP 200.
- [x] **Accurate Grounded Fact:** Verified Jupiter gravity extracted as `23.1 m/s²`.
- [x] **Table Relevance:** Ranked `[TABLE_P1]` as #1 evidence item with score `1.0`.
- [x] **Exact Citation:** Validated `[TABLE_P1]` citation matching evidence tag.
- [x] **Zero Hallucinations:** `warnings: []`, `hallucinated_citations: []`.
- [x] **Vision Flag Accuracy:** `is_vision_enabled: True` confirmed only upon successful transmission and generation.
- [x] **Strict Provider Integrity:** Mock provider was never invoked during live Gemini requests.

---

## 6. Regression Testing & Build Verification

To guarantee that the changes did not break existing system capabilities and did not initiate unauthorized API calls:

1. **Phase 12 Multimodal Regression Suite (`pytest /app/tests/test_multimodal_rag.py`):**
   - **Result:** `16 passed in 15.42s`.
   - Included new unit tests for query normalization, Markdown table context building, and `VisionProviderError` status code classification.
2. **Full System Backend Regression Suite (`pytest /app/tests/`):**
   - **Result:** `149 passed in 27.66s` (covering Phases 01 through 12).
   - Zero test failures, zero regressions.
3. **Frontend Production Build (`npm run build`):**
   - **Result:** Succeeded in $8.50\text{ s}$ (`dist/index.html`, `dist/assets/`). Zero syntax or bundle errors.
