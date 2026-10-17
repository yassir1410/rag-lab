import os
import io
import uuid
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.db.session import SessionLocal
from app.models.document import Document
from app.rag.multimodal.ingestion.image_loader import SafeImageLoader
from app.rag.multimodal.ingestion.scanned_pdf import ScannedPDFProcessor
from app.rag.multimodal.ingestion.layout_analyzer import DocumentLayoutAnalyzer
from app.rag.multimodal.ingestion.table_extractor import TableExtractor
from app.rag.multimodal.ocr.tesseract_provider import (
    TesseractOCRProvider,
    EducationalOCRComparator,
)
from app.rag.multimodal.embeddings.image_embedding_provider import (
    MultimodalEmbeddingProvider,
)
from app.rag.multimodal.embeddings.multimodal_registry import (
    MultimodalModelRegistry,
)
from app.rag.multimodal.retrieval.fusion import HybridMultimodalFusion
from app.rag.multimodal.generation.vision_provider import MockVisionProvider
from app.rag.multimodal.generation.citation_manager import MultimodalCitationManager
from app.rag.multimodal.evaluation.metrics import MultimodalMetrics
from app.rag.multimodal.evaluation.benchmark import MultimodalBenchmarkRunner
from app.rag.multimodal.service import MultimodalService

client = TestClient(app)


def test_safe_image_loader_validation_and_bomb_protection(tmp_path):
    # 1. Valid PNG
    valid_file = tmp_path / "test.png"
    img = Image.new("RGB", (100, 100), color="blue")
    img.save(valid_file, "PNG")

    is_valid, err = SafeImageLoader.validate_image_file(str(valid_file))
    assert is_valid is True
    assert err is None

    loaded = SafeImageLoader.load_rgb_image(str(valid_file))
    assert loaded.size == (100, 100)

    # 2. Corrupted file
    corrupt_file = tmp_path / "corrupt.png"
    corrupt_file.write_bytes(b"not-an-image")
    is_valid, err = SafeImageLoader.validate_image_file(str(corrupt_file))
    assert is_valid is False
    assert "Corrupted" in err or "Unsupported" in err


def test_scanned_pdf_processor_detection():
    # Character count threshold test
    assert ScannedPDFProcessor.SCANNED_CHAR_THRESHOLD == 40


def test_layout_analyzer_reading_order():
    # Verify reading order logic
    elements = [
        {"element_type": "title", "_geom": (50, 50, 500, 80)},
        {"element_type": "header", "_geom": (50, 10, 500, 30)},
        {"element_type": "paragraph", "_geom": (50, 100, 250, 300)},
        {"element_type": "paragraph", "_geom": (300, 100, 550, 300)},
        {"element_type": "footer", "_geom": (50, 750, 500, 780)},
    ]
    # Header should precede footer and body
    assert len(elements) == 5


def test_table_extractor_structure():
    # Create synthetic table data
    headers = ["Metric", "Baseline", "Phase 12"]
    rows = [["Recall@5", "0.50", "0.95"], ["Latency (ms)", "120", "42"]]
    
    table_dict = {
        "headers": headers,
        "rows": rows,
        "num_rows": 2,
        "num_cols": 3,
    }
    acc = MultimodalMetrics.calculate_table_accuracy(table_dict, table_dict)
    assert acc == 1.0


def test_tesseract_ocr_provider_and_cer_wer():
    provider = TesseractOCRProvider()
    assert isinstance(provider.get_supported_languages(), list)

    # Educational CER/WER calculation
    ref = "Software Engineer at TechCorp"
    hyp_exact = "Software Engineer at TechCorp"
    hyp_error = "Softwre Enginee at TechCrp"

    cer_exact = EducationalOCRComparator.compute_cer(ref, hyp_exact)
    wer_exact = EducationalOCRComparator.compute_wer(ref, hyp_exact)
    assert cer_exact == 0.0
    assert wer_exact == 0.0

    cer_noisy = EducationalOCRComparator.compute_cer(ref, hyp_error)
    wer_noisy = EducationalOCRComparator.compute_wer(ref, hyp_error)
    assert 0.0 < cer_noisy < 0.5
    assert 0.0 < wer_noisy <= 0.75


def test_multimodal_embeddings_dimension():
    provider = MultimodalEmbeddingProvider()
    vec_text = provider.encode_text("search for performance chart")
    assert len(vec_text) == 512

    test_img = Image.new("RGB", (64, 64), color="red")
    vec_img = provider.encode_image(test_img)
    assert len(vec_img) == 512


def test_hybrid_multimodal_fusion():
    text_results = [
        {"id": "elem-1", "score": 0.9, "citation_tag": "[PARAGRAPH_P1]"},
        {"id": "elem-2", "score": 0.7, "citation_tag": "[TBL_P1]"},
    ]
    visual_results = [
        {"id": "elem-3", "score": 0.95, "citation_tag": "[IMG_P1]"},
        {"id": "elem-1", "score": 0.6, "citation_tag": "[PARAGRAPH_P1]"},
    ]

    fused = HybridMultimodalFusion.rrf_fuse(
        text_results=text_results,
        visual_results=visual_results,
        top_k=3,
    )
    assert len(fused) == 3
    # elem-1 was present in both, should rank high
    assert fused[0]["strategy"] == "hybrid"


def test_vision_provider_and_citations():
    mock_prov = MockVisionProvider()
    assert mock_prov.is_available() is True

    res = mock_prov.generate_answer(
        question="What is shown in the diagram?",
        context_text="--- EVIDENCE [IMG_P1] --- Visual diagram on page 1",
        image_paths=["/dummy/path.png"],
    )
    assert "[IMG_P1]" in res["answer"]
    assert res["is_vision_enabled"] is True

    # Validate citations
    evidence = [{"citation_tag": "[IMG_P1]"}]
    val = MultimodalCitationManager.validate_citations(res["answer"], evidence)
    assert val["is_grounded"] is True
    assert "[IMG_P1]" in val["valid_citations"]
    assert len(val["hallucinated_citations"]) == 0


def test_multimodal_api_endpoints():
    # 1. Models endpoint
    resp = client.get("/api/v1/multimodal/models")
    assert resp.status_code == 200
    models = resp.json()
    assert len(models) >= 2
    assert any(m["model_id"] == "clip-ViT-B-32" for m in models)

    # 2. Benchmark endpoint
    resp_bm = client.post("/api/v1/multimodal/benchmark")
    assert resp_bm.status_code == 200
    bm = resp_bm.json()
    assert bm["total_scenarios"] == 15
    assert bm["pass_rate"] >= 90.0


def test_multimodal_benchmark_runner():
    report = MultimodalBenchmarkRunner.run_benchmark()
    assert report["total_scenarios"] == 15
    assert report["passed_count"] == 15
    assert report["pass_rate"] == 100.0


def test_documents_api_structure_for_multimodal_selector():
    resp = client.get("/api/v1/documents?limit=50&offset=0")
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert "total" in data
    assert data["total"] >= 2
    items = data["items"]
    assert len(items) >= 2

    # Verify solar system benchmark document exists with proper keys
    solar_doc = next((d for d in items if d.get("original_filename") == "rag_multimodal_benchmark_solar_system.pdf"), None)
    assert solar_doc is not None
    assert "id" in solar_doc
    assert solar_doc["status"] == "processed"
    assert "pdf" in solar_doc["mime_type"]


def test_solar_system_benchmark_document_layout_and_elements():
    # Retrieve documents to get the ID dynamically
    resp = client.get("/api/v1/documents?limit=50&offset=0")
    assert resp.status_code == 200
    items = resp.json()["items"]
    solar_doc = next((d for d in items if d.get("original_filename") == "rag_multimodal_benchmark_solar_system.pdf"), None)
    assert solar_doc is not None
    doc_id = solar_doc["id"]

    # Verify pages
    pages_resp = client.get(f"/api/v1/multimodal/documents/{doc_id}/pages")
    assert pages_resp.status_code == 200
    pages = pages_resp.json()
    assert len(pages) == 4
    assert pages[0]["page_number"] == 1
    assert pages[0]["width"] > 0
    assert pages[0]["height"] > 0

    # Verify elements and reading order
    elems_resp = client.get(f"/api/v1/multimodal/documents/{doc_id}/elements?page_number=1")
    assert elems_resp.status_code == 200
    elems = elems_resp.json()
    assert len(elems) > 0
    # Verify reading order is strictly ascending
    orders = [e["reading_order"] for e in elems]
    assert orders == sorted(orders)
    assert elems[0]["bounding_box"]["width"] > 0
    assert elems[0]["bounding_box"]["height"] > 0

    # Verify page image endpoint
    img_resp = client.get(f"/api/v1/multimodal/documents/{doc_id}/pages/1/image")
    assert img_resp.status_code == 200
    assert "image/png" in img_resp.headers.get("content-type", "")
    assert len(img_resp.content) > 1000


def test_table_extractor_multi_strategy_and_benchmark_pdf():
    # Retrieve documents to get the ID dynamically
    resp = client.get("/api/v1/documents?limit=50&offset=0")
    assert resp.status_code == 200
    items = resp.json()["items"]
    solar_doc = next((d for d in items if d.get("original_filename") == "rag_multimodal_benchmark_solar_system.pdf"), None)
    assert solar_doc is not None
    doc_id = solar_doc["id"]

    # 1. Document-wide extraction (all pages)
    table_resp = client.post("/api/v1/multimodal/tables/extract", json={
        "document_id": doc_id,
    })
    assert table_resp.status_code == 200
    data = table_resp.json()
    assert data["count"] == 4
    tables = data["tables"]
    assert len(tables) == 4

    # Verify Page 1 Table (Planetary Regimes Part 1)
    t1 = next((t for t in tables if t["page_number"] == 1), None)
    assert t1 is not None
    assert t1["num_cols"] == 10
    assert t1["num_rows"] == 6
    assert "Body" in t1["headers"][0]
    assert "Density" in t1["headers"][4]
    assert t1["rows"][0][0] == "Mercury"
    assert t1["rows"][-1][0] == "Jupiter"
    assert "|" in t1["markdown_table"]
    assert "---" in t1["markdown_table"]

    # Verify Page 2 Table (Planetary Regimes Part 2)
    t2 = next((t for t in tables if t["page_number"] == 2), None)
    assert t2 is not None
    assert t2["num_cols"] == 10
    assert t2["num_rows"] == 4
    assert t2["rows"][0][0] == "Saturn"
    assert t2["rows"][-1][0] == "Pluto"

    # Verify Page 3 Table (Orbital & Atmospheric Parameters)
    t3 = next((t for t in tables if t["page_number"] == 3), None)
    assert t3 is not None
    assert t3["num_cols"] == 10
    assert t3["num_rows"] == 10
    assert "Sun Dist." in t3["headers"][1]
    assert "Moons" in t3["headers"][9]
    assert t3["rows"][0][0] == "Mercury"
    assert t3["rows"][-1][0] == "Pluto"

    # Verify Page 4 Table (JWST Scientific Instruments)
    t4 = next((t for t in tables if t["page_number"] == 4), None)
    assert t4 is not None
    assert t4["num_cols"] == 4
    assert t4["num_rows"] == 4
    assert "Instrument" in t4["headers"][0]
    assert "Spectral" in t4["headers"][2]
    assert t4["rows"][0][0] == "NIRCam"
    assert t4["rows"][-1][0] == "MIRI"

    # 2. Targeted single-page extraction
    single_p3_resp = client.post("/api/v1/multimodal/tables/extract", json={
        "document_id": doc_id,
        "page_number": 3,
    })
    assert single_p3_resp.status_code == 200
    p3_data = single_p3_resp.json()
    assert p3_data["count"] == 1
    assert p3_data["tables"][0]["page_number"] == 3
    assert p3_data["tables"][0]["num_rows"] == 10

    # 3. Persistence verification: tables persisted in DocumentElement
    elems_resp = client.get(f"/api/v1/multimodal/documents/{doc_id}/elements?page_number=1")
    assert elems_resp.status_code == 200
    elems = elems_resp.json()
    table_elems = [e for e in elems if e["element_type"] == "table"]
    assert len(table_elems) >= 1
    assert table_elems[0]["structured_data"]["num_rows"] == 6
    assert table_elems[0]["structured_data"]["num_cols"] == 10


def test_ocr_text_retriever_query_normalization():
    from app.rag.multimodal.retrieval.text_retriever import OCRTextRetriever

    raw_query = "According to the planetary data table on page 1, what is Jupiter's surface gravity? Cite the source."
    terms = OCRTextRetriever.normalize_query_terms(raw_query)

    # Possessive stripped
    assert "jupiter" in terms
    assert "jupiter's" not in terms

    # Punctuation stripped
    assert "gravity" in terms
    assert "gravity?" not in terms

    # Stopwords removed
    for sw in ["the", "is", "on", "what", "according", "cite"]:
        assert sw not in terms

    # Important domain words preserved
    assert "planetary" in terms
    assert "data" in terms
    assert "table" in terms
    assert "surface" in terms


def test_multimodal_context_builder_markdown_tables_and_safety(tmp_path, monkeypatch):
    from app.rag.multimodal.generation.multimodal_context import MultimodalContextBuilder
    from app.core.config import settings

    monkeypatch.setattr(settings, "STORAGE_DIR", str(tmp_path))

    safe_img = tmp_path / "page_001.png"
    safe_img.write_bytes(b"dummy image bytes")

    unsafe_img = tmp_path.parent / "unsafe.png"
    unsafe_img.write_bytes(b"unsafe bytes")

    evidence = [
        {
            "citation_tag": "[TABLE_P1]",
            "element_type": "table",
            "page_number": 1,
            "content_text": "Markdown content here",
            "structured_data": {
                "markdown": "| Body | Gravity |\n| Jupiter | 23.1 |",
            },
            "_local_image_path": str(safe_img),
        },
        {
            "citation_tag": "[IMG_P1]",
            "element_type": "figure",
            "page_number": 1,
            "content_text": "Figure",
            "_local_image_path": str(unsafe_img),
        }
    ]

    context_str, images = MultimodalContextBuilder.build_context(evidence)
    assert "[TABLE_P1]" in context_str
    assert "| Body | Gravity |" in context_str
    assert "23.1" in context_str
    # Safe image within allowed storage dir is included
    assert str(safe_img.resolve()) in images
    # Unsafe image outside allowed storage dir is excluded
    assert str(unsafe_img.resolve()) not in images


def test_gemini_vision_provider_strict_error_handling(monkeypatch):
    from app.rag.multimodal.generation.vision_provider import (
        GeminiVisionProvider,
        VisionProviderError,
    )
    import httpx

    # Test 1: Missing API key
    provider = GeminiVisionProvider(api_key="")
    assert provider.is_available() is False

    # Test 2: Upstream HTTP 404 mapped to VisionProviderError
    def mock_post_404(*args, **kwargs):
        req = httpx.Request("POST", "https://mock.url")
        return httpx.Response(404, request=req, json={"error": {"message": "Model not found"}})

    monkeypatch.setattr(httpx.Client, "post", mock_post_404)
    prov_with_key = GeminiVisionProvider(api_key="test_key")

    with pytest.raises(VisionProviderError) as exc_info:
        prov_with_key.generate_answer(
            question="What is this?",
            context_text="Some context",
            image_paths=[],
        )
    assert exc_info.value.status_code == 404
    assert "Model not found" in exc_info.value.message

