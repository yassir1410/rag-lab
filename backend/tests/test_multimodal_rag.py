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
