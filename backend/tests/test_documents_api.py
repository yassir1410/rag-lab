import io
from fastapi.testclient import TestClient


def test_upload_and_retrieve_txt_flow(client: TestClient, synthetic_txt_bytes):
    # 1. Upload
    response = client.post(
        "/api/v1/documents",
        files={"file": ("rag_intro.txt", synthetic_txt_bytes, "text/plain")},
    )
    assert response.status_code == 201
    doc_data = response.json()
    doc_id = doc_data["id"]
    assert doc_data["original_filename"] == "rag_intro.txt"
    assert doc_data["status"] == "processed"
    assert doc_data["sections_count"] == 3

    # 2. List documents
    list_res = client.get("/api/v1/documents")
    assert list_res.status_code == 200
    list_data = list_res.json()
    assert list_data["total"] == 1
    assert list_data["items"][0]["id"] == doc_id

    # 3. Get document details
    get_res = client.get(f"/api/v1/documents/{doc_id}")
    assert get_res.status_code == 200
    assert get_res.json()["id"] == doc_id

    # 4. Get sections
    sec_res = client.get(f"/api/v1/documents/{doc_id}/sections")
    assert sec_res.status_code == 200
    sections = sec_res.json()
    assert len(sections) == 3
    assert sections[0]["section_index"] == 0
    assert "Retrieval-Augmented Generation" in sections[0]["content"]

    # 5. Get content comparison
    content_res = client.get(f"/api/v1/documents/{doc_id}/content")
    assert content_res.status_code == 200
    content_data = content_res.json()
    assert "raw_text" in content_data
    assert "normalized_text" in content_data
    assert "comparison_stats" in content_data
    assert content_data["sections_count"] == 3

    # 6. Delete document
    del_res = client.delete(f"/api/v1/documents/{doc_id}")
    assert del_res.status_code == 204

    # Verify deleted
    get_after_del = client.get(f"/api/v1/documents/{doc_id}")
    assert get_after_del.status_code == 404


def test_duplicate_document_rejection(client: TestClient, synthetic_txt_bytes):
    # Upload first time
    res1 = client.post(
        "/api/v1/documents",
        files={"file": ("file1.txt", synthetic_txt_bytes, "text/plain")},
    )
    assert res1.status_code == 201
    doc1_id = res1.json()["id"]

    # Upload second time with same content (even with different name)
    res2 = client.post(
        "/api/v1/documents",
        files={"file": ("file2_renamed.txt", synthetic_txt_bytes, "text/plain")},
    )
    assert res2.status_code == 409
    error_data = res2.json()
    assert "detail" in error_data
    assert error_data["detail"]["existing_document_id"] == doc1_id


def test_upload_pdf(client: TestClient, synthetic_pdf_bytes):
    response = client.post(
        "/api/v1/documents",
        files={"file": ("sample.pdf", synthetic_pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["page_count"] == 2
    assert data["extraction_method"] == "pymupdf"


def test_upload_empty_file_rejected(client: TestClient):
    response = client.post(
        "/api/v1/documents",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert response.status_code == 400


def test_upload_unsupported_format(client: TestClient):
    response = client.post(
        "/api/v1/documents",
        files={"file": ("app.bin", b"binary data", "application/octet-stream")},
    )
    assert response.status_code == 400
