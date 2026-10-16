import pytest
from app.evaluation.generation.claim_extractor import claim_extractor
from app.evaluation.generation.faithfulness import faithfulness_evaluator
from app.evaluation.generation.answer_relevance import answer_relevance_evaluator
from app.evaluation.generation.citation_evaluator import citation_evaluator
from app.rag.generation.schemas import ContextSourceItem


def test_claim_extractor_propositions_and_citations():
    text = (
        "The candidate has 3 years of experience in Python [S1]. "
        "He previously built distributed microservices with FastAPI and Docker [S2, S3]. "
        "He is certified in Kubernetes."
    )
    claims = claim_extractor.extract_claims(text)
    assert len(claims) == 3
    assert "[S1]" in claims[0]["citations"]
    assert "[S2]" in claims[1]["citations"]
    assert "[S3]" in claims[1]["citations"]
    assert len(claims[2]["citations"]) == 0


def test_claim_extractor_detects_refusal():
    text = "Based on the provided documents, there is insufficient evidence to answer this question."
    assert claim_extractor.is_refusal(text) is True
    claims = claim_extractor.extract_claims(text)
    assert len(claims) == 1
    assert claims[0]["is_refusal"] is True


def test_faithfulness_evaluator_supported_and_unsupported():
    context = [
        "Candidate Yassir FAHIMI graduated from INSEA with a Software Engineering degree in Rabat.",
        "Experienced in Python, FastAPI, and PostgreSQL with Docker deployments.",
    ]

    # Faithful answer
    faithful_answer = "Yassir FAHIMI graduated from INSEA [S1]. He is experienced in Python and FastAPI [S2]."
    res_faithful = faithfulness_evaluator.evaluate(
        generated_answer=faithful_answer,
        context_passages=context,
        judge_provider="heuristic",
    )
    assert res_faithful.score >= 0.8
    assert res_faithful.supported_claims >= 1

    # Hallucinated / Unsupported answer
    hallucinated_answer = "The candidate worked for 10 years at Google and invented the C++ language in Tokyo."
    res_hallucinated = faithfulness_evaluator.evaluate(
        generated_answer=hallucinated_answer,
        context_passages=context,
        judge_provider="heuristic",
    )
    assert res_hallucinated.score <= 0.3
    assert res_hallucinated.unsupported_claims >= 1


def test_faithfulness_evaluator_correct_refusal():
    context = ["Empty context with no relevant information."]
    refusal = "Based on the provided documents, there is insufficient evidence to answer this question."
    res = faithfulness_evaluator.evaluate(
        generated_answer=refusal,
        context_passages=context,
        judge_provider="heuristic",
    )
    assert res.score == 1.0
    assert res.total_claims == 1
    assert res.supported_claims == 1


def test_answer_relevance_evaluator():
    q = "What programming languages does the candidate know?"
    ref_ans = "The candidate knows Python, Java, and Node.js."

    # High relevance answer
    good_ans = "The candidate has experience with Python, Java, and Node.js for building backend services."
    res_good = answer_relevance_evaluator.evaluate(
        question=q,
        generated_answer=good_ans,
        reference_answer=ref_ans,
        is_answerable=True,
    )
    assert res_good.score >= 0.7
    assert res_good.is_refusal is False

    # Proper refusal for unanswerable question
    unanswerable_q = "What is the candidate's favorite childhood pet?"
    refusal_ans = "There is insufficient evidence in the provided documents to answer."
    res_refusal = answer_relevance_evaluator.evaluate(
        question=unanswerable_q,
        generated_answer=refusal_ans,
        reference_answer="N/A",
        is_answerable=False,
    )
    assert res_refusal.score == 1.0
    assert res_refusal.is_correct_refusal is True


def test_citation_evaluator_valid_and_invalid():
    import uuid
    c1_id = uuid.uuid4()
    c2_id = uuid.uuid4()
    doc_id = uuid.uuid4()

    sources = [
        ContextSourceItem(
            citation_id="[S1]",
            chunk_id=c1_id,
            document_id=doc_id,
            document_name="resume.pdf",
            sequence_index=0,
            token_count=50,
            char_count=150,
            content="Yassir has built backend services in Python and FastAPI.",
            similarity_score=0.9,
            distance_metric="cosine",
        ),
        ContextSourceItem(
            citation_id="[S2]",
            chunk_id=c2_id,
            document_id=doc_id,
            document_name="resume.pdf",
            sequence_index=1,
            token_count=40,
            char_count=120,
            content="Graduated from INSEA Rabat in software engineering.",
            similarity_score=0.85,
            distance_metric="cosine",
        ),
    ]

    answer = "He graduated from INSEA [S2]. He built systems in Python [S1], but also worked at NASA [S99]."
    faith_res = faithfulness_evaluator.evaluate(
        generated_answer=answer,
        context_passages=[s.content for s in sources],
        judge_provider="heuristic",
    )

    cite_report = citation_evaluator.evaluate(
        generated_answer=answer,
        sources=sources,
        faithfulness_result=faith_res,
    )

    assert cite_report.total_citations_made == 3
    assert cite_report.valid_citation_ids == 2  # S1 and S2
    assert "[S99]" in cite_report.invalid_citation_ids
    assert cite_report.citation_id_validity == round(2 / 3, 4)
