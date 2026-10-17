import re
import time
import uuid
import os
from typing import List, Dict, Any, Optional, Tuple, AsyncIterator
from sqlalchemy.orm import Session

from app.rag.unified.schemas import (
    UnifiedEvidenceItem,
    UnifiedCitationDetail,
    UnifiedTimingMetrics,
    UnifiedRAGQueryRequest,
    UnifiedRAGQueryResponse,
    RoutingDecision,
)
from app.rag.unified.router import intelligent_router
from app.rag.unified.retrieval_orchestrator import unified_retrieval_orchestrator
from app.rag.generation.registry import llm_registry
from app.rag.generation.prompt_builder import prompt_builder
from app.rag.generation.base import GenerationRequest
from app.rag.multimodal.generation.vision_provider import GeminiVisionProvider, MockVisionProvider, VisionProviderError


class UnifiedGenerationService:
    """
    Source-aware Context Assembly, Multimodal Vision Routing, and Grounded Generation.
    Supports text-based synthesis and genuine multimodal vision reasoning with verifiable [S#] citations.
    """

    CITATION_REGEX = re.compile(r"\[([^\]]*S\d+[^\]]*)\]", re.IGNORECASE)
    S_TOKEN_REGEX = re.compile(r"\b(S\d+)\b", re.IGNORECASE)

    def __init__(self):
        self.router = intelligent_router
        self.retriever = unified_retrieval_orchestrator
        self.gemini_vision = GeminiVisionProvider()
        self.mock_vision = MockVisionProvider()

    def build_unified_context(
        self,
        evidence_items: List[UnifiedEvidenceItem],
        max_tokens: int = 2048,
        attach_images: bool = True,
        target_page_number: Optional[int] = None,
    ) -> Tuple[str, List[str]]:
        """
        Assembles mixed evidence (text passages, Markdown tables, visual assets, graph triples)
        into a coherent cited prompt context. Collects valid server-side image paths only when
        attach_images is True. If target_page_number is specified, only images matching that page
        are collected; otherwise evidence-relevant images across pages are collected up to MAX_IMAGES.
        """
        context_blocks: List[str] = []
        image_paths: List[str] = []

        for item in evidence_items:
            header = f"--- EVIDENCE {item.citation_id} (Type: {item.evidence_type}, Origin: {item.origin_tag}, Page: {item.page_number or 'N/A'}, Doc: {item.document_name}) ---"
            body = item.content.strip()

            if attach_images:
                # If target_page_number is set, only attach image from that page
                if target_page_number is None or item.page_number == target_page_number:
                    local_p = getattr(item, "_local_image_path", None)
                    if local_p and os.path.exists(local_p):
                        real_p = os.path.realpath(local_p)
                        if (real_p.startswith("/app/data") or "uploads" in real_p) and real_p not in image_paths:
                            image_paths.append(real_p)

            context_blocks.append(f"{header}\n{body}")

        full_context = "\n\n".join(context_blocks)
        return full_context, image_paths

    def parse_citations(self, text: str) -> List[str]:
        if not text:
            return []
        found = []
        for bracket in self.CITATION_REGEX.findall(text):
            for tok in self.S_TOKEN_REGEX.findall(bracket):
                norm = f"[{tok.upper()}]"
                if norm not in found:
                    found.append(norm)
        return found

    def validate_citations(
        self,
        generated_answer: str,
        evidence_items: List[UnifiedEvidenceItem],
    ) -> Tuple[List[UnifiedCitationDetail], List[str], List[str]]:
        parsed_cits = self.parse_citations(generated_answer)
        evidence_by_tag: Dict[str, UnifiedEvidenceItem] = {
            item.citation_id.upper(): item for item in evidence_items
        }

        # Also map origin tags (e.g. [TABLE_P1])
        evidence_by_origin: Dict[str, UnifiedEvidenceItem] = {
            item.origin_tag.upper(): item for item in evidence_items
        }

        details: List[UnifiedCitationDetail] = []
        invalid: List[str] = []
        cited_keys = set()

        for cit in parsed_cits:
            matched_item = evidence_by_tag.get(cit) or evidence_by_origin.get(cit)
            if matched_item:
                matched_item.was_cited = True
                cited_keys.add(matched_item.citation_id.upper())
                details.append(
                    UnifiedCitationDetail(
                        citation_id=matched_item.citation_id,
                        origin_tag=matched_item.origin_tag,
                        is_valid=True,
                        document_name=matched_item.document_name,
                        page_number=matched_item.page_number,
                        evidence_type=matched_item.evidence_type,
                        supporting_passage=matched_item.content[:250] + "..." if len(matched_item.content) > 250 else matched_item.content,
                        calibrated_score=matched_item.calibrated_score,
                        image_url=matched_item.image_url,
                        structured_table=matched_item.structured_data,
                    )
                )
            else:
                invalid.append(cit)
                details.append(
                    UnifiedCitationDetail(
                        citation_id=cit,
                        origin_tag=cit,
                        is_valid=False,
                        caveat="Hallucinated citation: identifier was not present in the retrieved evidence.",
                    )
                )

        uncited = [it.citation_id for it in evidence_items if it.citation_id.upper() not in cited_keys]
        return details, invalid, uncited

    def execute_query(
        self,
        session: Session,
        request: UnifiedRAGQueryRequest,
    ) -> UnifiedRAGQueryResponse:
        t_total_start = time.perf_counter()
        timing = UnifiedTimingMetrics()
        trace_id = str(uuid.uuid4())
        warnings: List[str] = []

        # 1. Routing Phase
        t_route_start = time.perf_counter()
        routing_decision: RoutingDecision = self.router.route_query(
            session=session,
            question=request.question,
            document_id=request.document_id,
            page_number=request.page_number,
            mode=request.mode,
            manual_routes=request.manual_routes,
        )
        timing.routing_latency_ms = round((time.perf_counter() - t_route_start) * 1000.0, 2)
        warnings.extend(routing_decision.warnings)

        # 2. Retrieval Phase using resolved effective_page_number
        t_ret_start = time.perf_counter()
        evidence_items, per_route_latency = self.retriever.retrieve(
            session=session,
            query=request.question,
            routes=routing_decision.selected_routes,
            document_id=request.document_id,
            page_number=routing_decision.effective_page_number,
            top_k=request.top_k,
        )
        timing.retrieval_latency_ms = round((time.perf_counter() - t_ret_start) * 1000.0, 2)
        timing.per_route_latency_ms = per_route_latency

        if not evidence_items:
            timing.total_latency_ms = round((time.perf_counter() - t_total_start) * 1000.0, 2)
            return UnifiedRAGQueryResponse(
                trace_id=trace_id,
                question=request.question,
                answer="No relevant documentary evidence could be retrieved for this query. Please verify that the selected document has been indexed.",
                routing=routing_decision,
                evidence_items=[],
                citations=[],
                invalid_citations=[],
                uncited_sources=[],
                model_used="none",
                is_vision_enabled=False,
                timing_metrics=timing,
                warnings=warnings + ["No evidence retrieved across selected routes."],
            )

        # 3. Context Assembly & Visual Requirement Determination
        # Genuinely visual questions: visual route selected OR top evidence items are visual figures/images
        is_visual_required = (
            "multimodal_visual" in routing_decision.selected_routes
            or any(it.evidence_type in ("image", "figure", "visual", "diagram") for it in evidence_items[:2])
        )

        t_ctx_start = time.perf_counter()
        context_str, image_paths = self.build_unified_context(
            evidence_items=evidence_items,
            max_tokens=request.max_context_tokens,
            attach_images=is_visual_required,
            target_page_number=routing_decision.effective_page_number,
        )
        timing.fusion_latency_ms = round((time.perf_counter() - t_ctx_start) * 1000.0, 2)

        # 4. Generation Phase (Route-Aware Text vs Vision Routing)
        # Vision generation is only invoked when visual inspection is actually required AND images are present.
        # Structured table queries answered by Markdown evidence use text generation without image attachments.
        t_gen_start = time.perf_counter()
        use_vision = is_visual_required and len(image_paths) > 0

        if use_vision:
            # Genuine multimodal Vision reasoning with base64 image bytes
            if request.llm_provider == "gemini":
                if not self.gemini_vision.is_available():
                    raise VisionProviderError("GEMINI_API_KEY is not configured on the server.", status_code=503)
                gen_res = self.gemini_vision.generate_answer(
                    question=request.question,
                    context_text=context_str,
                    image_paths=image_paths,
                )
                raw_answer = gen_res["answer"]
                model_used = gen_res["model"]
                is_vision = gen_res["is_vision_enabled"]
            else:
                gen_res = self.mock_vision.generate_answer(
                    question=request.question,
                    context_text=context_str,
                    image_paths=image_paths,
                )
                raw_answer = gen_res["answer"]
                model_used = gen_res["model"]
                is_vision = gen_res["is_vision_enabled"]
        else:
            # Standard Text Generation (Gemini Text, Ollama, OpenAI, or Mock)
            built_prompt = prompt_builder.build_prompt(
                template_id=request.prompt_template,
                question=request.question,
                context=context_str,
            )
            provider = llm_registry.get_provider(request.llm_provider)
            gen_req = GenerationRequest(
                prompt=built_prompt.user_prompt,
                system_prompt=built_prompt.system_prompt,
                temperature=request.temperature,
                max_tokens=request.max_output_tokens,
                metadata={"model": request.llm_model} if request.llm_model else {},
            )
            gen_resp = provider.generate(gen_req)
            raw_answer = gen_resp.content
            model_used = gen_resp.model_id
            is_vision = False

        timing.generation_latency_ms = round((time.perf_counter() - t_gen_start) * 1000.0, 2)

        # 5. Citation Validation
        t_cite_start = time.perf_counter()
        citations, invalid_cits, uncited = self.validate_citations(
            generated_answer=raw_answer,
            evidence_items=evidence_items,
        )
        timing.citation_validation_ms = round((time.perf_counter() - t_cite_start) * 1000.0, 2)
        timing.total_latency_ms = round((time.perf_counter() - t_total_start) * 1000.0, 2)

        if invalid_cits:
            warnings.append(f"Model generated ungrounded citation(s): {', '.join(invalid_cits)}")

        # Clean evidence items for public response (strip any internal disk paths)
        clean_evidence = []
        for it in evidence_items:
            cleaned = it.model_copy()
            cleaned._local_image_path = None
            clean_evidence.append(cleaned)

        return UnifiedRAGQueryResponse(
            trace_id=trace_id,
            question=request.question,
            answer=raw_answer,
            routing=routing_decision,
            evidence_items=clean_evidence,
            citations=citations,
            invalid_citations=invalid_cits,
            uncited_sources=uncited,
            model_used=model_used,
            is_vision_enabled=is_vision,
            timing_metrics=timing,
            assembled_prompt=context_str if request.include_assembled_prompt else None,
            warnings=warnings,
        )

    async def execute_query_stream(
        self,
        session: Session,
        request: UnifiedRAGQueryRequest,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Server-Sent Events (SSE) streaming yielding status, routing, sources, chunk, and done events.
        """
        trace_id = str(uuid.uuid4())
        yield {"event": "status", "data": {"stage": "routing", "message": "Analyzing query semantics and index capabilities..."}}

        routing_decision = self.router.route_query(
            session=session,
            question=request.question,
            document_id=request.document_id,
            page_number=request.page_number,
            mode=request.mode,
            manual_routes=request.manual_routes,
        )
        yield {"event": "routing", "data": routing_decision.model_dump()}

        yield {
            "event": "status",
            "data": {
                "stage": "retrieval",
                "message": f"Dispatching retrieval across routes: {', '.join(routing_decision.selected_routes)}...",
            },
        }

        evidence_items, per_route_latency = self.retriever.retrieve(
            session=session,
            query=request.question,
            routes=routing_decision.selected_routes,
            document_id=request.document_id,
            page_number=routing_decision.effective_page_number,
            top_k=request.top_k,
        )

        clean_evidence = [it.model_dump() for it in evidence_items]
        yield {"event": "sources", "data": {"trace_id": trace_id, "sources": clean_evidence}}

        if not evidence_items:
            yield {
                "event": "chunk",
                "data": {"content": "No relevant documentary evidence could be retrieved for this question."},
            }
            yield {"event": "done", "data": {"trace_id": trace_id, "is_vision_enabled": False, "citations": []}}
            return

        yield {"event": "status", "data": {"stage": "generation", "message": "Synthesizing answer with source grounding..."}}

        # Run generation
        is_visual_required = (
            "multimodal_visual" in routing_decision.selected_routes
            or any(it.evidence_type in ("image", "figure", "visual", "diagram") for it in evidence_items[:2])
        )
        context_str, image_paths = self.build_unified_context(
            evidence_items,
            attach_images=is_visual_required,
            target_page_number=routing_decision.effective_page_number,
        )
        built_prompt = prompt_builder.build_prompt("grounded", question=request.question, context=context_str)
        provider = llm_registry.get_provider(request.llm_provider)

        gen_req = GenerationRequest(
            prompt=built_prompt.user_prompt,
            system_prompt=built_prompt.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_output_tokens,
            metadata={"model": request.llm_model} if request.llm_model else {},
        )

        accumulated = ""
        for chunk in provider.generate_stream(gen_req):
            if chunk.token:
                accumulated += chunk.token
                yield {"event": "chunk", "data": {"content": chunk.token}}

        citations, invalid, uncited = self.validate_citations(accumulated, evidence_items)
        yield {
            "event": "done",
            "data": {
                "trace_id": trace_id,
                "citations": [c.model_dump() for c in citations],
                "invalid_citations": invalid,
                "uncited_sources": uncited,
                "model_used": request.llm_model or "default",
                "is_vision_enabled": False,
            },
        }


unified_generation_service = UnifiedGenerationService()
