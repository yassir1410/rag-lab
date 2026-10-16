import math
import uuid
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import select, func, desc

from app.models.document import Document
from app.models.chunking import ChunkingExperiment, DocumentChunk
from app.models.embeddings import EmbeddingExperiment, ChunkEmbedding
from app.rag.embeddings.registry import embedding_registry
from app.rag.embeddings.similarity import (
    cosine_similarity,
    dot_product,
    euclidean_distance,
    pairwise_similarity_matrix,
)
from app.rag.embeddings.normalization import compute_l2_norm, l2_normalize
from app.rag.embeddings.tokenization import TokenizerInspector
from app.rag.embeddings.dimensionality import dimensionality_reducer
from app.rag.embeddings.evaluation import EmbeddingBenchmarkSuite
from app.schemas.embeddings import (
    EmbeddingExperimentCreate,
    ProjectEmbeddingsRequest,
)


class EmbeddingService:
    """
    High-level orchestration service for embedding operations, token analysis,
    similarity computations, dimensionality reductions, and PostgreSQL persistence.
    """

    def __init__(self):
        self.registry = embedding_registry
        self.reducer = dimensionality_reducer

    def list_models(self) -> List[Dict[str, Any]]:
        return self.registry.list_models()

    def tokenize_text(self, text: str, model_id: str) -> Dict[str, Any]:
        spec = self.registry.get_spec(model_id)
        inspector = TokenizerInspector(model_id=spec.model_id, max_seq_length=spec.max_seq_length)
        return inspector.inspect(text)

    def calculate_similarity(
        self,
        text_a: str,
        text_b: str,
        model_id: str,
        normalize: bool = True,
    ) -> Dict[str, Any]:
        provider = self.registry.get_provider(model_id)
        encode_result = provider.encode([text_a, text_b], normalize_embeddings=normalize)
        vec_a, vec_b = encode_result.vectors[0], encode_result.vectors[1]

        cos_sim = cosine_similarity(vec_a, vec_b)
        dot_prod = dot_product(vec_a, vec_b)
        eucl_dist = euclidean_distance(vec_a, vec_b)
        norm_a = compute_l2_norm(vec_a)
        norm_b = compute_l2_norm(vec_b)

        explanation = (
            f"Cosine similarity is {cos_sim:.4f} (bounded in [-1, 1]). "
            f"Euclidean distance is {eucl_dist:.4f} (0 is identical, higher is more distant). "
            f"Dot product is {dot_prod:.4f}."
        )
        if normalize:
            explanation += " Vectors are L2 normalized (||v|| = 1.0), meaning dot product equals cosine similarity."
        else:
            explanation += " Vectors are raw/unnormalized; magnitude influences dot product."

        return {
            "text_a": text_a,
            "text_b": text_b,
            "model_id": model_id,
            "cosine_similarity": round(cos_sim, 4),
            "dot_product": round(dot_prod, 4),
            "euclidean_distance": round(eucl_dist, 4),
            "vector_a_norm": round(norm_a, 4),
            "vector_b_norm": round(norm_b, 4),
            "explanation": explanation,
        }

    def calculate_matrix(
        self,
        texts: List[str],
        model_id: str,
        metric: str = "cosine",
        normalize: bool = True,
    ) -> Dict[str, Any]:
        if not texts:
            return {"texts": [], "model_id": model_id, "metric": metric, "matrix": [], "is_normalized": normalize}

        provider = self.registry.get_provider(model_id)
        encode_result = provider.encode(texts, normalize_embeddings=normalize)
        matrix = pairwise_similarity_matrix(encode_result.vectors, metric=metric)

        return {
            "texts": texts,
            "model_id": model_id,
            "metric": metric,
            "matrix": matrix,
            "is_normalized": normalize,
        }

    def generate_embeddings(
        self,
        texts: List[str],
        model_id: str,
        batch_size: int = 32,
        normalize: bool = True,
        is_query: bool = False,
    ) -> Dict[str, Any]:
        provider = self.registry.get_provider(model_id)
        result = provider.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            is_query=is_query,
        )

        items = []
        for idx, (t, vec) in enumerate(zip(texts, result.vectors)):
            norm = compute_l2_norm(vec)
            items.append({
                "index": idx,
                "text": t,
                "token_count": len(t.split()),
                "dimension": len(vec),
                "norm": round(norm, 4),
                "vector_preview": [round(x, 4) for x in vec[:8]],
                "full_vector": [round(x, 6) for x in vec],
            })

        return {
            "model_id": result.model_id,
            "dimension": result.dimension,
            "is_normalized": result.is_normalized,
            "total_items": len(texts),
            "total_tokens": result.total_tokens,
            "latency_ms": result.latency_ms,
            "items": items,
        }

    def run_benchmark(self, model_id: str) -> Dict[str, Any]:
        provider = self.registry.get_provider(model_id)
        suite = EmbeddingBenchmarkSuite(provider)
        return suite.run_benchmark()

    def project(
        self,
        db: Session,
        req: ProjectEmbeddingsRequest,
    ) -> Dict[str, Any]:
        vectors: List[List[float]] = []
        labels: List[str] = []
        metadata_list: List[Dict[str, Any]] = []

        # Option A: From existing embedding experiment
        if req.embedding_experiment_id:
            emb_exp = db.get(EmbeddingExperiment, req.embedding_experiment_id)
            if not emb_exp:
                raise ValueError(f"Embedding experiment {req.embedding_experiment_id} not found.")

            stmt = (
                select(ChunkEmbedding)
                .where(ChunkEmbedding.experiment_id == req.embedding_experiment_id)
                .order_by(ChunkEmbedding.sequence_index)
            )
            chunk_embs = db.scalars(stmt).all()
            for ce in chunk_embs:
                if ce.vector is not None:
                    vectors.append([float(x) for x in ce.vector])
                    labels.append(ce.text_snippet[:80])
                    meta = {"chunk_id": str(ce.chunk_id) if ce.chunk_id else None, "seq": ce.sequence_index}
                    if ce.chunk and ce.chunk.metadata_json:
                        meta.update(ce.chunk.metadata_json)
                    metadata_list.append(meta)

        # Option B: From existing chunking experiment
        elif req.chunking_experiment_id:
            chunk_exp = db.get(ChunkingExperiment, req.chunking_experiment_id)
            if not chunk_exp:
                raise ValueError(f"Chunking experiment {req.chunking_experiment_id} not found.")

            stmt = (
                select(DocumentChunk)
                .where(DocumentChunk.experiment_id == req.chunking_experiment_id)
                .order_by(DocumentChunk.sequence_index)
            )
            chunks = db.scalars(stmt).all()
            texts = [c.content for c in chunks]
            provider = self.registry.get_provider(req.model_id)
            encode_res = provider.encode(texts, normalize_embeddings=True)
            vectors = encode_res.vectors
            labels = [c.content[:80] for c in chunks]
            for c in chunks:
                metadata_list.append({
                    "chunk_id": str(c.id),
                    "seq": c.sequence_index,
                    "token_count": c.token_count,
                    "char_count": c.char_count,
                    **(c.metadata_json or {}),
                })

        # Option C: Custom input texts
        elif req.texts:
            provider = self.registry.get_provider(req.model_id)
            encode_res = provider.encode(req.texts, normalize_embeddings=True)
            vectors = encode_res.vectors
            labels = [t[:80] for t in req.texts]
            metadata_list = [{"seq": i} for i in range(len(req.texts))]
        else:
            raise ValueError("Must provide either texts, chunking_experiment_id, or embedding_experiment_id.")

        if not vectors:
            return {
                "algorithm": req.algorithm,
                "n_components": req.n_components,
                "model_id": req.model_id,
                "points": [],
                "warnings": ["No vectors found to project."],
            }

        proj_result = self.reducer.project(
            vectors=vectors,
            algorithm=req.algorithm,
            n_components=req.n_components,
            perplexity=req.perplexity,
            n_neighbors=req.n_neighbors,
            min_dist=req.min_dist,
        )

        projected_points = []
        for idx, (coords, lbl, meta) in enumerate(zip(proj_result["points"], labels, metadata_list)):
            projected_points.append({
                "index": idx,
                "coordinates": coords,
                "label": lbl,
                "metadata": meta,
            })

        return {
            "algorithm": req.algorithm,
            "n_components": req.n_components,
            "model_id": req.model_id,
            "points": projected_points,
            "explained_variance_ratio": proj_result.get("explained_variance_ratio"),
            "warnings": proj_result.get("warnings", []),
        }

    def create_experiment(
        self,
        db: Session,
        req: EmbeddingExperimentCreate,
    ) -> EmbeddingExperiment:
        spec = self.registry.get_spec(req.model_id)
        provider = self.registry.get_provider(req.model_id)

        # 1. Fetch chunks to embed
        chunks_to_embed: List[Tuple[Optional[uuid.UUID], str, int, int]] = []

        if req.chunking_experiment_id:
            chunk_exp = db.get(ChunkingExperiment, req.chunking_experiment_id)
            if not chunk_exp:
                raise ValueError(f"Chunking experiment {req.chunking_experiment_id} not found.")
            doc_id = chunk_exp.document_id

            stmt = (
                select(DocumentChunk)
                .where(DocumentChunk.experiment_id == req.chunking_experiment_id)
                .order_by(DocumentChunk.sequence_index)
            )
            chunks = db.scalars(stmt).all()
            for c in chunks:
                chunks_to_embed.append((c.id, c.content, c.sequence_index, c.token_count))
        elif req.document_id:
            doc = db.get(Document, req.document_id)
            if not doc:
                raise ValueError(f"Document {req.document_id} not found.")
            doc_id = req.document_id

            # Embed sections
            for s_idx, sec in enumerate(doc.sections):
                chunks_to_embed.append((None, sec.clean_text, s_idx, len(sec.clean_text.split())))
        else:
            raise ValueError("Must provide either chunking_experiment_id or document_id.")

        if not chunks_to_embed:
            raise ValueError("No text units found to generate embeddings.")

        texts = [item[1] for item in chunks_to_embed]

        # 2. Run batched model encoding
        encode_result = provider.encode(
            texts=texts,
            batch_size=req.batch_size,
            normalize_embeddings=req.normalize,
            is_query=False,
        )

        # 3. Create EmbeddingExperiment row
        experiment = EmbeddingExperiment(
            id=uuid.uuid4(),
            name=req.name or f"{spec.display_name} - {len(texts)} chunks",
            document_id=doc_id,
            chunking_experiment_id=req.chunking_experiment_id,
            model_id=req.model_id,
            provider=spec.provider,
            vector_dimension=encode_result.dimension,
            is_normalized=req.normalize,
            parameters={
                "batch_size": req.batch_size,
                "normalize": req.normalize,
                "max_seq_length": spec.max_seq_length,
            },
            metrics={
                "total_items": len(texts),
                "total_tokens": encode_result.total_tokens,
                "latency_ms": encode_result.latency_ms,
                "avg_latency_per_item_ms": round(
                    encode_result.latency_ms / max(len(texts), 1), 3
                ),
            },
        )
        db.add(experiment)

        # 4. Insert ChunkEmbedding rows
        for idx, (chunk_id, content, seq_idx, tok_count) in enumerate(chunks_to_embed):
            vec = encode_result.vectors[idx]
            ce = ChunkEmbedding(
                id=uuid.uuid4(),
                experiment_id=experiment.id,
                chunk_id=chunk_id,
                sequence_index=seq_idx,
                text_snippet=content[:300],
                token_count=tok_count,
                vector=vec,
            )
            db.add(ce)

        db.commit()
        db.refresh(experiment)
        return experiment

    def list_experiments(
        self,
        db: Session,
        document_id: Optional[uuid.UUID] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> Tuple[int, List[Dict[str, Any]]]:
        query = select(EmbeddingExperiment)
        count_query = select(func.count(EmbeddingExperiment.id))

        if document_id:
            query = query.where(EmbeddingExperiment.document_id == document_id)
            count_query = count_query.where(EmbeddingExperiment.document_id == document_id)

        total = db.scalar(count_query) or 0
        experiments = (
            db.scalars(
                query.order_by(desc(EmbeddingExperiment.created_at))
                .limit(limit)
                .offset(offset)
            ).all()
        )

        results = []
        for exp in experiments:
            emb_count = (
                db.scalar(
                    select(func.count(ChunkEmbedding.id)).where(
                        ChunkEmbedding.experiment_id == exp.id
                    )
                )
                or 0
            )
            results.append({
                "id": exp.id,
                "name": exp.name,
                "document_id": exp.document_id,
                "chunking_experiment_id": exp.chunking_experiment_id,
                "model_id": exp.model_id,
                "provider": exp.provider,
                "vector_dimension": exp.vector_dimension,
                "is_normalized": exp.is_normalized,
                "parameters": exp.parameters,
                "metrics": exp.metrics,
                "created_at": exp.created_at,
                "embeddings_count": emb_count,
            })

        return total, results

    def get_experiment(
        self,
        db: Session,
        experiment_id: uuid.UUID,
    ) -> Optional[Dict[str, Any]]:
        exp = db.get(EmbeddingExperiment, experiment_id)
        if not exp:
            return None

        stmt = (
            select(ChunkEmbedding)
            .where(ChunkEmbedding.experiment_id == experiment_id)
            .order_by(ChunkEmbedding.sequence_index)
            .limit(50)
        )
        sample_embeddings = db.scalars(stmt).all()

        samples = []
        for ce in sample_embeddings:
            samples.append({
                "id": ce.id,
                "chunk_id": ce.chunk_id,
                "sequence_index": ce.sequence_index,
                "text_snippet": ce.text_snippet,
                "token_count": ce.token_count,
                "vector_preview": list(ce.vector[:8]) if ce.vector is not None else [],
            })

        emb_count = (
            db.scalar(
                select(func.count(ChunkEmbedding.id)).where(
                    ChunkEmbedding.experiment_id == exp.id
                )
            )
            or 0
        )

        return {
            "id": exp.id,
            "name": exp.name,
            "document_id": exp.document_id,
            "chunking_experiment_id": exp.chunking_experiment_id,
            "model_id": exp.model_id,
            "provider": exp.provider,
            "vector_dimension": exp.vector_dimension,
            "is_normalized": exp.is_normalized,
            "parameters": exp.parameters,
            "metrics": exp.metrics,
            "created_at": exp.created_at,
            "embeddings_count": emb_count,
            "sample_embeddings": samples,
        }

    def delete_experiment(self, db: Session, experiment_id: uuid.UUID) -> bool:
        exp = db.get(EmbeddingExperiment, experiment_id)
        if not exp:
            return False
        db.delete(exp)
        db.commit()
        return True
