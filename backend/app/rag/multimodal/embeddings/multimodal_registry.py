from typing import List, Dict, Any, Optional
from app.rag.multimodal.embeddings.image_embedding_provider import MultimodalEmbeddingProvider


class MultimodalModelRegistry:
    """Registry tracking supported multimodal models, modalities, and vector dimensions."""

    _MODELS: Dict[str, Dict[str, Any]] = {
        "clip-ViT-B-32": {
            "model_id": "clip-ViT-B-32",
            "modality": "text_and_image",
            "dimension": 512,
            "provider": "sentence-transformers",
            "description": "OpenAI CLIP ViT-B/32 aligned image and text cross-modal encoder.",
        },
        "clip-ViT-L-14": {
            "model_id": "clip-ViT-L-14",
            "modality": "text_and_image",
            "dimension": 768,
            "provider": "sentence-transformers",
            "description": "High-capacity CLIP ViT-L/14 model for fine-grained image detail.",
        },
        "gemini-1.5-flash-vision": {
            "model_id": "gemini-1.5-flash-vision",
            "modality": "vision_language",
            "dimension": 0,
            "provider": "google-genai",
            "description": "Multimodal LLM generating answers grounded in text and images.",
        },
    }

    _provider_instance: Optional[MultimodalEmbeddingProvider] = None

    @classmethod
    def get_provider(cls, model_id: str = "clip-ViT-B-32") -> MultimodalEmbeddingProvider:
        if cls._provider_instance is None:
            cls._provider_instance = MultimodalEmbeddingProvider()
        return cls._provider_instance

    @classmethod
    def list_models(cls) -> List[Dict[str, Any]]:
        provider = cls.get_provider()
        is_real = provider.is_real_clip_loaded()
        models = []
        for mid, info in cls._MODELS.items():
            entry = dict(info)
            entry["is_loaded"] = is_real if mid == "clip-ViT-B-32" else False
            entry["status"] = "active" if (mid == "clip-ViT-B-32" or mid.startswith("gemini")) else "available"
            models.append(entry)
        return models
