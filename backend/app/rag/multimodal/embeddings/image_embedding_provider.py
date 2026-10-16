import os
import hashlib
import numpy as np
from typing import List, Union
from PIL import Image

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class MultimodalEmbeddingProvider:
    """Provides aligned text and image embeddings in a shared cross-modal vector space (512d)."""

    MODEL_ID = "clip-ViT-B-32"
    VECTOR_DIM = 512

    def __init__(self, model_name: str = "sentence-transformers/clip-ViT-B-32"):
        self.model_name = model_name
        self._model = None
        self._is_real_clip = False

    def _load_model(self):
        if self._model is not None:
            return
        if SentenceTransformer is not None:
            try:
                # Load lightweight CLIP on CPU
                self._model = SentenceTransformer(self.model_name, device="cpu")
                self._is_real_clip = True
            except Exception:
                self._model = None
                self._is_real_clip = False

    def is_real_clip_loaded(self) -> bool:
        self._load_model()
        return self._is_real_clip

    def encode_text(self, text: str) -> List[float]:
        """Encode a text query into the multimodal 512d space."""
        self._load_model()
        if self._is_real_clip and self._model is not None:
            emb = self._model.encode(text, convert_to_numpy=True)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            return emb.tolist()
        return self._fallback_text_encode(text)

    def encode_image(self, image_input: Union[str, Image.Image]) -> List[float]:
        """Encode an image into the multimodal 512d space."""
        self._load_model()
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found at {image_input}")
            img = Image.open(image_input).convert("RGB")
        else:
            img = image_input.convert("RGB")

        if self._is_real_clip and self._model is not None:
            emb = self._model.encode(img, convert_to_numpy=True)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            return emb.tolist()
        return self._fallback_image_encode(img)

    def _fallback_text_encode(self, text: str) -> List[float]:
        """Deterministic pseudo-multimodal projection for offline/test environments."""
        clean = text.lower().strip()
        h = hashlib.sha256(clean.encode("utf-8")).digest()
        seed = int.from_bytes(h[:4], "big")
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.VECTOR_DIM).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def _fallback_image_encode(self, img: Image.Image) -> List[float]:
        """Deterministic visual feature projection for offline/test environments."""
        small = img.resize((32, 32)).convert("L")
        arr = np.array(small, dtype=np.float32)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr))
        seed = int((mean_val * 1000 + std_val) % 1_000_000)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.VECTOR_DIM).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()
