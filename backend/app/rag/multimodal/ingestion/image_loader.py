import os
import uuid
from typing import Tuple, Optional
from PIL import Image

# Security: prevent image decompression bombs (limit to ~50 Megapixels)
Image.MAX_IMAGE_PIXELS = 50_000_000

ALLOWED_FORMATS = {"PNG", "JPEG", "JPG", "WEBP", "BMP"}
MAX_DIMENSION = 4096
DEFAULT_STORAGE_DIR = "/app/data/uploads/multimodal"


class SafeImageLoader:
    @staticmethod
    def validate_image_file(file_path: str) -> Tuple[bool, Optional[str]]:
        """Validate format, dimensions, and integrity without loading entire bitmap into RAM if corrupted."""
        if not os.path.exists(file_path):
            return False, f"File not found: {file_path}"
        try:
            with Image.open(file_path) as img:
                fmt = (img.format or "").upper()
                if fmt not in ALLOWED_FORMATS:
                    return False, f"Unsupported image format '{fmt}'. Allowed: {ALLOWED_FORMATS}"
                w, h = img.size
                if w > MAX_DIMENSION or h > MAX_DIMENSION:
                    return False, f"Image dimensions {w}x{h} exceed maximum allowed {MAX_DIMENSION}x{MAX_DIMENSION}"
                if w <= 0 or h <= 0:
                    return False, "Invalid image dimensions"
                img.verify()
            return True, None
        except Exception as e:
            return False, f"Corrupted or invalid image: {str(e)}"

    @staticmethod
    def load_rgb_image(file_path: str, max_size: Tuple[int, int] = (1024, 1024)) -> Image.Image:
        """Load image in RGB mode with thumbnail scaling for safe memory footprint."""
        with Image.open(file_path) as img:
            rgb_img = img.convert("RGB")
            rgb_img.thumbnail(max_size, Image.Resampling.LANCZOS)
            return rgb_img

    @staticmethod
    def save_page_image(
        image_bytes: bytes,
        document_id: uuid.UUID,
        page_number: int,
        storage_dir: str = DEFAULT_STORAGE_DIR,
    ) -> str:
        """Save rendered page image safely to disk with sanitized filename."""
        doc_dir = os.path.join(storage_dir, str(document_id))
        os.makedirs(doc_dir, exist_ok=True)
        filename = f"page_{page_number:03d}.png"
        target_path = os.path.join(doc_dir, filename)
        with open(target_path, "wb") as f:
            f.write(image_bytes)
        return target_path

    @staticmethod
    def crop_and_save_element(
        page_image_path: str,
        bbox: dict,
        document_id: uuid.UUID,
        element_id: uuid.UUID,
        storage_dir: str = DEFAULT_STORAGE_DIR,
    ) -> Optional[str]:
        """Crop a specific bounding box (figure or table) from a page image and save it."""
        if not os.path.exists(page_image_path):
            return None
        try:
            with Image.open(page_image_path) as img:
                w, h = img.size
                x0 = max(0, min(w, int(bbox.get("x0", 0))))
                y0 = max(0, min(h, int(bbox.get("y0", 0))))
                x1 = max(0, min(w, int(bbox.get("x1", w))))
                y1 = max(0, min(h, int(bbox.get("y1", h))))
                if x1 <= x0 or y1 <= y0:
                    return None
                crop_img = img.crop((x0, y0, x1, y1))
                doc_dir = os.path.join(storage_dir, str(document_id), "elements")
                os.makedirs(doc_dir, exist_ok=True)
                target_path = os.path.join(doc_dir, f"elem_{element_id}.png")
                crop_img.save(target_path, "PNG")
                return target_path
        except Exception:
            return None
