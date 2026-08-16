import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from pypdf import PdfReader

logger = logging.getLogger(__name__)

TEXT_EXTENSIONS = {".txt", ".md", ".pdf"}
DEFAULT_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | DEFAULT_IMAGE_EXTENSIONS

EXTRACTED_IMAGES_DIRNAME = "_extracted_images"


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    source_path: str
    extension: str
    file_type: str
    chunk_index: int
    chunk_type: str = "text"
    image_path: Optional[str] = None
    image_bytes: Optional[bytes] = field(default=None, repr=False)


def _read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")


def load_documents(kb_dir: str) -> List[Tuple[str, str, str, str]]:
    base = Path(kb_dir)
    if not base.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {kb_dir}")

    documents = []
    for path in sorted(base.glob("**/*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        if EXTRACTED_IMAGES_DIRNAME in path.parts:
            continue
        text = _read_file(path).strip()
        if text:
            rel_path = str(path.relative_to(base))
            ext = path.suffix.lower().lstrip(".")
            documents.append((path.name, rel_path, ext, text))
    return documents


def load_image_files(kb_dir: str, image_extensions: Optional[set] = None) -> List[Tuple[str, str, str, bytes]]:
    """Standalone image files under the knowledge base directory (excludes
    images already extracted from PDFs, which live under EXTRACTED_IMAGES_DIRNAME
    and are discovered separately by extract_pdf_images)."""
    extensions = image_extensions or DEFAULT_IMAGE_EXTENSIONS
    base = Path(kb_dir)
    if not base.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {kb_dir}")

    images = []
    for path in sorted(base.glob("**/*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if EXTRACTED_IMAGES_DIRNAME in path.parts:
            continue
        rel_path = str(path.relative_to(base))
        ext = path.suffix.lower().lstrip(".")
        images.append((path.name, rel_path, ext, path.read_bytes()))
    return images


def extract_pdf_images(kb_dir: str) -> List[Tuple[str, str, str, bytes]]:
    """Extracts embedded images from every PDF in the knowledge base, writing each
    one to <kb_dir>/_extracted_images/ so a real on-disk file exists for later
    query-time retrieval, and returns (name, rel_path_under_kb, ext, bytes) tuples."""
    base = Path(kb_dir)
    if not base.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {kb_dir}")

    out_dir = base / EXTRACTED_IMAGES_DIRNAME
    extracted: List[Tuple[str, str, str, bytes]] = []

    for pdf_path in sorted(base.glob("**/*.pdf")):
        if EXTRACTED_IMAGES_DIRNAME in pdf_path.parts:
            continue
        try:
            reader = PdfReader(str(pdf_path))
        except Exception:
            logger.warning("Could not open PDF for image extraction: %s", pdf_path, exc_info=True)
            continue

        stem = pdf_path.stem
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                page_images = list(page.images)
            except Exception:
                logger.warning("Could not read images on page %s of %s", page_num, pdf_path, exc_info=True)
                continue

            for idx, image_file in enumerate(page_images, start=1):
                try:
                    data = image_file.data
                    name_ext = Path(image_file.name).suffix.lower().lstrip(".") or "png"
                    out_name = f"{stem}_p{page_num}_{idx}.{name_ext}"
                    out_path = out_dir / out_name
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path.write_bytes(data)
                    rel_path = str(out_path.relative_to(base))
                    extracted.append((out_name, rel_path, name_ext, data))
                except Exception:
                    logger.warning(
                        "Skipping unreadable embedded image on page %s of %s", page_num, pdf_path, exc_info=True
                    )
                    continue

    return extracted


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = " ".join(text.split())
    if not text:
        return []

    chunks = []
    start = 0
    step = max(chunk_size - overlap, 1)
    while start < len(text):
        end = start + chunk_size
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        start += step
    return chunks


def build_chunks(
    kb_dir: str,
    chunk_size: int,
    overlap: int,
    image_extensions: Optional[set] = None,
) -> List[Chunk]:
    chunks: List[Chunk] = []

    for source, source_path, extension, text in load_documents(kb_dir):
        for i, piece in enumerate(chunk_text(text, chunk_size, overlap)):
            chunks.append(
                Chunk(
                    id=f"{source_path}::{i}",
                    text=piece,
                    source=source,
                    source_path=source_path,
                    extension=extension,
                    file_type=extension,
                    chunk_index=i,
                    chunk_type="text",
                )
            )

    for source, source_path, extension, image_bytes in load_image_files(kb_dir, image_extensions):
        chunks.append(
            Chunk(
                id=f"{source_path}::image",
                text="",
                source=source,
                source_path=source_path,
                extension=extension,
                file_type="image",
                chunk_index=0,
                chunk_type="image",
                image_path=source_path,
                image_bytes=image_bytes,
            )
        )

    for source, source_path, extension, image_bytes in extract_pdf_images(kb_dir):
        chunks.append(
            Chunk(
                id=f"{source_path}::image",
                text="",
                source=source,
                source_path=source_path,
                extension=extension,
                file_type="image",
                chunk_index=0,
                chunk_type="image",
                image_path=source_path,
                image_bytes=image_bytes,
            )
        )

    return chunks
