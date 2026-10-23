import hashlib
import re
import tempfile
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timezone

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.errors import AppError
from app.schemas.models import Chunk, DocumentInfo


class Ingestion:
    def __init__(self, settings):
        self.settings = settings
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap,
            add_start_index=True,
        )

    def validate(self, filename, mime, data):
        # Both POSIX and Windows separators are handled before any filesystem use.
        safe = re.split(r"[/\\]", filename or "")[-1]
        safe = re.sub(r"[^\w .()-]", "_", safe)[:180]
        suffix = Path(safe).suffix.lower()
        allowed = {".pdf": {"application/pdf"}, ".txt": {"text/plain"}}
        if suffix not in allowed:
            raise AppError(415, "unsupported_file", "Only PDF and UTF-8 TXT files are supported.")
        if mime not in allowed[suffix]:
            raise AppError(415, "invalid_mime", "File content type does not match its extension.")
        if not data:
            raise AppError(422, "empty_file", "The uploaded file is empty.")
        if len(data) > self.settings.max_upload_mb * 1024 * 1024:
            raise AppError(413, "file_too_large", f"Maximum file size is {self.settings.max_upload_mb} MB.")
        if suffix == ".pdf" and not data.startswith(b"%PDF-"):
            raise AppError(422, "invalid_pdf", "The file does not have a valid PDF header.")
        return safe, suffix, hashlib.sha256(data).hexdigest()

    def extract(self, filename, suffix, digest, data):
        # Temporary server-generated paths only; originals are not retained.
        try:
            with tempfile.TemporaryDirectory(prefix="docmind-") as folder:
                path = Path(folder) / f"upload{suffix}"
                path.write_bytes(data)
                loader = (PyPDFLoader(str(path)) if suffix == ".pdf" else
                          TextLoader(str(path), encoding="utf-8", autodetect_encoding=False))
                pages = []
                length = 0
                for page in loader.lazy_load():
                    length += len(page.page_content)
                    if length > self.settings.max_extracted_chars:
                        raise AppError(413, "extracted_text_too_large", "Extracted text exceeds the processing limit.")
                    pages.append(page)
        except AppError:
            raise
        except Exception as exc:
            raise AppError(422, "extraction_failed", "Cannot read this file. Use a readable PDF or UTF-8 TXT.") from exc
        split = self.splitter.split_documents(pages)
        if not split or not any(part.page_content.strip() for part in split):
            raise AppError(422, "no_text", "No readable text found. Scanned PDFs require OCR, which is not included.")
        if len(split) > self.settings.max_chunks_per_document:
            raise AppError(413, "too_many_chunks", "Document exceeds the chunk limit.")
        document_id = uuid4()
        chunks = [Chunk(id=uuid4(), document_id=document_id, text=part.page_content,
                        page=part.metadata["page"] + 1 if "page" in part.metadata else None)
                  for part in split if part.page_content.strip()]
        info = DocumentInfo(id=document_id, filename=filename, sha256=digest,
                            size_bytes=len(data), chunk_count=len(chunks),
                            created_at=datetime.now(timezone.utc))
        return info, chunks
