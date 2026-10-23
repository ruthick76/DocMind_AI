"""Atomic local snapshots: native FAISS binary + validated JSON, never pickle."""
import hashlib
import json
import os
import re
import shutil
import threading
from pathlib import Path
from uuid import uuid4

import faiss
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from pydantic import Field

from app.core.errors import AppError
from app.schemas.models import Chunk, Conversation, DocumentInfo, StrictModel


class Snapshot(StrictModel):
    version: int = Field(default=1, ge=1, le=1)
    embedding_signature: str
    documents: list[DocumentInfo]
    chunks: list[Chunk]
    index_sha256: str | None


class ChatFile(StrictModel):
    version: int = Field(default=1, ge=1, le=1)
    conversations: list[Conversation]


def atomic_json(path: Path, value):
    """Fsync data before replace; readers see either complete old or new JSON."""
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def fsync_dir(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class Store:
    def __init__(self, settings, providers):
        self.settings, self.providers = settings, providers
        self.root = settings.data_dir
        self.lock = threading.RLock()
        self.documents, self.chunks, self.conversations = [], [], []
        self.index = None
        self.generation = None
        self.error = None
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            self.snapshots = self.root / "snapshots"
            self.snapshots.mkdir(exist_ok=True)
            self._load()
        except Exception:
            # Fail closed: never replace corrupt state with a new empty library.
            self.error = "Saved storage is invalid, incompatible, or unreadable. Restore a trusted backup or use a new data directory."

    def require_ready(self):
        if self.error:
            raise AppError(503, "storage_unavailable", self.error)

    def _load(self):
        pointer = self.root / "current.json"
        if pointer.exists():
            current = json.loads(pointer.read_text())
            generation = current["generation"]
            if set(current) != {"generation"} or not re.fullmatch(r"[a-f0-9]{32}", generation):
                raise ValueError("Invalid snapshot pointer")
            folder = self.snapshots / generation
            state = Snapshot.model_validate_json((folder / "metadata.json").read_text())
            if state.embedding_signature != self.settings.embedding_signature:
                raise ValueError("Embedding configuration differs from saved index")
            ids = {str(document.id) for document in state.documents}
            if len(ids) != len(state.documents) or len({chunk.id for chunk in state.chunks}) != len(state.chunks):
                raise ValueError("Duplicate IDs")
            if any(str(chunk.document_id) not in ids for chunk in state.chunks):
                raise ValueError("Orphan chunks")
            for document in state.documents:
                if sum(c.document_id == document.id for c in state.chunks) != document.chunk_count:
                    raise ValueError("Invalid document chunk count")
            if state.chunks:
                path = folder / "index.faiss"
                if path.stat().st_size > 512 * 1024 * 1024:
                    raise ValueError("Index exceeds local storage limit")
                if hashlib.sha256(path.read_bytes()).hexdigest() != state.index_sha256:
                    raise ValueError("Index checksum mismatch")
                index = faiss.read_index(str(path))
                if not isinstance(index, faiss.IndexFlatIP) or index.ntotal != len(state.chunks) or not 1 <= index.d <= 65536:
                    raise ValueError("Invalid index structure")
                self.index = index
            elif state.documents or state.index_sha256 is not None:
                raise ValueError("Inconsistent empty snapshot")
            self.documents, self.chunks = state.documents, state.chunks
            self.generation = generation
        chats_path = self.root / "chats.json"
        if chats_path.exists():
            self.conversations = ChatFile.model_validate_json(chats_path.read_text()).conversations
            if len({chat.id for chat in self.conversations}) != len(self.conversations):
                raise ValueError("Duplicate chat IDs")

    def _commit(self, documents, chunks, index):
        """Commit the pointer last; unpublished generations cannot become active."""
        generation = uuid4().hex
        folder = self.snapshots / generation
        folder.mkdir()
        try:
            digest = None
            if index is not None:
                path = folder / "index.faiss"
                faiss.write_index(index, str(path))
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            state = Snapshot(embedding_signature=self.settings.embedding_signature,
                             documents=documents, chunks=chunks, index_sha256=digest)
            atomic_json(folder / "metadata.json", state.model_dump(mode="json"))
            fsync_dir(self.snapshots)
            atomic_json(self.root / "current.json", {"generation": generation})
        except OSError as exc:
            # A rename may have succeeded before fsync failed. Retain the files
            # and fail closed until restart rather than deleting a live snapshot.
            self.error = "Storage write failed. Check disk space and permissions, then restart the backend."
            raise AppError(503, "storage_unavailable", self.error) from exc
        previous = self.generation
        self.documents, self.chunks, self.index = documents, chunks, index
        self.generation = generation
        # Keep the actual previous committed snapshot, not the newest orphan.
        try:
            for old in self.snapshots.iterdir():
                if (old.is_dir() and re.fullmatch(r"[a-f0-9]{32}", old.name)
                        and old.name not in {generation, previous}):
                    shutil.rmtree(old, ignore_errors=True)
        except OSError:
            pass  # Cleanup failure cannot invalidate a successfully committed update.

    def duplicate(self, digest):
        return next((d for d in self.documents if d.sha256 == digest), None)

    def add(self, document, chunks):
        self.require_ready()
        try:
            vectors = self.providers.embeddings.embed_documents([c.text for c in chunks])
            matrix = self._vectors(vectors, len(chunks))
            if self.index is not None and self.index.d != matrix.shape[1]:
                raise ValueError("Embedding dimension changed")
            total = len(self.chunks) + len(chunks)
            if total * matrix.shape[1] * 4 + 1024 > 512 * 1024 * 1024:
                raise AppError(413, "index_limit", "The local index has reached its 512 MB limit.")
            index = faiss.IndexFlatIP(matrix.shape[1])
            if self.index is not None:
                index.add(self.index.reconstruct_n(0, self.index.ntotal))
            faiss.normalize_L2(matrix)
            index.add(matrix)
        except AppError:
            raise
        except Exception as exc:
            raise AppError(502, "embedding_failed", "Embedding provider failed or returned invalid vectors. Check model configuration and connectivity.") from exc
        self._commit([*self.documents, document], [*self.chunks, *chunks], index)

    @staticmethod
    def _vectors(vectors, expected):
        matrix = np.asarray(vectors, dtype="float32")
        if (matrix.ndim != 2 or matrix.shape[0] != expected or not 1 <= matrix.shape[1] <= 65536
                or not np.isfinite(matrix).all() or np.any(np.linalg.norm(matrix, axis=1) == 0)):
            raise ValueError("Invalid embedding vectors")
        return matrix

    def document(self, document_id):
        self.require_ready()
        result = next((d for d in self.documents if d.id == document_id), None)
        if result is None:
            raise AppError(404, "document_not_found", "Document does not exist.")
        return result

    def delete_document(self, document_id):
        self.document(document_id)
        positions = [i for i, chunk in enumerate(self.chunks) if chunk.document_id != document_id]
        index = None
        if positions:
            index = faiss.IndexFlatIP(self.index.d)
            matrix = np.stack([self.index.reconstruct(i) for i in positions])
            index.add(matrix)
        self._commit([d for d in self.documents if d.id != document_id],
                     [self.chunks[i] for i in positions], index)

    def search(self, question):
        self.require_ready()
        if self.index is None:
            return []
        # Reconstruct LangChain's docstore from safe JSON rather than pickle.
        docstore = InMemoryDocstore({str(c.id): Document(page_content=c.text,
            metadata={"document_id": str(c.document_id), "page": c.page}) for c in self.chunks})
        search = FAISS(embedding_function=self.providers.embeddings, index=self.index,
                       docstore=docstore,
                       index_to_docstore_id={i: str(c.id) for i, c in enumerate(self.chunks)},
                       distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
        # Index and query are explicitly normalized: raw inner product = cosine.
        # Read raw scores instead of a library-specific relevance conversion.
        try:
            query = self._vectors([self.providers.embeddings.embed_query(question)], 1)
            if query.shape[1] != self.index.d:
                raise ValueError("Query vector dimension differs")
            faiss.normalize_L2(query)
            results = search.similarity_search_with_score_by_vector(query[0], k=self.settings.top_k)
        except AppError:
            raise
        except Exception as exc:
            raise AppError(502, "retrieval_failed", "Could not generate a query embedding or search the index.") from exc
        return [(doc, float(score)) for doc, score in results if score >= self.settings.min_similarity]

    def chat(self, chat_id):
        self.require_ready()
        result = next((c for c in self.conversations if c.id == chat_id), None)
        if result is None:
            raise AppError(404, "chat_not_found", "Conversation does not exist.")
        return result

    def save_chat(self, chat):
        updated = [c for c in self.conversations if c.id != chat.id] + [chat]
        atomic_json(self.root / "chats.json", ChatFile(conversations=updated).model_dump(mode="json"))
        self.conversations = updated

    def delete_chat(self, chat_id):
        self.chat(chat_id)
        updated = [c for c in self.conversations if c.id != chat_id]
        atomic_json(self.root / "chats.json", ChatFile(conversations=updated).model_dump(mode="json"))
        self.conversations = updated
