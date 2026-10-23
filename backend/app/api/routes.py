from uuid import UUID

from fastapi import APIRouter, File, Request, Response, UploadFile

from app.schemas.models import (
    ChatRequest, ChatResponse, ChatSummary, Conversation, DocumentInfo,
    HealthResponse, UploadResult,
)

router = APIRouter(prefix="/api")


@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    state = request.app.state
    with state.store.lock:
        missing = state.providers.missing()
        return HealthResponse(
            status="storage_error" if state.store.error else "configuration_required" if missing else "ok",
            configured=not missing, missing=missing,
            documents=len(state.store.documents), chunks=len(state.store.chunks),
            llm_model=state.settings.llm_model, embedding_model=state.settings.embedding_model,
            storage_ready=not state.store.error, message=state.store.error,
        )


@router.post("/documents/upload", response_model=UploadResult, status_code=201)
def upload(request: Request, response: Response, file: UploadFile = File(...)):
    state = request.app.state
    try:
        # Bounded application read; reverse proxy must also cap multipart request size.
        data = file.file.read(state.settings.max_upload_mb * 1024 * 1024 + 1)
        filename, suffix, digest = state.ingestion.validate(file.filename, file.content_type, data)
        with state.store.lock:
            state.store.require_ready()
            duplicate = state.store.duplicate(digest)
            if duplicate:
                response.status_code = 200
                return UploadResult(document=duplicate, duplicate=True)
            document, chunks = state.ingestion.extract(filename, suffix, digest, data)
            state.store.add(document, chunks)
            return UploadResult(document=document, duplicate=False)
    finally:
        file.file.close()


@router.get("/documents", response_model=list[DocumentInfo])
def documents(request: Request):
    store = request.app.state.store
    with store.lock:
        store.require_ready()
        return list(store.documents)


@router.get("/documents/{document_id}", response_model=DocumentInfo)
def document(document_id: UUID, request: Request):
    store = request.app.state.store
    with store.lock:
        return store.document(document_id)


@router.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: UUID, request: Request):
    store = request.app.state.store
    with store.lock:
        store.delete_document(document_id)
    return Response(status_code=204)


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request):
    return request.app.state.rag.answer(payload)


@router.get("/chats", response_model=list[ChatSummary])
def chats(request: Request):
    store = request.app.state.store
    with store.lock:
        store.require_ready()
        return [ChatSummary(id=c.id, title=c.title, updated_at=c.updated_at, message_count=len(c.messages))
                for c in sorted(store.conversations, key=lambda c: c.updated_at, reverse=True)]


@router.get("/chats/{chat_id}", response_model=Conversation)
def conversation(chat_id: UUID, request: Request):
    store = request.app.state.store
    with store.lock:
        return store.chat(chat_id)


@router.delete("/chats/{chat_id}", status_code=204)
def delete_chat(chat_id: UUID, request: Request):
    store = request.app.state.store
    with store.lock:
        store.delete_chat(chat_id)
    return Response(status_code=204)
