# Build DocMind AI step by step

Read this alongside the source files. Each stage gives the relevant code, why it exists, what to verify, and an answer you can use in a technical discussion. The repository contains the complete implementation, not just these excerpts. The stages below follow its dependency order so you can reconstruct it yourself.

The useful distinction is between **ingestion** (turning documents into searchable vectors) and **inference** (retrieving evidence and asking a model to answer). They share the same embedding space. Chat history is a third concern: persisted conversation context, not evidence.

## 1. Define the boundaries before writing features

**Code:** [`backend/app/main.py`](../backend/app/main.py), [`backend/app/api/routes.py`](../backend/app/api/routes.py), [`frontend/src/App.jsx`](../frontend/src/App.jsx).

```python
app = FastAPI(title="DocMind AI", version="1.0.0", lifespan=lifespan)
app.include_router(router)
```

The browser renders React components and sends HTTP requests. FastAPI validates those requests and invokes services. Services implement ingestion, retrieval, generation, and persistence. Neither route handlers nor React components contain model credentials.

An app factory (`create_app`) creates each application instance. Its lifespan loads storage once and initializes the service objects. Tests can create a new instance with a temporary directory and controlled provider adapters without changing runtime behavior.

**Why:** Separate boundaries let us test the real API and storage while replacing only costly external model calls. Thin routes are easier to review than a single file containing uploads, vector operations, prompts, and HTML.

**Tradeoff:** Several small modules require navigating more files. There is no repository layer, dependency injection framework, or abstract base class for every object because those would add complexity without a current need.

**Technical answer:** “I organized the backend by responsibilities, used an application factory for isolation, and kept provider calls behind one boundary. FastAPI is the application server; Node is used only because Vite needs it to build React.”

**Verify:** Inspect the directory tree in the README, then start FastAPI and request `/api/health`.

## 2. Validate configuration and define data contracts

**Code:** [`core/config.py`](../backend/app/core/config.py), [`schemas/models.py`](../backend/app/schemas/models.py).

```python
class ChatRequest(StrictModel):
    question: str = Field(min_length=1, max_length=4000)
    chat_id: UUID | None = None
```

The question is stripped, whitespace-only input is rejected, IDs must parse as UUIDs, and unknown fields are forbidden. The same approach validates persisted document metadata and conversation JSON. Each response has an explicit shape.

```python
model_config = SettingsConfigDict(
    env_prefix="DOCMIND_", env_file=BACKEND_DIR / ".env", extra="ignore"
)
```

The environment selects keys, model IDs, endpoints, upload limits, chunking, and retrieval parameters. Provider secrets use `SecretStr` so accidental object printing masks their values. The explicit `key()` helper returns a value only at the provider boundary and allows separate keys with a shared-key fallback. `SecretStr` is not encryption and does not make logging raw values safe.

**Why:** Configuration should fail early if chunk overlap is invalid or an endpoint is insecure. Request contracts prevent malformed input from becoming downstream errors. Models used for persistence make restart validation explicit.

**Tradeoff:** Schema validation checks structure, not the truth of data. A valid document record can still refer to the wrong vector; storage performs association and count checks as well.

**Technical answer:** “Pydantic protects the API and persistence boundaries, while pydantic-settings handles environment configuration. I keep credentials server-side and return safe errors without echoing original request content.”

**Verify:** Run the chat validation tests and missing-configuration test. A blank question receives 422; a missing provider key produces a specific 503 when that provider is needed.

## 3. Ingest uploads safely

**Code:** [`services/ingestion.py`](../backend/app/services/ingestion.py), upload handler in [`api/routes.py`](../backend/app/api/routes.py).

```python
safe = re.split(r"[/\\]", filename or "")[-1]
safe = re.sub(r"[^\w .()-]", "_", safe)[:180]
```

This handles Windows and POSIX separators before sanitizing a filename. More importantly, that display name never becomes the storage path: extraction uses a generated temporary path. Document IDs and snapshot directories are also server-generated.

```python
loader = (PyPDFLoader(str(path)) if suffix == ".pdf" else
          TextLoader(str(path), encoding="utf-8", autodetect_encoding=False))
```

The route reads at most the configured file limit plus one byte, then validates extension, declared MIME type, empty content, size, and the PDF header. The loader must successfully parse the content. A `.pdf` extension or MIME header alone does not prove that bytes are a PDF.

An SHA-256 digest identifies repeated bytes. We check it before extraction and embedding, so a duplicate upload avoids repeated model costs. The same bytes under another filename are still one indexed document.

**Why:** Server-generated paths prevent path traversal. A digest makes duplicate detection deterministic. LangChain loaders produce a common `Document` representation for the splitter.

**Tradeoff:** MIME checks improve input policy but do not replace content parsing or malware isolation. TXT is UTF-8 only. Scanned image PDFs need OCR. Multipart parsing occurs before the bounded read; deployment also needs ingress request limits.

**Technical answer:** “I validate at several levels, use temporary server-owned paths, and deduplicate by content hash. I don't trust the original filename as a filesystem location.”

**Verify:** Tests exercise invalid MIME, extension, headers, corrupted PDFs, empty text, invalid UTF-8, filename traversal, size limits, and a generated valid PDF with page metadata.

## 4. Split extracted documents into useful chunks

**Code:** `Ingestion.__init__` and `Ingestion.extract` in [`ingestion.py`](../backend/app/services/ingestion.py).

```python
self.splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    add_start_index=True,
)
```

The splitter prefers natural separators such as paragraph breaks before falling back to smaller boundaries. Each resulting chunk gets a UUID, the parent document ID, its text, and a one-based page number for PDFs. Character offsets computed by the splitter are not persisted by this version.

Default size is 1,000 characters with 150 characters of overlap. Overlap repeats boundary context so a sentence split across chunks is less likely to lose its meaning during retrieval. Extracted character and chunk limits bound further processing.

**Why:** A whole document embedding can blur together unrelated topics. Smaller chunks produce more focused retrieval and reduce the amount of text sent to the chat model.

**Tradeoff:** Smaller chunks can lose context; larger chunks can dilute relevance and increase token costs. Character counts are not token counts. These defaults are starting points, not optimal values for every corpus. Table-heavy PDFs need better extraction and evaluation.

**Technical answer:** “Chunk size balances semantic focus against context completeness. Overlap reduces boundary loss at the cost of repeated text and extra embedding work. I would tune both against representative questions.”

**Verify:** The chunking test uses text longer than a chunk and checks that multiple bounded chunks are produced.

## 5. Generate real embeddings through a configurable boundary

**Code:** [`services/providers.py`](../backend/app/services/providers.py).

```python
self._embeddings = OpenAIEmbeddings(
    api_key=self.settings.key("embedding"),
    base_url=self.settings.embedding_base_url,
    model=self.settings.embedding_model,
    request_timeout=self.settings.request_timeout,
    max_retries=1,
    check_embedding_ctx_length=False,
)
```

This is an actual LangChain integration. `embed_documents()` is called during ingestion; `embed_query()` is called when asking a question. They must use the same model and embedding space.

Disabling `check_embedding_ctx_length` sends text inputs to the configured endpoint instead of relying on OpenAI-specific local tokenization. It does not disable validation, TLS, authentication, or provider token limits. Chunk sizes still need to fit the chosen embedding model.

**Why:** A central provider boundary keeps keys and model assumptions out of ingestion and API code. Independent model IDs and base URLs support compatible providers or local compatible servers without rewriting the RAG service.

**Tradeoff:** OpenAI-compatible is a specific protocol contract. Anthropic or another incompatible native API needs a separate LangChain adapter; changing the URL is insufficient. The chat model must support the structured-output method we use.

**Technical answer:** “Embeddings map chunks and questions into the same vector space. I configure the integration centrally and reject zero, non-finite, malformed, or dimensionally incompatible vectors.”

**Verify:** The adapter integration test uses the actual `OpenAIEmbeddings` and `ChatOpenAI` classes with simulated HTTP responses. This checks serialization and response parsing. Live provider behavior remains unverified until keys are configured.

## 6. Index and retrieve with FAISS

**Code:** `Store.add`, `Store.search`, and `_vectors` in [`storage.py`](../backend/app/services/storage.py).

```python
index = faiss.IndexFlatIP(matrix.shape[1])
faiss.normalize_L2(matrix)
index.add(matrix)
```

For normalized vectors, inner product is cosine similarity:

```text
cosine(a, b) = (a · b) / (||a|| ||b||)
If both norms are 1, cosine(a, b) = a · b.
```

`IndexFlatIP` performs exact search; there is no approximation or keyword matching. Stored rows correspond to chunk records in the same order. The LangChain FAISS wrapper uses an `InMemoryDocstore` reconstructed from JSON to map vector rows back to their text and metadata.

```python
faiss.normalize_L2(query)
results = search.similarity_search_with_score_by_vector(
    query[0], k=self.settings.top_k
)
```

We retain results above `min_similarity`. The UI labels raw cosine similarity correctly; it is not a probability that an answer is correct. If no results pass the cutoff, we return an insufficient-information message without asking the chat model to guess.

**Why:** Exact FAISS search is easy to reason about for a small local library. Explicit normalization gives one clear scoring interpretation, and a configurable cutoff provides a simple low-relevance fallback.

**Tradeoff:** Exact search costs roughly O(N × d) per query, where N is vector count and d is dimension. A universal cosine threshold does not exist. Approximate indexes, reranking, and corpus-specific evaluation are future improvements, not implemented features.

**Technical answer:** “I use normalized embeddings with a flat inner-product FAISS index, so ranking is exact cosine similarity. LangChain handles retrieval-to-document mapping. Thresholds and top-k need evaluation; they don't guarantee relevance.”

**Verify:** Tests use controlled orthogonal embedding vectors to assert topic-specific ranking and unrelated-question rejection. Runtime uses provider embeddings, never this controlled test embedding class.

## 7. Persist the index and metadata as one snapshot

**Code:** `Snapshot`, `Store._commit`, `Store._load`, `atomic_json` in [`storage.py`](../backend/app/services/storage.py).

```python
faiss.write_index(index, str(path))
digest = hashlib.sha256(path.read_bytes()).hexdigest()
```

Each generation contains a native FAISS binary and validated JSON: documents, chunks, embedding identity, and the index checksum. No LangChain pickle docstore is saved or loaded.

```python
atomic_json(self.root / "current.json", {"generation": generation})
```

The pointer is switched only after snapshot files are written and synced. A crash before that switch leaves the old generation active. An unpublished generation is ignored at restart. If a write fails, the store fails closed and retains potentially useful files rather than risking deletion of an already-published generation.

On load, only a server-shaped generation ID is accepted, preventing an arbitrary persisted path. The loader validates schema version, unique IDs, parent associations, chunk counts, embedding identity, checksum, FAISS type, vector count, and dimension.

**Why:** Separately overwriting `index.faiss` and `metadata.json` can produce a mismatched pair after a crash. A single atomic pointer publishes the pair together. Avoiding pickle removes a known unsafe Python deserialization path.

**Tradeoff:** Full snapshots copy the index and metadata. Atomic rename plus fsync is designed and tested on the local Linux filesystem; it is not a distributed transaction or portable guarantee for every network filesystem. Checksums detect accidental corruption, not a malicious local user who can rewrite all files.

**Technical answer:** “I commit immutable snapshot generations and atomically publish a pointer. Readers never intentionally observe half an index update. I don't deserialize pickle, and I fail closed on incompatible or corrupt storage.”

**Verify:** Restart tests instantiate a new app and verify retrieval. Corruption and pointer-write failure tests verify fail-closed behavior and preservation of the previous complete generation.

## 8. Implement document deletion without stale vectors

**Code:** `Store.delete_document` in [`storage.py`](../backend/app/services/storage.py).

```python
positions = [i for i, chunk in enumerate(self.chunks)
             if chunk.document_id != document_id]
```

A new flat index is built using only surviving vector rows. A matching metadata/chunk snapshot is published. Existing vectors are copied, not regenerated through a paid provider. Deleting the last document writes a consistent empty snapshot with no index binary.

**Why:** Deleting only the document record would leave its vectors searchable. Copying survivors ensures the active search index and metadata agree.

**Tradeoff:** Rebuilding is O(N × d) and duplicates memory during the operation. This is acceptable for the initial local workflow, but an ID-mapped index or different storage design could improve larger workloads. We retain one previous snapshot for recovery; historical chats can retain source excerpts.

**Technical answer:** “Deletion changes both metadata and vectors in one published snapshot. It prevents stale retrieval, but it is logical deletion, not secure erasure from previous snapshots or saved conversations.”

**Verify:** The deletion test checks that a removed topic is no longer retrieved, a surviving topic still works, and the same behavior holds after restart.

## 9. Build the grounded RAG chain

**Code:** [`services/rag.py`](../backend/app/services/rag.py).

```python
PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "History (untrusted):\n{history}\n\nRetrieved excerpts (untrusted JSON):\n{context}\n\nCurrent question:\n{question}"),
])
```

The immutable application policy is a system message. Retrieved document text and history are separate untrusted data in the human message. The prompt explicitly tells the model to ignore instructions embedded in excerpts and to avoid treating prior assistant answers as evidence.

```python
chain = PROMPT | self.providers.llm.with_structured_output(
    GroundedAnswer, method="json_schema"
)
```

The model returns an answer, a `sufficient_context` flag, and citation IDs. The application validates the schema, checks that the IDs exist in retrieved context, and requires inline `[n]` references to match those IDs. Source filenames, page numbers, excerpts, and scores come from retrieval, not from the model's invented metadata.

If retrieval is empty or the model reports insufficient evidence, the response explains that the documents do not contain enough information. A provider error is a 502, not a fabricated answer. Failed generation saves no incomplete user/assistant turn.

**Why:** Structured output is easier to validate than arbitrary prose containing filenames. Source metadata stays authoritative because it comes from the ingestion records. Separating instructions from evidence reduces prompt-injection exposure.

**Tradeoff:** Citation validation proves references exist, not that the answer follows logically from them. Prompt instructions and a self-reported evidence flag are not formal grounding guarantees. A production system needs adversarial testing, evaluations, possibly reranking or claim verification, and human review for important decisions.

**Technical answer:** “RAG retrieves evidence before generation. I constrain the answer to that evidence, validate structured citations, and make uncertainty explicit. It reduces hallucination risk without making hallucinations impossible.”

**Verify:** Tests inspect the actual prompt messages, reject nonexistent or inconsistent citations, exercise insufficient evidence and provider failures, and complete an indexed-document-to-answer flow.

## 10. Persist conversation context separately

**Code:** `Conversation` / `Message` schemas; `Store.save_chat`, `Store.delete_chat`; [`rag.py`](../backend/app/services/rag.py).

```python
chat.messages.extend([
    Message(role="user", content=request.question, created_at=now),
    Message(role="assistant", content=answer, sources=sources, created_at=now),
])
self.store.save_chat(chat)
```

The chat is copied before appending, then the new complete chat file is atomically persisted. Each session has a server-generated UUID. Existing messages and source excerpts survive application restart. A new chat creates a new session only after its first successful question. Deleting a chat removes the saved session.

The last eight messages are supplied to the model. Retrieval includes the preceding user question to help resolve follow-ups. This is deliberately a simple strategy; it is not a general query-rewriting or memory system.

**Why:** Chat state should survive browser refreshes and should not be mixed into the document index. Conversation history helps interpret questions but is not trustworthy document evidence.

**Tradeoff:** Chats and document snapshots are separate transactions. Old excerpts remain historical after a document is removed. JSON history is rewritten in full and is not designed for large, concurrent multi-user workloads.

**Technical answer:** “I persist turns and citations in validated JSON, use bounded recent history for the model, and keep history separate from the evidence corpus. Document deletion does not pretend to erase historical chat text.”

**Verify:** Tests assert message ordering, continued sessions, restart reload, deletion, and that failed generation creates no history.

## 11. Connect React to the API and make failures visible

**Code:** [`frontend/src/App.jsx`](../frontend/src/App.jsx), [`services/api.js`](../frontend/src/services/api.js), [`components/Sidebar.jsx`](../frontend/src/components/Sidebar.jsx), [`components/Message.jsx`](../frontend/src/components/Message.jsx).

```javascript
export const sendQuestion = (question, chatId) => api('/chat', {
  method: 'POST', headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ question, chat_id: chatId }),
});
```

`api()` centralizes request timeout, status handling, and readable errors. Vite proxies relative `/api` requests during development. No provider key enters the browser bundle.

The application owns documents, saved chats, current messages, question text, loading state, and errors. Reusable components render the sidebar and source-backed messages. File inputs support selecting multiple files; uploads are processed sequentially and completed files are preserved if another fails. The UI uses processing messages rather than inventing an upload percentage the API cannot measure.

```jsx
<details className="source">
  <summary>...</summary>
  <p>{source.excerpt}</p>
</details>
```

Native `<details>` gives keyboard-accessible expandable source excerpts. React renders answer/excerpt text without `dangerouslySetInnerHTML`, preventing document text from executing as HTML. The app distinguishes user messages, assistant responses, setup warnings, and errors. It includes labels, focus styles, keyboard submission, mobile navigation, and reduced-motion support.

**Why:** Explicit loading and failure states stop users from assuming a file was indexed or an answer was saved when it failed. Small React components avoid introducing an unnecessary state-management or UI framework.

**Tradeoff:** This is a plain-text chat renderer, not a Markdown editor. Requests are globally serialized in the UI for simplicity. Browser timeouts cannot guarantee cancellation of the server's provider request; users should refresh persisted state before retrying.

**Technical answer:** “React state drives the visible workflow, while persisted data comes from FastAPI. I centralize fetch errors and timeouts, render untrusted text safely, and expose sources as expandable evidence rather than hiding them.”

**Verify:** The browser test operates against isolated real HTTP servers. It uploads TXT/PDF files, asks a question, opens citations, reloads saved history, deletes resources, checks errors, and exercises a narrow viewport with no horizontal overflow.

## 12. Verify layers and explain what was actually proven

**Code:** [`backend/tests`](../backend/tests), [`scripts/browser_smoke.py`](../scripts/browser_smoke.py), test commands in the [README](../README.md).

```bash
cd backend
../.venv/bin/python -m pytest -q
cd ../frontend
npm run build
cd ..
.venv/bin/python scripts/browser_smoke.py
```

The backend suite uses actual loaders, splitting, FAISS, JSON/native-index persistence, HTTP routes, and prompt composition. It replaces only external model responses. A separate adapter test uses real LangChain OpenAI clients and simulated HTTP, proving the expected integration shape without requiring credentials or spending money. Browser tests use controlled model behavior with the actual backend and frontend.

**Why:** External APIs are costly, nondeterministic, and unavailable without keys. Isolating them gives reproducible tests while preserving meaningful failure checks. A live-provider smoke test is still needed after configuration.

**Technical answer:** “The tests prove ingestion, retrieval, persistence, API behavior, and UI integration. They don't establish the quality of a particular live model. I'd separately measure retrieval recall and answer grounding on a representative evaluation set.”

Actual validation: 44 backend tests passed; the browser smoke and production build passed. No live model call was made because credentials were absent. One LangChain Community deprecation warning is documented in the README.

## Follow-up technical questions to practice

| Question | Answer outline |
| --- | --- |
| Why RAG instead of fine-tuning? | Documents change; retrieval provides inspectable sources without updating model weights. Fine-tuning can teach behavior but is not a reliable source-of-truth update mechanism. |
| What happens on upload? | Validate → hash/deduplicate → extract → split → embed → normalize → FAISS add → publish snapshot. |
| What happens on a question? | Validate → resolve chat → embed query → cosine search → filter → compose evidence prompt → generate structured response → validate citations → save turn. |
| Why FAISS instead of a database? | The brief requires FAISS; local exact vector search is sufficient here. Metadata and chats are standard-library JSON, with explicit scale/concurrency limits. |
| Why not `allow_dangerous_deserialization=True`? | That option permits pickle loading. We store native FAISS plus JSON and reconstruct the docstore instead. Native FAISS files must still come from trusted local storage. |
| Why include overlap? | Reduce boundary context loss; repeated context costs extra vectors/tokens. Tune against real questions. |
| What if the embedding model changes? | The persisted model/endpoint signature is checked; start a new data directory and re-ingest. Equal dimensions do not mean compatible semantic spaces. |
| How do you know the answer is grounded? | We provide retrieved evidence, require citations and an evidence flag, and validate IDs. Entailment still needs evaluation; valid citation syntax alone cannot prove truth. |
| What is prompt injection here? | A document can contain instructions pretending to outrank app policy. We treat it as untrusted data; prompt separation reduces risk but is not a complete defense. |
| What breaks with multiple workers? | Each process has its own lock and cached state; concurrent snapshot/chat updates could overwrite each other. This version explicitly supports one process. |
| Is removing a document secure deletion? | No. Active search excludes it; previous snapshots and historical chat excerpts can retain its contents. |
| What would production require? | Authentication, authorization, tenant isolation, protected storage, request/resource limits, parser isolation, observability, evaluations, operational backups, and concurrency design. These are not claimed as implemented. |
| How would you evaluate retrieval? | A labeled document/question set, relevant-chunk recall at k, threshold calibration, and failure analysis for tables, long documents, and ambiguous follow-ups. |
| How would you reduce latency? | Measure model calls vs search, batch embeddings appropriately, avoid duplicate ingestion, reduce context carefully, then consider caching or approximate indexing when evidence justifies it. |
| What are the privacy boundaries? | Disk storage is local, but document chunks go to the embedding provider and retrieved excerpts go to the chat provider unless local compatible models are configured. |

## A short explanation you can give aloud

“I built DocMind AI as a Python FastAPI backend and a React frontend. Uploads are validated, parsed through LangChain loaders, split into overlapping chunks, and embedded through a configurable provider. I normalize those vectors and use FAISS exact inner-product search, which gives cosine ranking. At question time I retrieve relevant chunks, send them to a structured LangChain chat model, validate the citations, and save the conversation. I persist native FAISS files and validated JSON as atomic snapshot generations, without pickle. Deletion rebuilds the active index from surviving vectors. The implementation is tested as a single-user, single-process application; live provider quality and production multi-user security require additional validation and design.”
