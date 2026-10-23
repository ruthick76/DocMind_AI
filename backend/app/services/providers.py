from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.core.errors import AppError


class Providers:
    """One boundary for OpenAI-compatible hosted or local model integrations."""

    def __init__(self, settings):
        self.settings = settings
        self._embeddings = None
        self._llm = None

    def missing(self):
        return [f"DOCMIND_{kind.upper()}_API_KEY (or DOCMIND_API_KEY)"
                for kind in ("llm", "embedding") if not self.settings.key(kind)]

    @property
    def embeddings(self):
        if not self.settings.key("embedding"):
            raise AppError(503, "configuration_required", "Configure an embedding API key in backend/.env.")
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                api_key=self.settings.key("embedding"),
                base_url=self.settings.embedding_base_url,
                model=self.settings.embedding_model,
                request_timeout=self.settings.request_timeout,
                max_retries=1,
                check_embedding_ctx_length=False,
            )
        return self._embeddings

    @property
    def llm(self):
        if not self.settings.key("llm"):
            raise AppError(503, "configuration_required", "Configure an LLM API key in backend/.env.")
        if self._llm is None:
            self._llm = ChatOpenAI(
                api_key=self.settings.key("llm"), base_url=self.settings.llm_base_url,
                model=self.settings.llm_model, temperature=0,
                timeout=self.settings.request_timeout, max_retries=1,
            )
        return self._llm
