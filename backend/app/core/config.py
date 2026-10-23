from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DOCMIND_", env_file=BACKEND_DIR / ".env", extra="ignore"
    )
    api_key: SecretStr | None = None
    llm_api_key: SecretStr | None = None
    embedding_api_key: SecretStr | None = None
    llm_base_url: str = "https://api.openai.com/v1"
    embedding_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    data_dir: Path = BACKEND_DIR / "data"
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    max_upload_mb: int = Field(10, ge=1, le=50)
    max_extracted_chars: int = Field(2_000_000, ge=1000)
    max_chunks_per_document: int = Field(5000, ge=1)
    chunk_size: int = Field(1000, ge=100, le=4000)
    chunk_overlap: int = Field(150, ge=0)
    top_k: int = Field(5, ge=1, le=12)
    min_similarity: float = Field(0.25, ge=-1, le=1)
    request_timeout: float = Field(45, ge=1, le=180)

    @model_validator(mode="after")
    def validate_settings(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be smaller than chunk size")
        if not self.data_dir.is_absolute():
            self.data_dir = BACKEND_DIR / self.data_dir
        for url in (self.llm_base_url, self.embedding_base_url):
            if not url.startswith(("https://", "http://localhost:", "http://127.0.0.1:")):
                raise ValueError("Model endpoints must use HTTPS (HTTP allowed on loopback)")
        return self

    def key(self, kind: str) -> str | None:
        for value in (getattr(self, f"{kind}_api_key"), self.api_key):
            if value is not None and value.get_secret_value():
                return value.get_secret_value()
        return None

    @property
    def embedding_signature(self) -> str:
        # Changing either requires rebuilding the index, even at equal dimension.
        return f"{self.embedding_base_url.rstrip('/')}|{self.embedding_model}"
