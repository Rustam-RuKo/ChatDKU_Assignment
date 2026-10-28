"""Configuration for the standalone ChatDKU agent harness."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv


_PACKAGE_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _PACKAGE_DIR.parents[1]


def _env(name: str, default: str) -> str:
    """Return a non-empty environment value, or its documented default."""
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _optional_env(name: str) -> str | None:
    """Return a stripped environment value, preserving an intentional blank."""
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


def _env_int(name: str, default: int) -> int:
    value = _env(name, str(default))
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer; received {value!r}.") from exc


def _env_float(name: str, default: float) -> float:
    value = _env(name, str(default))
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number; received {value!r}.") from exc


def _env_path(name: str, default: Path) -> Path:
    path = Path(_env(name, str(default))).expanduser()
    if not path.is_absolute():
        path = _PROJECT_DIR / path
    return path.resolve()


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings for the agent, retrieval, and local workspace."""

    environment: str
    project_dir: Path
    data_dir: Path
    document_dir: Path
    index_dir: Path
    cache_dir: Path
    vector_collection: str
    log_level: str

    llm_model: str
    llm_base_url: str
    llm_api_key: str | None
    llm_temperature: float
    llm_max_tokens: int
    llm_timeout_seconds: int

    local_embedding_model: str | None
    embedding_provider: str
    embedding_base_url: str | None
    tei_embedding_model: str | None
    embedding_api_key: str | None
    chunk_size: int
    chunk_overlap: int
    retrieval_top_k: int
    evidence_max_bytes: int = 24000
    context_tokens: int = 16384
    context_safety_tokens: int = 1024
    tokenizer_path: str | None = None

    @classmethod
    def from_environment(cls, env_file: Path | None = None) -> "Settings":
        """Build settings from ``.env`` and process environment variables.

        Process environment variables take precedence over the optional file.
        """
        load_dotenv(env_file or _PROJECT_DIR / ".env", override=False)

        data_dir = _env_path("CHATDKU_DATA_DIR", _PROJECT_DIR / "data")
        document_dir = _env_path("CHATDKU_DOCUMENT_DIR", data_dir / "documents")
        index_dir = _env_path("CHATDKU_INDEX_DIR", data_dir / "index")
        cache_dir = _env_path("CHATDKU_CACHE_DIR", data_dir / "cache")

        settings = cls(
            environment=_env("CHATDKU_ENV", "development"),
            project_dir=_PROJECT_DIR,
            data_dir=data_dir,
            document_dir=document_dir,
            index_dir=index_dir,
            cache_dir=cache_dir,
            vector_collection=_env("CHATDKU_VECTOR_COLLECTION", "chatdku"),
            log_level=_env("CHATDKU_LOG_LEVEL", "INFO").upper(),
            llm_model=_env("CHATDKU_LLM_MODEL", ""),
            llm_base_url=_env("CHATDKU_LLM_BASE_URL", "http://localhost:8000/v1").rstrip("/"),
            llm_api_key=os.getenv("CHATDKU_LLM_API_KEY") or None,
            llm_temperature=_env_float("CHATDKU_LLM_TEMPERATURE", 0.2),
            llm_max_tokens=_env_int("CHATDKU_LLM_MAX_TOKENS", 2048),
            llm_timeout_seconds=_env_int("CHATDKU_LLM_TIMEOUT_SECONDS", 120),
            # This is a Hugging Face model name or local model path. The
            # embedding implementation downloads it on first use and caches it
            # locally, so no embedding service or GPU is required.
            local_embedding_model=_optional_env("CHATDKU_LOCAL_EMBEDDING_MODEL"),
            embedding_provider=_env("CHATDKU_EMBEDDING_PROVIDER", "auto").lower(),
            embedding_base_url=(_optional_env("CHATDKU_EMBEDDING_BASE_URL") or "").rstrip("/") or None,
            tei_embedding_model=_optional_env("CHATDKU_TEI_EMBEDDING_MODEL")
            or _optional_env("CHATDKU_REMOTE_EMBEDDING_MODEL"),
            embedding_api_key=_optional_env("CHATDKU_EMBEDDING_API_KEY"),
            chunk_size=_env_int("CHATDKU_CHUNK_SIZE", 512),
            chunk_overlap=_env_int("CHATDKU_CHUNK_OVERLAP", 64),
            retrieval_top_k=_env_int("CHATDKU_RETRIEVAL_TOP_K", 8),
            evidence_max_bytes=_env_int("CHATDKU_EVIDENCE_MAX_BYTES", 24000),
            context_tokens=_env_int("CHATDKU_CONTEXT_TOKENS", 16384),
            context_safety_tokens=_env_int("CHATDKU_CONTEXT_SAFETY_TOKENS", 1024),
            tokenizer_path=_optional_env("CHATDKU_TOKENIZER_PATH"),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        """Fail early for settings that would otherwise fail during a run."""
        if self.llm_model and not self.llm_base_url.startswith(("http://", "https://")):
            raise ValueError("CHATDKU_LLM_BASE_URL must start with http:// or https://.")
        if self.embedding_provider not in {"auto", "local", "tei"}:
            raise ValueError("CHATDKU_EMBEDDING_PROVIDER must be auto, local, or tei.")
        if self.embedding_base_url and not self.embedding_base_url.startswith(("http://", "https://")):
            raise ValueError("CHATDKU_EMBEDDING_BASE_URL must start with http:// or https://.")
        if self.resolved_embedding_provider == "tei" and not self.embedding_base_url:
            raise ValueError(
                "Set CHATDKU_EMBEDDING_BASE_URL when CHATDKU_EMBEDDING_PROVIDER=tei."
            )
        if self.resolved_embedding_provider == "tei" and not self.tei_embedding_model:
            raise ValueError(
                "Set CHATDKU_TEI_EMBEDDING_MODEL when CHATDKU_EMBEDDING_PROVIDER=tei."
            )
        if self.resolved_embedding_provider == "local" and not self.local_embedding_model:
            raise ValueError(
                "Set CHATDKU_LOCAL_EMBEDDING_MODEL when CHATDKU_EMBEDDING_PROVIDER=local."
            )
        if not 0 <= self.llm_temperature <= 2:
            raise ValueError("CHATDKU_LLM_TEMPERATURE must be between 0 and 2.")
        if self.llm_max_tokens < 1 or self.llm_timeout_seconds < 1:
            raise ValueError("LLM token and timeout values must be positive.")
        if self.chunk_size < 1 or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("CHATDKU_CHUNK_OVERLAP must be non-negative and smaller than CHATDKU_CHUNK_SIZE.")
        if self.retrieval_top_k < 1:
            raise ValueError("CHATDKU_RETRIEVAL_TOP_K must be positive.")
        if (self.evidence_max_bytes < 1 or self.context_safety_tokens < 1024 or
                self.context_tokens <= self.llm_max_tokens + self.context_safety_tokens):
            raise ValueError("Evidence/context budgets must be positive and reserve output plus a safety margin >=1024.")
        if not self.vector_collection:
            raise ValueError("CHATDKU_VECTOR_COLLECTION must not be empty.")

    @property
    def resolved_embedding_provider(self) -> str:
        """Resolve auto mode while keeping an explicit provider authoritative."""
        if self.embedding_provider != "auto":
            return self.embedding_provider
        if self.embedding_base_url and self.tei_embedding_model:
            return "tei"
        if self.local_embedding_model:
            return "local"
        raise ValueError(
            "Configure a local embedding model or a TEI model and endpoint."
        )

    @property
    def uses_tei_embeddings(self) -> bool:
        """Whether retrieval should use the configured TEI embedding service."""
        return self.resolved_embedding_provider == "tei"

    def ensure_workspace(self) -> None:
        """Create the local directories owned by the harness when needed."""
        for path in (self.document_dir, self.index_dir, self.cache_dir):
            path.mkdir(parents=True, exist_ok=True)

    def public_values(self) -> Mapping[str, object]:
        """Return settings suitable for logs without exposing API keys."""
        return {
            "environment": self.environment,
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
            "local_embedding_model": self.local_embedding_model,
            "embedding_provider": self.resolved_embedding_provider,
            "tei_embedding_model": self.tei_embedding_model,
            "embedding_base_url": self.embedding_base_url,
            "uses_tei_embeddings": self.uses_tei_embeddings,
            "document_dir": str(self.document_dir),
            "index_dir": str(self.index_dir),
            "vector_collection": self.vector_collection,
            "evidence_max_bytes": self.evidence_max_bytes,
            "context_tokens": self.context_tokens,
            "context_safety_tokens": self.context_safety_tokens,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and validate settings when the harness first needs them."""
    return Settings.from_environment()
