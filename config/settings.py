"""
config/settings.py

Central configuration for KIRA.
Every other file imports from here — nothing reads .env directly.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings

ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):

    # ── Project ───────────────────────────────────────
    project_name: str = "KIRA"
    version: str = "0.1.0"
    environment: Literal["development", "production"] = "development"

    # ── OpenAI ────────────────────────────────────────
    openai_api_key: Optional[str] = Field(default=None)
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_max_tokens: int = 2048
    openai_temperature: float = 0.0

    # ── Vector store ──────────────────────────────────
    # development  → chromadb  (local, no account needed)
    # production   → pinecone  (managed, namespace isolation)
    vector_store_backend: Literal["chromadb", "pinecone"] = "chromadb"

    chroma_persist_dir: str = "data/chroma"
    chroma_collection_name: str = "kira_knowledge"

    pinecone_api_key: Optional[str] = Field(default=None)
    pinecone_index_name: str = "kira-knowledge"
    pinecone_default_namespace: str = "global"

    # ── Chunking ──────────────────────────────────────
    # Update chunking_strategy after running eval_chunking.py
    chunking_strategy: Literal["fixed_size", "recursive", "sentence_window"] = "sentence_window"
    chunk_size: int = 600
    chunk_overlap: int = 100
    retrieval_top_k: int = 5

    # ── Embedding cache ───────────────────────────────
    embedding_cache_enabled: bool = True
    embedding_cost_per_1k_tokens: float = 0.00002

    # ── Agent pipeline ────────────────────────────────
    agent_max_iterations: int = 5
    compliance_confidence_threshold: float = 0.85
    pipeline_latency_target_seconds: float = 30.0

    # ── LoRA ──────────────────────────────────────────
    lora_enabled: bool = False
    lora_base_model: str = "microsoft/Phi-3-mini-4k-instruct"
    lora_model_path: Optional[str] = Field(default=None)

    # ── Memory ────────────────────────────────────────
    mem0_enabled: bool = True
    memory_top_k: int = 3

    # ── Observability ─────────────────────────────────
    langsmith_enabled: bool = False
    langsmith_api_key: Optional[str] = Field(default=None)
    langsmith_project: str = "kira"
    langsmith_sample_rate: float = 1.0

    # ── Ragas thresholds ──────────────────────────────
    ragas_faithfulness_threshold: float = 0.85
    ragas_context_precision_threshold: float = 0.80

    # ── Output ────────────────────────────────────────
    aws_access_key_id: Optional[str] = Field(default=None)
    aws_secret_access_key: Optional[str] = Field(default=None)
    s3_bucket_name: Optional[str] = Field(default=None)
    s3_region: str = "us-east-1"
    s3_reports_prefix: str = "kira-reports"
    slack_webhook_url: Optional[str] = Field(default=None)
    database_url: str = Field(default="postgresql://kira:kira_dev@localhost:5432/kira")

    # ── Computed paths ────────────────────────────────
    @property
    def data_dir(self) -> Path:
        return ROOT / "data"

    @property
    def raw_dir(self) -> Path:
        return ROOT / "data" / "raw"

    @property
    def chunks_dir(self) -> Path:
        return ROOT / "data" / "chunks"

    @property
    def embeddings_dir(self) -> Path:
        return ROOT / "data" / "embeddings"

    @property
    def metrics_dir(self) -> Path:
        return ROOT / "data" / "metrics"

    @property
    def compliance_dir(self) -> Path:
        return ROOT / "data" / "compliance"

    @property
    def chroma_dir(self) -> Path:
        return ROOT / self.chroma_persist_dir

    # ── Guards ────────────────────────────────────────
    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def has_pinecone(self) -> bool:
        return bool(self.pinecone_api_key)

    @property
    def use_pinecone(self) -> bool:
        return self.vector_store_backend == "pinecone" and self.has_pinecone

    @property
    def has_s3(self) -> bool:
        return bool(
            self.aws_access_key_id
            and self.aws_secret_access_key
            and self.s3_bucket_name
        )
    
    @property
    def s3_endpoint_url(self) -> str:
        return f"https://s3.{self.s3_region}.amazonaws.com"

    @property
    def has_slack(self) -> bool:
        return bool(self.slack_webhook_url)

    @property
    def has_lora(self) -> bool:
        return self.lora_enabled and bool(self.lora_model_path)

    def ensure_dirs(self):
        """Create all data directories if they don't exist."""
        dirs = [
            self.raw_dir,
            self.chunks_dir,
            self.embeddings_dir,
            self.metrics_dir,
            self.compliance_dir,
            self.chroma_dir,
            ROOT / "output" / "reports",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    model_config = {
        "env_file": str(ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "case_sensitive": False,
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns the same Settings instance every time.
    Import this everywhere — never instantiate Settings() directly.

    Usage:
        from config.settings import get_settings
        s = get_settings()
        print(s.openai_chat_model)
    """
    s = Settings()
    s.ensure_dirs()
    return s


if __name__ == "__main__":
    s = get_settings()
    print(f"\n{'─' * 40}")
    print(f"  KIRA v{s.version}")
    print(f"{'─' * 40}")
    print(f"  environment      : {s.environment}")
    print(f"  vector backend   : {s.vector_store_backend}")
    print(f"  chat model       : {s.openai_chat_model}")
    print(f"  embedding model  : {s.openai_embedding_model}")
    print(f"  chunking         : {s.chunking_strategy}")
    print(f"  lora enabled     : {s.lora_enabled}")
    print(f"{'─' * 40}")
    print(f"  has_openai       : {s.has_openai}")
    print(f"  has_pinecone     : {s.has_pinecone}")
    print(f"  use_pinecone     : {s.use_pinecone}")
    print(f"  has_s3           : {s.has_s3}")
    print(f"  has_lora         : {s.has_lora}")
    print(f"{'─' * 40}\n")