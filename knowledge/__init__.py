# knowledge/__init__.py

from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import get_settings
from knowledge.vector_store import VectorStore

def get_vector_store() -> VectorStore:
    """
    Returns the correct VectorStore backend based on settings.
    Call this everywhere — never instantiate ChromaStore or
    PineconeStore directly outside this module.

    Usage:
        from knowledge import get_vector_store
        store = get_vector_store()
        results = store.query(embedding, top_k=5)
    """
    settings = get_settings()
    if settings.use_pinecone:
        from knowledge.pinecone_store import PineconeStore
        return PineconeStore()
    else:
        from knowledge.chroma_store import ChromaStore
        return ChromaStore()