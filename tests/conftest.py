"""
tests/conftest.py

Shared fixtures for the KIRA test harness.

What is a fixture?
  A fixture is a function that sets up something a test needs.
  pytest automatically calls fixtures before each test that requests them.
  This means every test gets a fresh, correctly configured setup
  without duplicating setup code across test files.

Why conftest.py specifically?
  pytest has a convention: fixtures defined in conftest.py are
  automatically available to every test file in the same directory
  and subdirectories. No import needed. This is pytest's way of
  sharing fixtures without coupling test files to each other.

Why not just import directly in each test?
  If setup logic lives in each test file, changing the setup
  means changing every file. A bug in setup appears in every file.
  conftest.py is the single source of truth for test setup.
"""

import os
import pytest
from dotenv import load_dotenv

load_dotenv()

from config.settings import get_settings
from knowledge.retriever import Retriever
from intelligence.memory import MemoryStore


# ── Why we define fixtures this way ──────────────────────────────────
# @pytest.fixture means pytest will call this function automatically
# when a test requests it by name.
#
# scope="session" means the fixture is created ONCE per test session
# and shared across all tests. The alternative is scope="function"
# which creates a fresh instance for every test.
#
# Why session scope for retriever?
#   Creating a retriever connects to ChromaDB or Pinecone. That
#   connection is expensive — it loads embedding models, opens
#   database connections. Creating it once and sharing it across
#   all tests is faster and avoids hitting API rate limits.
#
# Why session scope for memory store?
#   Same reason. But there's a tradeoff: if one test modifies the
#   memory store, subsequent tests see that modification. We handle
#   this by cleaning up in each test that writes to memory.

@pytest.fixture(scope="session")
def settings():
    """
    Returns the application settings.

    Why expose settings as a fixture?
    Tests need to know things like:
    - Which backend is being tested?
    - What are the retrieval thresholds?
    - What client IDs exist?
    Centralising this in settings means tests don't hardcode
    configuration values that might change.
    """
    return get_settings()


@pytest.fixture(scope="session")
def retriever():
    """
    Returns a configured Retriever instance.

    The retriever uses whatever backend is set in VECTOR_STORE_BACKEND.
    This means the same tests run against ChromaDB or Pinecone
    by changing one environment variable — the tests don't change.

    This is the abstract interface paying off:
    tests depend on Retriever, not on ChromaDB or Pinecone directly.
    """
    return Retriever()


@pytest.fixture(scope="session")
def memory_store():
    """
    Returns a configured MemoryStore instance.
    """
    return MemoryStore()


@pytest.fixture
def clean_memory(memory_store):
    """
    Yields memory_store with guaranteed cleanup after each test.
    
    Why yield instead of return?
    Code before yield runs before the test.
    Code after yield runs after the test — even if the test fails.
    This guarantees cleanup regardless of what happens in the test.
    
    Why is this safer than just deleting in the test body?
    If the test fails halfway through, code after the failure
    doesn't run. The yield fixture's cleanup always runs.
    """
    yield memory_store  # test runs here
    # cleanup always runs after, even on failure
    memory_store.delete_all("test_client")
    memory_store.delete_all("natura_test")


@pytest.fixture(scope="session")
def backend_name():
    """
    Returns the current vector store backend name as a string.
    Used in tests to label which backend is being tested
    and to skip tests that are backend-specific.

    Example usage in a test:
        def test_something(backend_name):
            if backend_name == "chromadb":
                pytest.skip("This test requires Pinecone namespaces")
    """
    return os.getenv("VECTOR_STORE_BACKEND", "chromadb").lower()


# ── Test client IDs ───────────────────────────────────────────────────
# Why define these as fixtures rather than hardcoding in tests?
# If you add a fifth client, you change it here once.
# Every test that uses client_ids gets the updated list automatically.

@pytest.fixture(scope="session")
def client_ids():
    """Returns the list of test clients."""
    return ["natura", "vitalblend", "peakgear", "lumina"]


@pytest.fixture(scope="session")
def global_namespace():
    """
    Returns the namespace name for global documents.

    Why a fixture for a string?
    If you rename the global namespace, one change propagates
    to every test. Hardcoding "global" in 10 test files creates
    10 places to update.
    """
    return "global"


# ── Known questions with ground truth ────────────────────────────────
# These are the questions where we manually identified the correct
# source and answer. Tests use these to verify retrieval correctness.
#
# Why define these in conftest rather than in individual test files?
# Multiple test files need the same questions:
#   test_retrieval.py  needs the question to check what came back
#   test_ranking.py    needs the question to check what rank it got
#   test_generation.py needs the question to check the answer
# One definition, shared everywhere.

@pytest.fixture(scope="session")
def known_questions():
    """
    Questions with manually verified ground truth.

    Structure:
      question:              the query string
      expected_source_type:  what doc_type should be retrieved
      expected_namespace:    which namespace should be searched
      ground_truth:          what the correct answer contains
      expected_keywords:     words that must appear in a correct answer
      should_not_contain:    words that indicate a wrong answer
    """
    return [
        {
            "question": "What ACOS threshold triggers bid reduction on keywords?",
            "expected_source_type": "sop",
            "expected_namespace": "global",
            "ground_truth": (
                "If ACOS exceeds the target by more than 10% for 3 consecutive "
                "days, bids on keywords with ACOS above 60% should be reduced by 15%."
            ),
            "expected_keywords": ["10%", "3", "60%", "15%"],
            "should_not_contain": ["anti-aging", "buy box", "listing title"],
        },
        {
            "question": "What terms are absolutely prohibited in Amazon listing copy?",
            "expected_source_type": "compliance",
            "expected_namespace": "global",
            "ground_truth": (
                "Prohibited terms include best, number one, cure, treat, "
                "prevent, guaranteed, chemical-free, and competitor brand names."
            ),
            "expected_keywords": ["prohibited", "best", "cure", "guarantee"],
            "should_not_contain": ["ACOS", "bid", "keyword"],
        },
        {
            "question": "What format should Amazon listing titles follow?",
            "expected_source_type": "sop",
            "expected_namespace": "global",
            "ground_truth": (
                "Brand, Primary Keyword, Key Feature, Size or Variant, "
                "Secondary Keyword. Maximum 200 characters."
            ),
            "expected_keywords": ["brand", "keyword", "200"],
            "should_not_contain": ["ACOS", "prohibited", "violation"],
        },
        {
            "question": "What steps should be taken when a product loses Buy Box?",
            "expected_source_type": "sop",
            "expected_namespace": "global",
            "ground_truth": (
                "Check if competitor is undercutting price. "
                "Verify inventory levels. Check seller feedback score."
            ),
            "expected_keywords": ["price", "inventory", "feedback"],
            "should_not_contain": ["prohibited", "anti-aging"],
        },
    ]