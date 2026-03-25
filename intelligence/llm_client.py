"""
intelligence/llm_client.py

Centralised OpenAI wrapper for all agents.
Adds: retry, token counting, cost logging.

Why centralise?
  - Token costs are invisible when every agent calls OpenAI directly
  - Retry logic duplicated across 4 agents
  - No single place to swap models or add observability

Every agent can import get_llm() instead of OpenAI() directly.
"""

import time
import tiktoken
from dataclasses import dataclass, field
from openai import OpenAI, RateLimitError, APIError

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import get_settings

settings = get_settings()

# OpenAI pricing per token (approximate 2024-2025 rates)
COST_TABLE = {
    "gpt-4o": {
        "input":  0.000005,
        "output": 0.000015,
    },
    "gpt-4o-mini": {
        "input":  0.0000006,
        "output": 0.0000024,
    },
}


@dataclass
class LLMResponse:
    """Structured response from a single LLM call."""
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_seconds: float
    retries: int = 0


@dataclass
class UsageTracker:
    """Accumulates token usage and cost across an entire pipeline run."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_calls: int = 0
    total_latency_seconds: float = 0.0
    calls: list[dict] = field(default_factory=list)

    def record(self, response: LLMResponse, agent_name: str = "") -> None:
        self.total_input_tokens += response.input_tokens
        self.total_output_tokens += response.output_tokens
        self.total_cost_usd += response.cost_usd
        self.total_calls += 1
        self.total_latency_seconds += response.latency_seconds
        self.calls.append({
            "agent": agent_name,
            "model": response.model,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost_usd": response.cost_usd,
            "latency_seconds": response.latency_seconds,
        })

    def summary(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_latency_seconds": round(self.total_latency_seconds, 2),
            "avg_latency_seconds": round(
                self.total_latency_seconds / max(self.total_calls, 1), 2
            ),
        }

    def print_summary(self) -> None:
        s = self.summary()
        print(f"\n── LLM usage summary ──")
        print(f"  Calls        : {s['total_calls']}")
        print(f"  Input tokens : {s['total_input_tokens']:,}")
        print(f"  Output tokens: {s['total_output_tokens']:,}")
        print(f"  Total cost   : ${s['total_cost_usd']:.6f}")
        print(f"  Total latency: {s['total_latency_seconds']}s")
        print(f"  Avg latency  : {s['avg_latency_seconds']}s/call")


class LLMClient:
    """
    Drop-in replacement for direct OpenAI calls.
    Adds retry, cost tracking, and token counting.

    Usage:
        from intelligence.llm_client import get_llm
        llm = get_llm(agent_name="research_agent")
        response = llm.chat(messages=[...])
        print(response.content)
        print(response.cost_usd)
    """

    def __init__(
        self,
        tracker: UsageTracker | None = None,
        agent_name: str = "unknown",
        model: str | None = None,
    ):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = model or settings.openai_chat_model
        self.tracker = tracker
        self.agent_name = agent_name
        self._encoder = None

    def _get_encoder(self) -> tiktoken.Encoding:
        if self._encoder is None:
            try:
                self._encoder = tiktoken.encoding_for_model(self.model)
            except KeyError:
                self._encoder = tiktoken.get_encoding("cl100k_base")
        return self._encoder

    def count_tokens(self, messages: list[dict]) -> int:
        """Counts tokens in a messages list before sending."""
        encoder = self._get_encoder()
        total = 0
        for msg in messages:
            total += 4
            total += len(encoder.encode(msg.get("content", "")))
            total += len(encoder.encode(msg.get("role", "")))
        return total

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        rates = COST_TABLE.get(model, COST_TABLE["gpt-4o-mini"])
        return (input_tokens * rates["input"]) + (output_tokens * rates["output"])

    def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_retries: int = 3,
    ) -> LLMResponse:
        """
        Makes a chat completion call with exponential backoff retry.
        Retries on RateLimitError and transient APIErrors only.
        """
        temp = temperature if temperature is not None else settings.openai_temperature
        max_tok = max_tokens or settings.openai_max_tokens
        last_error = None
        start_time = time.time()

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tok,
                )
                latency = time.time() - start_time
                usage = response.usage
                cost = self._calculate_cost(
                    self.model, usage.prompt_tokens, usage.completion_tokens
                )
                result = LLMResponse(
                    content=response.choices[0].message.content,
                    model=self.model,
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                    cost_usd=cost,
                    latency_seconds=round(latency, 2),
                    retries=attempt,
                )
                if self.tracker:
                    self.tracker.record(result, self.agent_name)
                return result

            except RateLimitError as e:
                last_error = e
                wait = 2 ** attempt
                print(f"  [{self.agent_name}] Rate limited. Waiting {wait}s...")
                time.sleep(wait)

            except APIError as e:
                if "auth" in str(e).lower():
                    raise
                last_error = e
                wait = 2 ** attempt
                print(f"  [{self.agent_name}] API error. Waiting {wait}s...")
                time.sleep(wait)

        raise RuntimeError(
            f"LLM call failed after {max_retries} attempts: {last_error}"
        )


# ── Module-level shared tracker ────────────────────────────────────────
_global_tracker = UsageTracker()


def get_llm(agent_name: str = "unknown", model: str | None = None) -> LLMClient:
    """
    Returns an LLMClient connected to the global tracker.
    Import this in agents instead of OpenAI() directly.
    """
    return LLMClient(tracker=_global_tracker, agent_name=agent_name, model=model)


def get_global_tracker() -> UsageTracker:
    """Returns the global usage tracker for end-of-pipeline reporting."""
    return _global_tracker


def reset_global_tracker() -> None:
    """Resets the global tracker — call at the start of each pipeline run."""
    global _global_tracker
    _global_tracker = UsageTracker()


if __name__ == "__main__":
    reset_global_tracker()
    llm = get_llm(agent_name="test")

    print("── Token count before call ──")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is ACOS in Amazon advertising?"},
    ]
    print(f"  Estimated input tokens: {llm.count_tokens(messages)}")

    print("\n── LLM call ──")
    response = llm.chat(messages=messages, temperature=0)
    print(f"  Response     : {response.content[:80]}...")
    print(f"  Input tokens : {response.input_tokens}")
    print(f"  Output tokens: {response.output_tokens}")
    print(f"  Cost         : ${response.cost_usd:.6f}")
    print(f"  Latency      : {response.latency_seconds}s")
    print(f"  Retries      : {response.retries}")

    get_global_tracker().print_summary()