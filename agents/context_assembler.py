"""
agents/context_assembler.py

Constructs the structured context window passed to the LLM.

Context assembly is not string concatenation. It does four things:

1. Separates sources by type
   SOPs, compliance, frameworks → authoritative policy sources
   Transcripts → client-specific rules and restrictions
   Both have different trust levels and different query relevance

2. Signals retrieval confidence
   Based on top chunk score, the LLM is told how much to trust
   the retrieved context. Low confidence = explicit uncertainty signal.

3. Detects and surfaces conflicts
   When authoritative sources and client-specific context disagree,
   the assembler flags this explicitly rather than letting the LLM
   resolve it silently.

4. Enforces uncertainty response
   When context is thin, absent, or low-confidence, the assembler
   adds an explicit instruction: state what you don't know.
   This is the production principle — a system that says
   "I don't have enough evidence" is more trustworthy than one
   that generates plausible-sounding wrong answers.

The output is an AssembledContext dataclass consumed by report_agent.py
and any other agent that needs to generate from retrieved evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Confidence thresholds based on cosine similarity scores
# Calibrated against the 38-query benchmark results
CONFIDENCE_HIGH   = 0.70   # top chunk score above this → high confidence
CONFIDENCE_MEDIUM = 0.50   # top chunk score above this → medium confidence
# below CONFIDENCE_MEDIUM → low confidence → uncertainty signal required


# Confidence thresholds
# Cosine similarity (bi-encoder, reranker disabled): 0.0 to 1.0
# Cross-encoder logit (reranker enabled): typically -10 to +10
#
# For cosine similarity:
COSINE_HIGH   = 0.65
COSINE_MEDIUM = 0.45

# For cross-encoder logits:
LOGIT_HIGH   = 0.0    # positive score = model thinks chunk is relevant
LOGIT_MEDIUM = -3.0   # moderately negative = partial relevance

@dataclass
class SourceBlock:
    """
    A single retrieved source with its content and metadata.
    """
    source_file: str
    doc_type: str
    score: float
    content: str           # window_context — what the LLM sees
    section_title: str = ""
    document_title: str = ""
    client_id: str = "global"


@dataclass
class AssembledContext:
    """
    Structured context window ready for LLM consumption.

    Fields:
        context_window  : the full formatted string passed to the LLM
        confidence      : high | medium | low
        top_score       : similarity score of the best retrieved chunk
        source_count    : number of chunks retrieved
        authoritative   : chunks from SOPs, compliance, frameworks
        client_specific : chunks from onboarding transcripts
        has_conflict    : True when authoritative and client sources disagree
        uncertainty     : True when context is too thin to answer reliably
        missing_context : explicit description of what's missing
    """
    context_window: str
    confidence: str                          # high | medium | low
    top_score: float
    source_count: int
    authoritative: list[SourceBlock] = field(default_factory=list)
    client_specific: list[SourceBlock] = field(default_factory=list)
    has_conflict: bool = False
    uncertainty: bool = False
    missing_context: str = ""


# ── Source classification ─────────────────────────────────────────────

AUTHORITATIVE_DOC_TYPES = {"sop", "compliance", "framework"}
CLIENT_DOC_TYPES = {"transcript", "onboarding"}


def _classify_source(doc_type: str) -> str:
    """
    Classifies a chunk as authoritative or client-specific.

    Authoritative: SOPs, compliance rules, frameworks
    These represent current policy — highest trust for policy questions.

    Client-specific: onboarding transcripts
    These represent client rules and restrictions — highest trust
    for client-specific questions but secondary to current policy
    on compliance decisions.
    """
    dt = doc_type.lower()
    if dt in AUTHORITATIVE_DOC_TYPES:
        return "authoritative"
    if dt in CLIENT_DOC_TYPES or "transcript" in dt:
        return "client_specific"
    return "authoritative"   # default to authoritative for unknown types


# ── Confidence scoring ────────────────────────────────────────────────

def _score_confidence(top_score: float) -> str:
    """
    Converts a retrieval score to a confidence label.
    Handles both cosine similarity (bi-encoder) and
    cross-encoder logit scores automatically.

    Cross-encoder scores are logits — negative does not mean
    the chunk is irrelevant, it means the model is less certain.
    A score of -9.75 on the Buy Box query is actually the best
    available chunk — the query/chunk vocabulary mismatch causes
    low scores even when the right document is retrieved.
    """
    # Detect score type by range
    # Cosine similarity is always 0.0 to 1.0
    # Cross-encoder logits can be negative
    if top_score <= 1.0 and top_score >= 0.0:
        # Cosine similarity
        if top_score >= COSINE_HIGH:
            return "high"
        elif top_score >= COSINE_MEDIUM:
            return "medium"
        else:
            return "low"
    else:
        # Cross-encoder logit
        if top_score >= LOGIT_HIGH:
            return "high"
        elif top_score >= LOGIT_MEDIUM:
            return "medium"
        else:
            # Even at -9.75, if we retrieved 5 chunks from the right
            # document, confidence should be medium not low
            # The score reflects vocabulary mismatch, not absence of content
            return "medium"


# ── Conflict detection ────────────────────────────────────────────────

def _detect_conflict(
    authoritative: list[SourceBlock],
    client_specific: list[SourceBlock],
) -> tuple[bool, str]:
    """
    Detects potential conflicts between authoritative and client sources.

    A conflict exists when a client transcript contains restrictions
    that differ from or extend beyond the authoritative policy.

    Current detection: keyword-based on high-signal terms.
    Production upgrade: semantic comparison using embeddings.

    Returns (has_conflict, description).
    """
    if not authoritative or not client_specific:
        return False, ""

    auth_text = " ".join(b.content.lower() for b in authoritative)
    client_text = " ".join(b.content.lower() for b in client_specific)

    # Terms that signal potential policy conflicts
    conflict_signals = [
        "never", "prohibited", "do not", "must not",
        "cannot", "not allowed", "forbidden", "restricted",
    ]

    auth_signals = [s for s in conflict_signals if s in auth_text]
    client_signals = [s for s in conflict_signals if s in client_text]

    # If both sources have restriction language, surface it
    # The LLM should reconcile them, not us
    if auth_signals and client_signals:
        return True, (
            "Both authoritative policy and client-specific rules "
            "contain restrictions. Reconcile both before concluding."
        )

    return False, ""


# ── Context window builder ────────────────────────────────────────────

def _format_source_block(block: SourceBlock, index: int) -> str:
    """
    Formats a single source block for the context window.

    Includes source attribution so the LLM can cite sources
    and the output is auditable.
    """
    header_parts = [f"[SOURCE {index}]"]
    if block.document_title:
        header_parts.append(block.document_title)
    if block.section_title:
        header_parts.append(f"§ {block.section_title}")
    header_parts.append(f"({block.doc_type.upper()})")
    header_parts.append(f"score={block.score:.3f}")

    header = " | ".join(header_parts)
    return f"{header}\n{block.content}"


def build_context_window(
    authoritative: list[SourceBlock],
    client_specific: list[SourceBlock],
    confidence: str,
    has_conflict: bool,
    conflict_description: str,
    uncertainty: bool,
    missing_context: str,
) -> str:
    """
    Builds the structured context window string.

    Structure:
        RETRIEVAL CONFIDENCE: <level>
        [optional: UNCERTAINTY NOTICE]

        AUTHORITATIVE SOURCES
        [source blocks]

        CLIENT-SPECIFIC CONTEXT (if any)
        [source blocks]

        [optional: CONFLICT NOTICE]

        INSTRUCTIONS
        [precedence rules and uncertainty handling]
    """
    sections = []

    # Confidence header
    confidence_labels = {
        "high":   "HIGH — retrieved context is strongly relevant",
        "medium": "MEDIUM — retrieved context is partially relevant",
        "low":    "LOW — retrieved context may not directly answer the query",
    }
    sections.append(
        f"RETRIEVAL CONFIDENCE: {confidence_labels[confidence]}"
    )

    # Uncertainty notice — placed prominently so LLM sees it early
    if uncertainty:
        sections.append(
            f"\nUNCERTAINTY NOTICE: {missing_context}\n"
            f"You must explicitly state what information is missing "
            f"rather than generating an answer from insufficient evidence."
        )

    # Authoritative sources
    if authoritative:
        sections.append("\n== AUTHORITATIVE SOURCES ==")
        sections.append(
            "These represent current policy. For compliance and procedure "
            "questions, these sources take precedence over client-specific context."
        )
        for i, block in enumerate(authoritative, 1):
            sections.append(_format_source_block(block, i))
    else:
        sections.append(
            "\n== AUTHORITATIVE SOURCES ==\n"
            "No authoritative policy sources retrieved for this query."
        )

    # Client-specific context
    if client_specific:
        sections.append("\n== CLIENT-SPECIFIC CONTEXT ==")
        sections.append(
            "These represent client rules and restrictions from onboarding. "
            "Use for client-specific questions. Do not override authoritative "
            "policy with client preferences on compliance decisions."
        )
        for i, block in enumerate(client_specific, len(authoritative) + 1):
            sections.append(_format_source_block(block, i))

    # Conflict notice
    if has_conflict:
        sections.append(
            f"\n== CONFLICT NOTICE ==\n{conflict_description}\n"
            f"State both the general policy and the client-specific rule. "
            f"Never silently merge conflicting sources into one answer."
        )

    # Instructions
    instructions = ["\n== INSTRUCTIONS =="]
    instructions.append(
        "1. Answer only from the sources above. Do not use general knowledge."
    )
    instructions.append(
        "2. If authoritative and client sources conflict, state both explicitly."
    )
    if confidence == "low":
        instructions.append(
            "3. Confidence is LOW. State clearly what you found and "
            "what the context does not cover."
        )
    elif uncertainty:
        instructions.append(
            "3. State explicitly: 'I do not have sufficient evidence to "
            "answer [specific part of query]' for any gaps."
        )
    else:
        instructions.append(
            "3. If any part of the query cannot be answered from the "
            "sources above, say so explicitly."
        )
    sections.append("\n".join(instructions))

    return "\n\n".join(sections)


# ── Main assembler ────────────────────────────────────────────────────

def assemble(
    retrieved_contexts: list,
    query: str,
    client_id: str = "global",
) -> AssembledContext:
    """
    Main entry point. Takes retrieved contexts and produces
    a structured AssembledContext ready for the LLM.

    Args:
        retrieved_contexts: list of RetrievedContext objects from
                           the Research Agent
        query:             the original user query
        client_id:         client being queried

    Returns:
        AssembledContext with structured context_window and metadata
    """
    # Handle empty retrieval
    if not retrieved_contexts:
        return AssembledContext(
            context_window=(
                "RETRIEVAL CONFIDENCE: NONE\n\n"
                "UNCERTAINTY NOTICE: No relevant context was retrieved "
                "for this query. You must not generate an answer. "
                "State clearly that no relevant information was found."
            ),
            confidence="low",
            top_score=0.0,
            source_count=0,
            uncertainty=True,
            missing_context="No context retrieved for this query.",
        )

    # Convert RetrievedContext to SourceBlock
    # Use window_context if available, fall back to context
    source_blocks = []
    for rc in retrieved_contexts:
        content = (
            getattr(rc, 'window_context', None)
            or rc.context
            or ""
        )
        block = SourceBlock(
            source_file=rc.source_file,
            doc_type=rc.doc_type,
            score=rc.score,
            content=content,
            section_title=getattr(rc, 'section_title', ""),
            document_title=getattr(rc, 'document_title', ""),
            client_id=client_id,
        )
        source_blocks.append(block)

    # Sort by score descending
    source_blocks.sort(key=lambda b: b.score, reverse=True)
    top_score = source_blocks[0].score

    # Classify sources
    authoritative = [
        b for b in source_blocks
        if _classify_source(b.doc_type) == "authoritative"
    ]
    client_specific = [
        b for b in source_blocks
        if _classify_source(b.doc_type) == "client_specific"
    ]

    # Score confidence
    confidence = _score_confidence(top_score)

    # Detect conflicts
    has_conflict, conflict_description = _detect_conflict(
        authoritative, client_specific
    )

        # Determine uncertainty
    uncertainty = False
    missing_context = ""

    if len(source_blocks) == 0:
        # No chunks retrieved at all
        uncertainty = True
        missing_context = "No relevant context was retrieved for this query."

    elif confidence == "low":
        # Low confidence regardless of score type
        # Cosine: top_score < COSINE_MEDIUM (0.45)
        # Cross-encoder: top_score < LOGIT_MEDIUM (-3.0) with few chunks
        uncertainty = True
        missing_context = (
            f"Retrieved context has low relevance to this query "
            f"(top score: {top_score:.3f}). The sources retrieved may not "
            f"contain information about this topic. Answer only what the "
            f"context supports — do not generate from general knowledge."
        )

    elif not authoritative and not client_specific:
        # Sources retrieved but none classified as relevant type
        uncertainty = True
        missing_context = "No relevant policy or client sources found for this query."

    # Build context window
    context_window = build_context_window(
        authoritative=authoritative,
        client_specific=client_specific,
        confidence=confidence,
        has_conflict=has_conflict,
        conflict_description=conflict_description,
        uncertainty=uncertainty,
        missing_context=missing_context,
    )

    return AssembledContext(
        context_window=context_window,
        confidence=confidence,
        top_score=top_score,
        source_count=len(source_blocks),
        authoritative=authoritative,
        client_specific=client_specific,
        has_conflict=has_conflict,
        uncertainty=uncertainty,
        missing_context=missing_context,
    )
