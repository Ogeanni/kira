"""
ingestion/chunker.py

Splits raw documents into chunks ready for embedding.
Three strategies benchmarked — the winner gets set in config.

Run directly to see chunk stats across all strategies:
    python ingestion/chunker.py
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Literal

from langchain_text_splitters import RecursiveCharacterTextSplitter


ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import get_settings

settings = get_settings()

@dataclass
class Chunk:
    """
    A single chunk of text ready for embedding.

    Two text fields matter here:
      text         — what gets embedded (always short and precise)
      window_context — what the LLM sees when generating an answer

    For fixed_size and recursive: text == window_context.
    For sentence_window: text is one sentence, window_context
    includes 2 sentences before and after for richer LLM context.

    This separation is the key insight of sentence_window chunking:
    embed small for precision, retrieve large for context.
    """
    chunk_id: str
    text: str              # embedded — kept short for retrieval precision
    window_context: str    # returned to LLM — includes surrounding context
    strategy: str
    source_file: str
    doc_type: str
    client: str
    char_count: int
    token_estimate: int    # rough: chars / 4
    chunk_index: int
    total_chunks_in_doc: int


class Chunker:
    """
    Produces Chunk objects from raw document text.

    Strategy comparison:
      fixed_size      — simple, fast, loses context at boundaries
      recursive       — respects paragraph/sentence structure
      sentence_window — best for Q&A: embed precise, retrieve rich
    """

    def __init__(self, strategy: Literal["fixed_size", "recursive", "sentence_window"]):
        self.strategy = strategy

    def chunk_document(self, text: str, metadata: dict) -> list[Chunk]:
        if self.strategy == "fixed_size":
            return self._fixed_size(text, metadata)
        elif self.strategy == "recursive":
            return self._recursive(text, metadata)
        elif self.strategy == "sentence_window":
            return self._sentence_window(text, metadata)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _fixed_size(self, text: str, metadata: dict) -> list[Chunk]:
        """
        Splits every 400 chars with 50-char overlap.

        The problem: cuts mid-sentence and mid-word.
        You will see this in Ragas scores — faithfulness drops
        because retrieved chunks start or end mid-thought.
        This is intentionally the worst strategy so you can
        measure the gap vs the better ones.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""],
        )
        raw_chunks = splitter.split_text(text)
        return self._build_chunks(raw_chunks, metadata)
    

    def _recursive(self, text: str, metadata: dict) -> list[Chunk]:
        """
        Respects document structure — tries double newlines first,
        then single newlines, then sentences, then words.

        chunk_size=600, chunk_overlap=100 is the practical default
        for text-embedding-3-small which handles up to 8191 tokens.
        600 chars ≈ 150 tokens — well within limits, good density.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        raw_chunks = splitter.split_text(text)
        return self._build_chunks(raw_chunks, metadata)

    def _sentence_window(self, text: str, metadata: dict) -> list[Chunk]:
        """
        Splits into individual sentences.

        embed text   = just the sentence (precise matching)
        window_context = 2 sentences before + current + 2 after
                        (rich context for LLM generation)

        Best for structured documents like SOPs where a question
        maps to a specific sentence but the answer needs context.
        Example: "What ACOS threshold triggers bid reduction?"
        matches one sentence precisely, but the answer makes
        more sense with the surrounding procedure steps.
        """
        # Split on sentence boundaries — handles ., !, ?
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Filter out fragments shorter than 20 chars
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        chunks = []
        for i, sentence in enumerate(sentences):
            # Window: 2 before + current + 2 after
            window_start = max(0, i - 2)
            window_end = min(len(sentences), i + 3)
            window = " ".join(sentences[window_start:window_end])

            chunk = Chunk(
                chunk_id=f"{metadata['filename']}_{self.strategy}_{i}",
                text=sentence,          # what gets embedded
                window_context=window,  # what the LLM sees
                strategy=self.strategy,
                source_file=metadata["filename"],
                doc_type=metadata["doc_type"],
                client=metadata["client"],
                char_count=len(sentence),
                token_estimate=len(sentence) // 4,
                chunk_index=i,
                total_chunks_in_doc=len(sentences),
            )
            chunks.append(chunk)
        return chunks

    def _build_chunks(self, raw_chunks: list[str], metadata: dict) -> list[Chunk]:
        """Converts raw string chunks into Chunk objects."""
        return [
            Chunk(
                chunk_id=f"{metadata['filename']}_{self.strategy}_{i}",
                text=chunk,
                window_context=chunk,   # same as text for non-window strategies
                strategy=self.strategy,
                source_file=metadata["filename"],
                doc_type=metadata["doc_type"],
                client=metadata["client"],
                char_count=len(chunk),
                token_estimate=len(chunk) // 4,
                chunk_index=i,
                total_chunks_in_doc=len(raw_chunks),
            )
            for i, chunk in enumerate(raw_chunks)
        ]
    

def chunk_all_documents(strategy: str) -> list[Chunk]:
    """
    Chunks every document in data/raw/ using the given strategy.
    Reads metadata from metadata_index.json for accurate doc_type and client.
    """
    raw_dir = settings.raw_dir
    index_path = raw_dir / "metadata_index.json"

    with open(index_path) as f:
        metadata_index = {m["filename"]: m for m in json.load(f)}

    chunker = Chunker(strategy=strategy)
    all_chunks = []

    for txt_file in sorted(raw_dir.glob("*.txt")):
        text = txt_file.read_text()
        meta = metadata_index.get(txt_file.name, {
            "filename": txt_file.name,
            "doc_type": "unknown",
            "client": "global",
        })
        chunks = chunker.chunk_document(text, meta)
        all_chunks.extend(chunks)

    return all_chunks


def save_chunks(chunks: list[Chunk], strategy: str) -> Path:
    """Saves chunks to data/chunks/{strategy}.json"""
    out_path = settings.chunks_dir / f"{strategy}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump([vars(c) for c in chunks], f, indent=2)

    return out_path


if __name__ == "__main__":
    print("\nRunning all chunking strategies...\n")

    for strategy in ["fixed_size", "recursive", "sentence_window"]:
        chunks = chunk_all_documents(strategy)
        path = save_chunks(chunks, strategy)

        avg_chars = sum(c.char_count for c in chunks) / len(chunks)
        print(f"{strategy:20s}: {len(chunks):4d} chunks | "
              f"avg {avg_chars:.0f} chars/chunk | saved → {path}")

    print("\n── Spot check: fixed_size mid-sentence cuts ──")
    import json
    with open(settings.chunks_dir / "fixed_size.json") as f:
        fixed = json.load(f)

    cuts = [c for c in fixed if not c["text"].strip()[-1:] in ".!?:"]
    print(f"Chunks ending mid-sentence: {len(cuts)} / {len(fixed)}")
    print(f"Example: ...{cuts[0]['text'][-80:]!r}" if cuts else "None found")

    print("\n── Spot check: sentence_window context expansion ──")
    with open(settings.chunks_dir / "sentence_window.json") as f:
        sw = json.load(f)

    example = sw[10]
    print(f"Embedded text  ({len(example['text'])} chars):")
    print(f"  {example['text']!r}")
    print(f"Window context ({len(example['window_context'])} chars):")
    print(f"  {example['window_context']!r}")