"""
ingestion/pipeline.py

Production document ingestion pipeline for KIRA.

Handles extraction, structure detection, chunking, quality validation,
and reporting for all supported document formats.

Supported formats (Priority 1):
    .docx  — Word documents (SOPs, frameworks)
    .pdf   — PDF documents (compliance docs)
    .txt   — Plain text (transcripts)
    .xlsx  — Excel spreadsheets (campaign reports, catalogues)
    .csv   — CSV files (ACOS targets, keyword exports)

Failure policy:
    A document that fails any quality gate does not proceed.
    Partial ingestion is not permitted — the cost of wrong answers
    downstream exceeds the cost of requiring document resubmission.
    Every failure is logged with a specific reason so the submitter
    knows exactly what to fix.

Chunking strategies by content structure:
    narrative  → sentence_window  (SOPs, frameworks, transcripts)
    list/term  → paragraph        (compliance docs, term lists)
    tabular    → row_context      (XLSX, CSV)
    mixed      → section-routed   (document with multiple structure types)

Run:
    python ingestion/pipeline.py --input data/raw/documents/
    python ingestion/pipeline.py --input data/raw/documents/sop/sop_acos_management.docx
    python ingestion/pipeline.py --input data/raw/tabular/ --client natura
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ── Data contracts ────────────────────────────────────────────────────

@dataclass
class ExtractedDocument:
    """
    Output of the extraction stage.
    Contains raw text with structural metadata preserved.
    """
    file_path: str
    file_format: str        # docx | pdf | txt | xlsx | csv
    raw_text: str           # full extracted text (for narrative docs)
    structured_rows: list[dict] = field(default_factory=list)  # for tabular
    headings: list[dict] = field(default_factory=list)  # {level, text, char_pos}
    page_count: int = 1
    extraction_warnings: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    """
    A single retrievable unit after chunking.
    Preserves source truth separately from retrieval representation.
    """
    chunk_id: str
    chunk_text: str         # original source content — never modified
    source_file: str
    doc_type: str
    client_id: str
    chunk_index: int
    total_chunks: int
    chunk_strategy: str     # sentence_window | paragraph | row_context
    structure_type: str     # narrative | list | tabular | mixed
    section_title: str = ""
    document_title: str = ""
    window_context: str = ""
    char_count: int = 0
    token_estimate: int = 0

    def __post_init__(self):
        self.char_count = len(self.chunk_text)
        self.token_estimate = self.char_count // 4


@dataclass
class IngestionResult:
    file_path: str
    file_name: str
    status: str = "pending"
    chunks_produced: int = 0
    chunks_rejected: int = 0
    failure_reason: str = ""
    warnings: list[str] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    processed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ── Format extractors ─────────────────────────────────────────────────

def extract_docx(file_path: Path) -> ExtractedDocument:
    """
    Extracts text and structure from a DOCX file.

    Uses python-docx to read paragraphs and their styles.
    Heading styles (Heading 1, Heading 2) are preserved as structural
    metadata — this is more reliable than detecting headers from text
    patterns because structure is explicit in the XML.

    Failure modes handled:
    - File not readable → raises, caught by caller
    - No text content → empty raw_text, caught by Gate 1
    - Tracked changes → python-docx reads accepted text by default
    """
    from docx import Document as DocxDocument

    doc = DocxDocument(str(file_path))
    text_parts = []
    headings = []
    warnings = []
    char_pos = 0

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            text_parts.append('')
            char_pos += 1
            continue

        # Record heading positions for structure-aware chunking
        style_name = para.style.name if para.style else ''
        if 'Heading' in style_name:
            try:
                level = int(style_name.split()[-1])
            except (ValueError, IndexError):
                level = 1
            headings.append({
                'level': level,
                'text': text,
                'char_pos': char_pos,
            })

        text_parts.append(text)
        char_pos += len(text) + 1

    raw_text = '\n\n'.join(p for p in text_parts if p)

    if not raw_text.strip():
        warnings.append("DOCX contains no extractable text paragraphs")

    return ExtractedDocument(
        file_path=str(file_path),
        file_format='docx',
        raw_text=raw_text,
        headings=headings,
        extraction_warnings=warnings,
    )


def extract_pdf(file_path: Path) -> ExtractedDocument:
    """
    Extracts text from a text-based PDF using pdfplumber.

    pdfplumber is used over pypdf because it better preserves
    layout for structured documents — table rows, indented lists,
    and multi-column text are handled more accurately.

    Failure modes handled:
    - Scanned PDF (no embedded text) → empty pages detected, flagged
    - Password protected → raises PasswordEncryptedPdfError, caught
    - Corrupt file → raises, caught by caller
    - Mixed pages (some text, some scanned) → partial extraction warning
    """
    import pdfplumber

    pages_text = []
    warnings = []
    empty_pages = []

    try:
        with pdfplumber.open(str(file_path)) as pdf:
            page_count = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    pages_text.append(text)
                else:
                    empty_pages.append(i + 1)
    except Exception as e:
        if 'password' in str(e).lower() or 'encrypt' in str(e).lower():
            raise ValueError(f"PDF is password protected: {file_path.name}")
        raise

    if empty_pages:
        if len(empty_pages) == page_count:
            warnings.append(
                f"All {page_count} pages returned no text. "
                f"This may be a scanned PDF requiring OCR. "
                f"OCR support is Priority 3 — not yet implemented."
            )
        else:
            warnings.append(
                f"Pages {empty_pages} returned no text — "
                f"may contain scanned content or images only."
            )

    def _reconstruct_paragraphs(text: str) -> str:
        """
        Reconstructs paragraph boundaries after pdfplumber extraction.

        pdfplumber's extract_text() strips blank lines — all content
        arrives as single-newline-separated lines. This function
        reinserts blank lines at structural boundaries so paragraph
        chunking can split correctly.

        Works on KIRA's compliance document structure. A general-purpose
        pipeline would need a more sophisticated approach.
        """
        lines = text.split('\n')
        result = []

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Insert blank line before sub-category headers (end with colon)
            if (stripped.endswith(':') and
                    stripped == stripped.lstrip() and
                    i > 0 and result and
                    result[-1].strip() and
                    not result[-1].strip().endswith(':')):
                result.append('')

            # Insert blank line before ALL CAPS section headers
            elif (stripped and
                stripped == stripped.upper() and
                len(stripped) > 3 and
                i > 0 and result and
                result[-1].strip() and
                not result[-1].strip().endswith(':') and
                not result[-1].strip() == result[-1].strip().upper()):
                result.append('')

            result.append(line)

            # Insert blank line after parenthetical qualifiers
            if stripped.startswith('(unless') or stripped.startswith('(only'):
                result.append('')

        return '\n'.join(result)

    raw_text = '\n'.join(pages_text)
    raw_text = _reconstruct_paragraphs('\n'.join(pages_text))

    return ExtractedDocument(
        file_path=str(file_path),
        file_format='pdf',
        raw_text=raw_text,
        page_count=page_count,
        extraction_warnings=warnings,
    )


def extract_txt(file_path: Path) -> ExtractedDocument:
    """
    Reads plain text files.

    Handles encoding detection — tries UTF-8 first, falls back to
    latin-1. Most text files in the wild are one of these two.
    A file that can't be decoded in either is flagged.
    """
    warnings = []

    try:
        raw_text = file_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        try:
            raw_text = file_path.read_text(encoding='latin-1')
            warnings.append("File decoded as latin-1 (not UTF-8) — verify content")
        except UnicodeDecodeError:
            raise ValueError(
                f"Cannot decode {file_path.name} — "
                f"not UTF-8 or latin-1. Re-save as UTF-8."
            )

    return ExtractedDocument(
        file_path=str(file_path),
        file_format='txt',
        raw_text=raw_text,
        extraction_warnings=warnings,
    )


def extract_xlsx(file_path: Path) -> ExtractedDocument:
    """
    Extracts rows from an Excel file.

    Reads the first sheet by default. Each row becomes a structured
    dict keyed by column headers. Empty rows are skipped.

    Failure modes handled:
    - No header row → detected, flagged
    - Merged cells → openpyxl returns value for top-left only, rest None
    - Multiple sheets → reads first sheet, warns about others
    - Formula cells → reads cached values (data_only=True)
    """
    import openpyxl

    warnings = []

    wb = openpyxl.load_workbook(str(file_path), data_only=True)

    if len(wb.sheetnames) > 1:
        warnings.append(
            f"File has {len(wb.sheetnames)} sheets "
            f"({wb.sheetnames}). Only first sheet ingested: "
            f"'{wb.sheetnames[0]}'"
        )

    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))

    if not rows:
        return ExtractedDocument(
            file_path=str(file_path),
            file_format='xlsx',
            raw_text='',
            structured_rows=[],
            extraction_warnings=["Spreadsheet is empty"],
        )

    # First row is headers
    headers = [str(h).strip() if h is not None else f"Column_{i}"
               for i, h in enumerate(rows[0])]

    if all(h.startswith('Column_') for h in headers):
        warnings.append(
            "No header row detected — first row used as data, "
            "not headers. Chunk context will be generic."
        )

    structured_rows = []
    for row_idx, row in enumerate(rows[1:], start=2):
        # Skip completely empty rows
        if all(cell is None or str(cell).strip() == '' for cell in row):
            continue
        row_dict = {}
        for header, value in zip(headers, row):
            if value is not None and str(value).strip():
                row_dict[header] = str(value).strip()
        if row_dict:
            row_dict['_row_index'] = row_idx
            structured_rows.append(row_dict)

    return ExtractedDocument(
        file_path=str(file_path),
        file_format='xlsx',
        raw_text='',
        structured_rows=structured_rows,
        extraction_warnings=warnings,
    )


def extract_csv(file_path: Path) -> ExtractedDocument:
    """
    Extracts rows from a CSV file.

    Uses the csv module with automatic delimiter detection via Sniffer.
    Falls back to comma if sniffer fails.
    """
    warnings = []
    structured_rows = []

    try:
        content = file_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        content = file_path.read_text(encoding='latin-1')
        warnings.append("CSV decoded as latin-1")

    try:
        dialect = csv.Sniffer().sniff(content[:1024])
    except csv.Error:
        dialect = csv.excel
        warnings.append("Could not detect CSV delimiter — assuming comma")

    reader = csv.DictReader(content.splitlines(), dialect=dialect)
    for row_idx, row in enumerate(reader, start=2):
        cleaned = {
            k.strip(): v.strip()
            for k, v in row.items()
            if k and v and v.strip()
        }
        if cleaned:
            cleaned['_row_index'] = row_idx
            structured_rows.append(cleaned)

    return ExtractedDocument(
        file_path=str(file_path),
        file_format='csv',
        raw_text='',
        structured_rows=structured_rows,
        extraction_warnings=warnings,
    )


# ── Quality Gate 1 — extraction ───────────────────────────────────────

def gate_1_extraction(doc: ExtractedDocument) -> tuple[bool, str]:
    """
    Validates extraction quality before chunking.

    Returns (passed, reason). If not passed, reason explains what failed.
    """
    # Tabular docs — check row count
    if doc.file_format in ('xlsx', 'csv'):
        if not doc.structured_rows:
            return False, "No data rows extracted from tabular file"
        return True, ""

    # Narrative docs — check text quality
    text = doc.raw_text.strip()

    if len(text) < 100:
        return False, (
            f"Extracted text too short ({len(text)} chars). "
            f"Minimum is 100. File may be empty or extraction failed."
        )

    alpha_chars = sum(c.isalpha() for c in text)
    alpha_ratio = alpha_chars / len(text) if text else 0
    if alpha_ratio < 0.40:
        return False, (
            f"Alpha character ratio {alpha_ratio:.2f} below threshold 0.40. "
            f"File may contain mostly symbols, numbers, or extraction artifacts."
        )

    # Scanned PDF — all pages empty
    if doc.file_format == 'pdf':
        scanned_warning = any(
            'scanned' in w.lower() for w in doc.extraction_warnings
        )
        if scanned_warning:
            return False, (
                "PDF appears to be scanned (no embedded text). "
                "OCR is not yet supported. Convert to text-based PDF."
            )

    return True, ""


# ── Structure detector ────────────────────────────────────────────────

def detect_content_structure(text: str) -> str:
    """
    Detects the dominant content structure of a text block.

    Returns: 'narrative', 'list', or 'mixed'

    Decision logic based on measured signal values across the corpus:

    document                sentence_ratio  list_signal  correct
    ─────────────────────────────────────────────────────────────
    sop_acos_management     0.276           0.517        narrative
    sop_buy_box_recovery    0.370           0.963        narrative
    compliance_restricted   0.143           0.964        list
    compliance_image        0.750           0.542        narrative
    framework_weekly        0.407           0.370        narrative
    transcript_natura       0.391           0.217        narrative

    Key insight: compliance term lists have sentence_ratio < 0.20
    AND list_signal > 0.80. SOPs with bullet points have
    sentence_ratio > 0.20 even when list_signal is high.

    Thresholds derived from data, not guessed:
    - narrative: sentence_ratio >= 0.25  (catches ACOS SOP at 0.276)
    - list:      sentence_ratio < 0.20 AND list_signal >= 0.80
    - mixed:     everything else
    """
    lines = [l for l in text.split('\n') if l.strip()]
    if not lines:
        return 'narrative'

    n = len(lines)
    sentence_endings = sum(
        1 for l in lines if l.rstrip().endswith(('.', '!', '?'))
    )
    colon_endings = sum(
        1 for l in lines if l.rstrip().endswith(':')
    )
    indented = sum(
        1 for l in lines if l != l.lstrip()
    )
    short_lines = sum(
        1 for l in lines if len(l.strip()) < 50
    )

    sentence_ratio = sentence_endings / n
    list_signal = (colon_endings + indented + short_lines) / n

    # Narrative: enough sentence structure — includes SOPs with bullets
    if sentence_ratio >= 0.25:
        return 'narrative'

    # Pure list: very low sentence ratio AND strong list signal
    # Only compliance-style term lists meet both conditions
    if sentence_ratio < 0.20 and list_signal >= 0.80:
        return 'list'

    # Mixed: ambiguous — default to narrative for sentence_window
    # sentence_window degrades less severely than paragraph on mixed content
    return 'narrative'


# ── Chunking strategies ───────────────────────────────────────────────

def _is_section_header(line: str) -> bool:
    """Validated section header detector from corpus inventory."""
    stripped = line.strip()
    if not stripped: return False
    if '|' in stripped: return False
    if stripped.startswith('[') and ']' in stripped[:10]: return False
    if stripped.startswith('('): return False
    if stripped.endswith(':'): return False
    if stripped.endswith(('.', '?', '!')): return False
    if re.match(r'^Step\s+\d+\s*—', stripped): return False
    if re.match(r'^[a-z]\)', stripped): return False
    if stripped.startswith('- '): return False
    if re.match(r'^[a-z].+—.+[a-z]', stripped): return False
    if stripped.startswith('Example:'): return False
    if stripped.endswith(','): return False
    if stripped != stripped.lstrip(): return False
    if len(stripped) > 100: return False
    return True


def chunk_sentence_window(
    text: str,
    source_file: str,
    doc_type: str,
    client_id: str,
    document_title: str = "",
    window_size: int = 2,
) -> list[Chunk]:
    """
    Sentence-window chunking for narrative documents.

    Embed text  = the sentence (precise retrieval signal)
    Window      = 2 sentences before + sentence + 2 after (LLM context)

    Preserves section titles for contextual enrichment (D2 representation).
    """
    # Split on sentence boundaries — raw sentences preserve line structure
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    raw_sentences = [s for s in raw_sentences if len(s.strip()) > 20]

    # Build section map using raw sentences (line structure intact)
    lines = text.split('\n')
    current_section = ""
    prev_blank = True
    line_to_section = {}
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            prev_blank = True
        elif prev_blank and _is_section_header(stripped):
            current_section = stripped
            prev_blank = False
        else:
            prev_blank = False
        line_to_section[i] = current_section

    if not document_title:
        for line in lines:
            stripped = line.strip()
            if stripped and '|' not in stripped:
                document_title = stripped
                break

    # Map each raw sentence to its section before normalisation
    sentence_sections = []
    for raw_sent in raw_sentences:
        section = ""
        search = raw_sent.strip()[:30]
        for line_idx, line in enumerate(lines):
            if search in line:
                section = line_to_section.get(line_idx, "")
                break
        sentence_sections.append(section)

    # Now normalise sentences for clean chunk text
    sentences = [' '.join(s.split()) for s in raw_sentences]

    if not sentences:
        return []

    # Detect section for each sentence position
    lines = text.split('\n')
    current_section = ""
    line_to_section = {}
    prev_blank = True
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            prev_blank = True
        elif prev_blank and _is_section_header(stripped):
            current_section = stripped
            prev_blank = False
        else:
            prev_blank = False
        line_to_section[i] = current_section

    # Extract document title (first non-empty non-metadata line)
    if not document_title:
        for line in lines:
            stripped = line.strip()
            if stripped and '|' not in stripped:
                document_title = stripped
                break

    chunks = []
    for i, sentence in enumerate(sentences):
        window_start = max(0, i - window_size)
        window_end = min(len(sentences), i + window_size + 1)
        window_context = " ".join(sentences[window_start:window_end])

        chunk_id = f"{source_file}_sentence_window_{i}"

        chunks.append(Chunk(
            chunk_id=chunk_id,
            chunk_text=sentence,
            window_context=window_context,
            source_file=source_file,
            doc_type=doc_type,
            client_id=client_id,
            chunk_index=i,
            total_chunks=len(sentences),
            chunk_strategy='sentence_window',
            structure_type='narrative',
            section_title=sentence_sections[i],    # ← pre-computed
            document_title=document_title,
        ))

    return chunks


def chunk_paragraph(
    text: str,
    source_file: str,
    doc_type: str,
    client_id: str,
    document_title: str = "",
) -> list[Chunk]:
    """
    Paragraph chunking for list/term-structured documents.

    Splits on double newlines (blank lines between paragraphs).
    Each paragraph becomes one chunk — preserves logical groupings
    like "Health and medical claims: cure, treat, prevent..."
    as a single retrievable unit rather than fragmenting across
    multiple sentence chunks.

    This is the fix for the compliance document chunking failure.
    The sentence chunker produced 4 chunks from the restricted
    claims document. Paragraph chunking produces ~9 chunks, each
    covering one logical category of prohibited terms.

    Why paragraph chunking works here:
    - The document uses blank lines as semantic separators
    - Each paragraph is one logical category (health claims,
      superlatives, guarantee language)
    - Retrieval for "what guarantee language is prohibited" correctly
      returns the guarantee paragraph, not a 738-char chunk mixing
      guarantee, conditional terms, and compliance process
    """
    # Primary split: blank lines
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip() and len(p.strip()) > 20]

    # Fallback: if blank-line split produces only 1 block,
    # split on section headers instead
    # This handles PDFs where pdfplumber strips blank lines
    if len(paragraphs) <= 1 and text.strip():
        lines = text.split('\n')
        current_block = []
        blocks = []
        prev_blank = True
        for line in lines:
            stripped = line.strip()
            if not stripped:
                prev_blank = True
                continue
            if prev_blank and _is_section_header(stripped):
                if current_block:
                    blocks.append('\n'.join(current_block))
                current_block = [stripped]
                prev_blank = False
            else:
                current_block.append(stripped)
                prev_blank = False
        if current_block:
            blocks.append('\n'.join(current_block))
        paragraphs = [b for b in blocks if len(b.strip()) > 20]

    if not paragraphs:
        return []

    # Extract document title
    if not document_title and paragraphs:
        first = paragraphs[0]
        for line in first.split('\n'):
            stripped = line.strip()
            if stripped and '|' not in stripped:
                document_title = stripped
                break

    # Detect section context per paragraph
    lines = text.split('\n')
    current_section = ""
    prev_blank = True
    line_to_section = {}
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            prev_blank = True
        elif prev_blank and _is_section_header(stripped):
            current_section = stripped
            prev_blank = False
        else:
            prev_blank = False
        line_to_section[i] = current_section

    chunks = []
    for i, para in enumerate(paragraphs):
        # Find section for this paragraph
        section = ""
        para_start = para[:30]
        for line_idx, line in enumerate(lines):
            if para_start in line:
                section = line_to_section.get(line_idx, "")
                break

        chunk_id = f"{source_file}_paragraph_{i}"

        chunks.append(Chunk(
            chunk_id=chunk_id,
            chunk_text=para,
            window_context=para,  # paragraph is its own context
            source_file=source_file,
            doc_type=doc_type,
            client_id=client_id,
            chunk_index=i,
            total_chunks=len(paragraphs),
            chunk_strategy='paragraph',
            structure_type='list',
            section_title=section,
            document_title=document_title,
        ))

    return chunks


def chunk_rows(
    rows: list[dict],
    source_file: str,
    doc_type: str,
    client_id: str,
    document_title: str = "",
) -> list[Chunk]:
    """
    Row-context chunking for tabular data.

    Each row becomes one chunk. Column headers are prepended to each
    value to make the chunk self-describing.

    Example output for a campaign performance row:
    "Client: Natura Skincare | Week: 2024-01-08 |
     Campaign Type: Sponsored Products | Spend ($): 1204.50 |
     Revenue ($): 4301.79 | ACOS (%): 28.0"

    Why row-context chunking?
    Without column headers, a value like "28.0" is meaningless.
    With headers, "ACOS (%): 28.0" is retrievable — a user asking
    "what was Natura's ACOS last week" gets a chunk that contains
    both the client name and the ACOS value in context.

    Limitations (documented, not hidden):
    - Aggregations across rows ("total spend for Natura") require
      the system to retrieve multiple chunks and sum — the generation
      layer must handle this, not the retrieval layer
    - Row order is not preserved in retrieval — temporal ordering
      of weekly data is not guaranteed in the retrieved context
    """
    if not rows:
        return []

    # Exclude internal metadata keys
    skip_keys = {'_row_index'}

    chunks = []
    for i, row in enumerate(rows):
        parts = []
        for k, v in row.items():
            if k not in skip_keys and v:
                parts.append(f"{k}: {v}")

        chunk_text = " | ".join(parts)

        if len(chunk_text) < 20:
            continue

        chunk_id = f"{source_file}_row_{i}"
        chunks.append(Chunk(
            chunk_id=chunk_id,
            chunk_text=chunk_text,
            source_file=source_file,
            doc_type=doc_type,
            client_id=client_id,
            chunk_index=i,
            total_chunks=len(rows),
            chunk_strategy='row_context',
            structure_type='tabular',
            document_title=document_title,
        ))

    return chunks


# ── Chunking router ───────────────────────────────────────────────────

def route_and_chunk(
    doc: ExtractedDocument,
    source_file: str,
    doc_type: str,
    client_id: str,
) -> list[Chunk]:
    """
    Routes a document to the appropriate chunking strategy based on
    content structure detection.

    For tabular formats: always row_context — no detection needed.
    For narrative formats: detect structure, then route.

    The detection result is logged so ingestion reports show which
    strategy was applied and why — making the pipeline auditable.
    """
    file_name = Path(source_file).name

    # Tabular formats — no detection needed
    if doc.file_format in ('xlsx', 'csv'):
        logger.info(
            f"  {file_name}: tabular format → row_context chunking "
            f"({len(doc.structured_rows)} rows)"
        )
        return chunk_rows(
            doc.structured_rows, source_file, doc_type, client_id
        )

    # Narrative formats — detect structure
    structure = detect_content_structure(doc.raw_text)
    logger.info(
        f"  {file_name}: detected structure='{structure}' "
        f"→ {'sentence_window' if structure == 'narrative' else 'paragraph'} chunking"
    )

    if structure == 'narrative' or structure == 'mixed':
        # mixed defaults to sentence_window — handles transcripts
        # and documents that blend narrative with some list elements
        return chunk_sentence_window(
            doc.raw_text, source_file, doc_type, client_id
        )
    else:
        # list/term structure → paragraph chunking
        return chunk_paragraph(
            doc.raw_text, source_file, doc_type, client_id
        )


# ── Quality Gate 2 — chunk validation ────────────────────────────────

def gate_2_chunks(chunks: list[Chunk]) -> tuple[bool, str, list[str]]:
    """
    Validates all chunks from a document before any are ingested.

    Returns (passed, failure_reason, warnings).

    Why validate all or reject all?
    Partial ingestion means retrieval has an incomplete view of a
    document. A compliance document missing two chunks might return
    an answer that omits key prohibited terms — a failure the system
    would not detect. Complete rejection forces the document to be
    fixed at the source.

    Checks:
    1. Min/max chunk length
    2. Alpha character ratio
    3. Truncation indicators (ends mid-sentence)
    4. Duplicate chunk IDs within the document
    5. At least one chunk produced
    """
    if not chunks:
        return False, "No chunks produced after chunking", []

    MIN_CHARS = 15
    MAX_CHARS = 1500
    MIN_ALPHA = 0.35

    warnings = []
    seen_ids = set()

    for chunk in chunks:
        text = chunk.chunk_text

        # Length bounds
        if len(text) < MIN_CHARS:
            return False, (
                f"Chunk {chunk.chunk_id} is too short ({len(text)} chars). "
                f"Minimum is {MIN_CHARS}. Check chunking strategy."
            ), warnings

        if len(text) > MAX_CHARS:
            return False, (
                f"Chunk {chunk.chunk_id} is too long ({len(text)} chars). "
                f"Maximum is {MAX_CHARS}. Chunking strategy may have failed — "
                f"possible structure detection error."
            ), warnings

        # Alpha ratio
        alpha = sum(c.isalpha() for c in text) / len(text)
        if alpha < MIN_ALPHA:
            return False, (
                f"Chunk {chunk.chunk_id} has low alpha ratio ({alpha:.2f}). "
                f"Minimum is {MIN_ALPHA}. May be extraction artifact."
            ), warnings

        # Truncation indicators
        if text.rstrip().endswith((',', '—', ' and', ' or', ' the')):
            warnings.append(
                f"Chunk {chunk.chunk_id} may be truncated "
                f"(ends with: '{text[-10:]}')"
            )

        # Duplicate chunk IDs
        if chunk.chunk_id in seen_ids:
            return False, (
                f"Duplicate chunk_id: {chunk.chunk_id}. "
                f"This indicates a chunking logic error."
            ), warnings
        seen_ids.add(chunk.chunk_id)

    return True, "", warnings


# ── Document processor ────────────────────────────────────────────────

EXTRACTORS = {
    '.docx': extract_docx,
    '.pdf':  extract_pdf,
    '.txt':  extract_txt,
    '.xlsx': extract_xlsx,
    '.csv':  extract_csv,
}

SUPPORTED_EXTENSIONS = set(EXTRACTORS.keys())


def infer_doc_type(file_path: Path) -> str:
    """
    Infers document type from filename patterns.
    Used when metadata_index.json does not specify doc_type.
    """
    name = file_path.stem.lower()
    if name.startswith('sop_'):
        return 'sop'
    if name.startswith('compliance_'):
        return 'compliance'
    if name.startswith('framework_'):
        return 'framework'
    if name.startswith('transcript_'):
        return 'transcript'
    if any(kw in name for kw in ['campaign', 'keyword', 'catalogue', 'catalog']):
        return 'report'
    if any(kw in name for kw in ['acos', 'target', 'performance']):
        return 'report'
    return 'document'


def process_document(
    file_path: Path,
    client_id: str = "global",
    doc_type: Optional[str] = None,
) -> IngestionResult:
    """
    Processes a single document through the full ingestion pipeline.

    Stages:
    1. Format check — is this a supported format?
    2. Extraction — get clean text from the file
    3. Gate 1 — is the extracted content usable?
    4. Structure detection + chunking routing
    5. Chunking — produce candidate chunks
    6. Gate 2 — are all chunks valid?
    7. Return IngestionResult

    Every failure returns a complete IngestionResult with status='failed'
    and a specific failure_reason. No partial results, no silent failures.
    """
    result = IngestionResult(
        file_path=str(file_path),
        file_name=file_path.name,
    )

    # ── Stage 1: Format check ─────────────────────────────────────────
    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        result.status = 'failed'
        result.failure_reason = (
            f"Unsupported format: '{ext}'. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
        return result

    # ── Stage 2: Extraction ───────────────────────────────────────────
    extractor = EXTRACTORS[ext]
    try:
        doc = extractor(file_path)
    except Exception as e:
        result.status = 'failed'
        result.failure_reason = f"Extraction failed: {type(e).__name__}: {e}"
        return result

    result.warnings.extend(doc.extraction_warnings)

    # ── Stage 3: Gate 1 ───────────────────────────────────────────────
    gate1_passed, gate1_reason = gate_1_extraction(doc)
    if not gate1_passed:
        result.status = 'failed'
        result.failure_reason = f"Extraction quality gate failed: {gate1_reason}"
        return result

    # ── Stage 4-5: Structure detection + chunking ─────────────────────
    inferred_doc_type = doc_type or infer_doc_type(file_path)
    source_file = file_path.name

    try:
        chunks = route_and_chunk(doc, source_file, inferred_doc_type, client_id)
    except Exception as e:
        result.status = 'failed'
        result.failure_reason = f"Chunking failed: {type(e).__name__}: {e}"
        return result

    # ── Stage 6: Gate 2 ───────────────────────────────────────────────
    gate2_passed, gate2_reason, chunk_warnings = gate_2_chunks(chunks)
    result.warnings.extend(chunk_warnings)

    if not gate2_passed:
        result.status = 'failed'
        result.failure_reason = f"Chunk quality gate failed: {gate2_reason}"
        result.chunks_rejected = len(chunks)
        return result

    # ── Stage 7: Success ──────────────────────────────────────────────
    result.status = 'success'
    result.chunks_produced = len(chunks)
    result.chunks = chunks
    return result


# ── Batch processor ───────────────────────────────────────────────────

def process_directory(
    input_path: Path,
    client_id: str = "global",
) -> list[IngestionResult]:
    """
    Processes all supported documents in a directory recursively.
    Returns one IngestionResult per file.
    """
    files = [
        f for f in sorted(input_path.rglob('*'))
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        logger.warning(f"No supported files found in {input_path}")
        return []

    logger.info(f"Found {len(files)} files to process")
    results = []

    for file_path in files:
        logger.info(f"Processing: {file_path.name}")
        result = process_document(file_path, client_id=client_id)
        results.append(result)

        if result.status == 'success':
            logger.info(
                f"  ✓ {result.file_name}: "
                f"{result.chunks_produced} chunks"
            )
        else:
            logger.error(
                f"  ✗ {result.file_name}: {result.failure_reason}"
            )

        for warning in result.warnings:
            logger.warning(f"  ⚠ {warning}")

    return results


# ── Ingestion report ──────────────────────────────────────────────────

def print_ingestion_report(results: list[IngestionResult]) -> None:
    """
    Prints a structured ingestion report and returns exit code.
    Failed documents are listed with their specific failure reasons
    so the submitter knows exactly what to fix.
    """
    succeeded = [r for r in results if r.status == 'success']
    failed = [r for r in results if r.status == 'failed']
    total_chunks = sum(r.chunks_produced for r in succeeded)

    print(f"\n{'='*60}")
    print(f"INGESTION REPORT")
    print(f"{'='*60}")
    print(f"Documents processed : {len(results)}")
    print(f"Succeeded           : {len(succeeded)}")
    print(f"Failed              : {len(failed)}")
    print(f"Total chunks        : {total_chunks}")

    if succeeded:
        print(f"\n── Succeeded ────────────────────────────────────────")
        for r in succeeded:
            strategy = r.chunks[0].chunk_strategy if r.chunks else 'unknown'
            structure = r.chunks[0].structure_type if r.chunks else 'unknown'
            print(
                f"  ✓ {r.file_name:<45} "
                f"{r.chunks_produced:>4} chunks  "
                f"[{strategy}/{structure}]"
            )
            for w in r.warnings:
                print(f"    ⚠ {w}")

    if failed:
        print(f"\n── Failed — requires attention ──────────────────────")
        for r in failed:
            print(f"  ✗ {r.file_name}")
            print(f"    Reason: {r.failure_reason}")

    print(f"\n{'='*60}")

    if failed:
        print(f"\n{len(failed)} document(s) failed ingestion.")
        print("Fix the issues above and resubmit.")


def save_chunks_json(
    results: list[IngestionResult],
    output_path: Path,
) -> None:
    """Saves all successful chunks to JSON for embedding."""
    all_chunks = []
    for result in results:
        if result.status == 'success':
            for chunk in result.chunks:
                all_chunks.append({
                    'chunk_id':       chunk.chunk_id,
                    'text':           chunk.chunk_text,
                    'window_context': chunk.window_context,
                    'source_file':    chunk.source_file,
                    'doc_type':       chunk.doc_type,
                    'client':         chunk.client_id,
                    'chunk_index':    chunk.chunk_index,
                    'total_chunks':   chunk.total_chunks,
                    'chunk_strategy': chunk.chunk_strategy,
                    'structure_type': chunk.structure_type,
                    'section_title':  chunk.section_title,
                    'document_title': chunk.document_title,
                    'char_count':     chunk.char_count,
                    'token_estimate': chunk.token_estimate,
                })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_chunks, f, indent=2)

    logger.info(f"Saved {len(all_chunks)} chunks → {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="KIRA document ingestion pipeline"
    )
    parser.add_argument(
        '--input', required=True,
        help="File or directory to ingest"
    )
    parser.add_argument(
        '--client', default='global',
        help="Client ID for namespace routing (default: global)"
    )
    parser.add_argument(
        '--output',
        help="Output JSON path for chunks (default: data/chunks/pipeline.json)"
    )
    parser.add_argument(
        '--doc-type',
        help="Override doc_type (default: inferred from filename)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else Path('data/chunks/pipeline.json')

    if input_path.is_file():
        result = process_document(
            input_path,
            client_id=args.client,
            doc_type=args.doc_type,
        )
        results = [result]
    elif input_path.is_dir():
        results = process_directory(input_path, client_id=args.client)
    else:
        print(f"Error: {input_path} does not exist")
        sys.exit(1)

    print_ingestion_report(results)
    save_chunks_json(results, output_path)

    # Exit 1 if any failures — enables CI integration
    failed_count = sum(1 for r in results if r.status == 'failed')
    sys.exit(1 if failed_count > 0 else 0)