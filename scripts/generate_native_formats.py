"""
scripts/generate_native_formats.py

Converts KIRA plain text source documents to their natural file formats
and generates tabular data that Amazon agencies actually use.

Run from project root:
    python scripts/generate_native_formats.py

Output structure:
    data/raw/
      documents/
        sop/                    ← DOCX files
        compliance/             ← PDF files
        frameworks/             ← DOCX files
        transcripts/            ← TXT files (copied)
      tabular/                  ← XLSX and CSV files
"""

import csv
import random
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path

# ── DOCX generation ───────────────────────────────────────────────────

from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH


def is_section_header(line: str) -> bool:
    """
    Validated section header detector from corpus inventory work.
    Identifies section headers in SOP and framework documents.
    """
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


def make_docx_from_txt(txt_path: Path, docx_path: Path) -> None:
    """
    Converts a plain text SOP or framework to a structured DOCX.

    Why DOCX for SOPs and frameworks?
    In Amazon agencies, SOPs are written and shared as Word documents.
    They get emailed, edited, version-controlled in SharePoint or Google Drive.
    Producing DOCX tests the extraction path that real agency documents
    will actually take.

    Design decisions:
    - Section headers detected using the validated corpus detector
    - Step headers get Heading 2 style
    - Metadata lines (Version, Owner) get small grey text
    - List items get List Bullet style
    - No \n in paragraphs — docx-js/python-docx requirement
    """
    text = txt_path.read_text()
    lines = text.split('\n')

    doc = Document()

    # Set default font
    style = doc.styles['Normal']
    style.font.name = 'Arial'
    style.font.size = Pt(11)

    first_line_done = False
    prev_blank = True

    for line in lines:
        stripped = line.strip()

        if not stripped:
            prev_blank = True
            continue

        # Document title — first non-empty line containing an em dash
        if not first_line_done and '—' in stripped:
            p = doc.add_heading(stripped, level=0)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            first_line_done = True
            prev_blank = False
            continue

        first_line_done = True

        # Metadata line (Version | Owner | Date)
        if '|' in stripped and ('Version' in stripped or 'Owner' in stripped):
            p = doc.add_paragraph()
            run = p.add_run(stripped)
            run.font.size = Pt(9)
            run.font.color.rgb = RGBColor(0x80, 0x80, 0x80)
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            prev_blank = False
            continue

        # Section header — must be preceded by blank line
        if prev_blank and is_section_header(stripped):
            doc.add_heading(stripped, level=1)
            prev_blank = False
            continue

        # Step header within a section
        if re.match(r'^Step\s+\d+\s*—', stripped):
            doc.add_heading(stripped, level=2)
            prev_blank = False
            continue

        # List items
        if (stripped.startswith('- ') or
                re.match(r'^[a-e]\)', stripped) or
                stripped.startswith('[') or
                (line != line.lstrip() and stripped)):
            doc.add_paragraph(stripped, style='List Bullet')
            prev_blank = False
            continue

        # Regular body text
        doc.add_paragraph(stripped)
        prev_blank = False

    doc.save(docx_path)
    size = docx_path.stat().st_size
    print(f"  DOCX: {docx_path.name} ({size:,} bytes)")


# ── PDF generation ────────────────────────────────────────────────────

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER


def make_pdf_from_txt(txt_path: Path, pdf_path: Path) -> None:
    """
    Converts a plain text compliance document to PDF.

    Why PDF for compliance docs?
    Compliance documents are distributed as PDFs in agencies because:
    1. PDF prevents casual editing — compliance rules should not be modified
    2. PDFs are the standard for regulatory and policy documents
    3. They're emailed to clients, printed, signed

    This tests the PDF extraction path in the ingestion pipeline.
    pdfplumber will extract the text, which is then chunked.

    The compliance document uses list/term structure — this is exactly
    the format that caused our 4-chunk problem with sentence chunking.
    As a PDF, it must go through extraction before chunking, giving us
    the opportunity to detect its structure and apply paragraph chunking.
    """
    text = txt_path.read_text()

    processed_lines = []
    lines = text.split('\n')
    for i, line in enumerate(lines):
        processed_lines.append(line)
        stripped = line.strip()
        # After a parenthetical qualifier, insert blank line
        if stripped.startswith('(unless') or stripped.startswith('(only'):
            processed_lines.append('')
        # Before a sub-header (colon-ending non-indented line),
        # insert blank line if previous line wasn't blank
        elif (stripped.endswith(':') and
              line == line.lstrip() and
              i > 0 and
              lines[i-1].strip() != ''):
            processed_lines.insert(len(processed_lines) - 1, '')

    lines = text.split('\n')

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=inch,
        rightMargin=inch,
        topMargin=inch,
        bottomMargin=inch,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Normal'],
        fontSize=15, fontName='Helvetica-Bold',
        spaceAfter=4, alignment=TA_CENTER,
    )
    meta_style = ParagraphStyle(
        'Meta', parent=styles['Normal'],
        fontSize=8, textColor=colors.grey,
        spaceAfter=10, alignment=TA_CENTER,
    )
    section_style = ParagraphStyle(
        'Section', parent=styles['Normal'],
        fontSize=12, fontName='Helvetica-Bold',
        spaceBefore=14, spaceAfter=4,
    )
    sub_header_style = ParagraphStyle(
        'SubHeader', parent=styles['Normal'],
        fontSize=10, fontName='Helvetica-Bold',
        spaceBefore=8, spaceAfter=2,
    )
    body_style = ParagraphStyle(
        'Body', parent=styles['Normal'],
        fontSize=10, spaceAfter=4, leading=14,
    )
    term_style = ParagraphStyle(
        'Term', parent=styles['Normal'],
        fontSize=10, leftIndent=20, spaceAfter=2,
        fontName='Helvetica-Oblique',
    )

    story = []
    first_line_done = False
    prev_blank = True

    for line in lines:
        stripped = line.strip()

        if not stripped:
            prev_blank = True
            continue

        # Document title
        if not first_line_done:
            story.append(Paragraph(stripped, title_style))
            first_line_done = True
            prev_blank = False
            continue

        # Metadata
        if '|' in stripped:
            story.append(Paragraph(stripped, meta_style))
            story.append(HRFlowable(
                width="100%", thickness=0.5,
                color=colors.lightgrey
            ))
            story.append(Spacer(1, 4))
            prev_blank = False
            continue

        # Section header
        if prev_blank and is_section_header(stripped):
            story.append(Paragraph(stripped, section_style))
            prev_blank = False
            continue

        # Sub-headers end with colon and are not indented
        if stripped.endswith(':') and line == line.lstrip():
            # Empty paragraph creates a real blank line in extracted text
            # Spacer(1, 8) only adds visual space — pdfplumber ignores it
            story.append(Paragraph("&nbsp;", body_style))
            story.append(Paragraph(stripped, sub_header_style))
            prev_blank = False
            continue

        

        # Indented term items
        if line != line.lstrip() and stripped:
            story.append(Paragraph(stripped, term_style))
            if stripped.startswith('(unless') or stripped.startswith('(only'):
                story.append(Paragraph("&nbsp;", body_style))
            prev_blank = False
            continue

        story.append(Paragraph(stripped, body_style))
        prev_blank = False

    doc.build(story)
    size = pdf_path.stat().st_size
    print(f"  PDF:  {pdf_path.name} ({size:,} bytes)")


# ── Tabular data ──────────────────────────────────────────────────────

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

random.seed(42)

CLIENTS = [
    'Natura Skincare',
    'VitalBlend Supplements',
    'PeakGear Outdoors',
    'Lumina Home Lighting',
]
CLIENT_IDS = ['natura', 'vitalblend', 'peakgear', 'lumina']


def style_header(ws, row: int = 1) -> None:
    fill = PatternFill(
        start_color="1a1a2e", end_color="1a1a2e", fill_type="solid"
    )
    font = Font(bold=True, color="FFFFFF", name='Arial', size=10)
    for cell in ws[row]:
        cell.fill = fill
        cell.font = font
        cell.alignment = Alignment(horizontal='center', vertical='center')


def auto_width(ws) -> None:
    for col in ws.columns:
        max_len = max(
            (len(str(cell.value or '')) for cell in col), default=0
        )
        ws.column_dimensions[
            get_column_letter(col[0].column)
        ].width = min(max_len + 4, 45)


def make_campaign_performance_xlsx(path: Path) -> None:
    """
    Weekly campaign performance report.

    This is the most common tabular document in Amazon agencies.
    Account managers export this from Amazon Ads console weekly.
    The ingestion pipeline must handle it as structured tabular data —
    each row is a record, not a sentence to be chunked.

    What makes this a challenge for naive RAG:
    - No sentence structure
    - Numeric data mixed with categorical
    - Column headers carry the semantic meaning
    - A user asking "what was Natura's ACOS last week" needs
      the system to understand row + column relationships

    For the ingestion pipeline: row-based chunking where each row
    becomes a chunk with column context prepended.
    Example chunk: "Client: Natura Skincare | Week: 2024-03-11 |
    Campaign: Natura - Sponsored Products | Spend: $1204.50 |
    ACOS: 28.3%"
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Campaign Performance"

    headers = [
        'Week', 'Client', 'Campaign Name', 'Campaign Type',
        'Spend ($)', 'Revenue ($)', 'ACOS (%)',
        'Impressions', 'Clicks', 'CTR (%)', 'CPC ($)',
        'Units Sold', 'ROAS',
    ]
    ws.append(headers)
    style_header(ws)

    start_date = datetime(2024, 1, 1)
    campaign_types = [
        'Sponsored Products', 'Sponsored Brands', 'Sponsored Display'
    ]

    row_count = 0
    for week_offset in range(12):
        week_label = (
            start_date + timedelta(weeks=week_offset)
        ).strftime('%Y-%m-%d')
        for client, cid in zip(CLIENTS, CLIENT_IDS):
            for c_type in campaign_types:
                spend = round(random.uniform(200, 2000), 2)
                acos = round(random.uniform(0.18, 0.45), 3)
                revenue = round(spend / acos, 2)
                impressions = random.randint(5000, 50000)
                clicks = random.randint(100, 2000)
                units = random.randint(10, 200)
                ws.append([
                    week_label, client,
                    f"{client.split()[0]} - {c_type} - Auto",
                    c_type,
                    spend, revenue,
                    round(acos * 100, 1),
                    impressions, clicks,
                    round(clicks / impressions * 100, 2),
                    round(spend / clicks, 2),
                    units,
                    round(revenue / spend, 2),
                ])
                row_count += 1

    auto_width(ws)
    wb.save(path)
    print(f"  XLSX: {path.name} ({row_count} rows)")


def make_keyword_tracker_xlsx(path: Path) -> None:
    """
    Keyword ranking tracker.
    Agencies track primary keyword positions weekly.
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Keyword Tracker"

    headers = [
        'Client', 'ASIN', 'Keyword', 'Match Type',
        'Monthly Search Volume', 'Current Rank', 'Previous Rank',
        'Rank Change', 'Bid ($)', 'Relevance Score',
        'Priority', 'Notes',
    ]
    ws.append(headers)
    style_header(ws)

    kw_data = {
        'Natura Skincare': [
            ('B08XK9L2MN', 'vitamin c serum for face', 3200),
            ('B08XK9L2MN', 'vitamin c serum brightening', 1800),
            ('B08XK9L2MN', 'natural vitamin c serum ewg', 900),
            ('B08XK9L2MN', 'eye cream for dark circles', 4100),
        ],
        'VitalBlend Supplements': [
            ('B09KL3MN2P', 'collagen peptides powder', 8900),
            ('B09KL3MN2P', 'grass fed collagen peptides', 3200),
            ('B09KL3MN2P', 'magnesium glycinate 400mg', 5600),
            ('B09KL3MN2P', 'vitamin d3 k2 supplement', 4300),
        ],
        'PeakGear Outdoors': [
            ('B07WQ2KP4M', '20 degree sleeping bag', 2100),
            ('B07WQ2KP4M', 'backpacking sleeping bag lightweight', 5400),
            ('B07WQ2KP4M', 'camping sleeping bag cold weather', 1900),
        ],
        'Lumina Home Lighting': [
            ('B0CK2L9MNP', 'pendant light living room', 3800),
            ('B0CK2L9MNP', 'brass pendant light fixture', 2100),
            ('B0CK2L9MNP', 'mid century modern pendant light', 1400),
        ],
    }

    for client in CLIENTS:
        for asin, keyword, volume in kw_data[client]:
            current = random.randint(1, 50)
            previous = current + random.randint(-8, 8)
            priority = (
                'High' if volume > 3000
                else 'Medium' if volume > 1000
                else 'Low'
            )
            ws.append([
                client, asin, keyword,
                random.choice(['Exact', 'Phrase', 'Broad']),
                volume, current, previous,
                previous - current,
                round(random.uniform(0.50, 3.50), 2),
                round(random.uniform(0.6, 1.0), 2),
                priority, '',
            ])

    auto_width(ws)
    wb.save(path)
    print(f"  XLSX: {path.name}")


def make_product_catalogue_xlsx(path: Path) -> None:
    """
    Product catalogue — full ASIN portfolio with targets and status.
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Product Catalogue"

    headers = [
        'Client', 'ASIN', 'Product Title', 'Category', 'Sub-category',
        'ASP ($)', 'Units FBA', 'Buy Box Status', 'Buy Box %',
        'Star Rating', 'Reviews', 'ACOS Target (%)',
        'Break-even ACOS (%)', 'Status', 'Notes',
    ]
    ws.append(headers)
    style_header(ws)

    products = [
        ['Natura Skincare', 'B08XK9L2MN',
         'Natura Vitamin C Brightening Serum 30ml EWG Verified',
         'Beauty', 'Serums', 42.99, 340, 'Won', 94,
         4.7, 1823, 28, 38, 'Active', ''],
        ['Natura Skincare', 'B09MK2LP3N',
         'Natura Renewal Eye Cream 15ml Dark Circles',
         'Beauty', 'Eye Treatments', 38.99, 180, 'Won', 89,
         4.4, 642, 30, 40, 'Active',
         'Image non-compliant — fix pending'],
        ['VitalBlend Supplements', 'B09KL3MN2P',
         'VitalBlend Grass-Fed Collagen Peptides Powder 1lb Unflavored',
         'Health', 'Protein Supplements', 34.99, 520, 'At Risk', 71,
         4.6, 3241, 24, 28, 'Active',
         'Third-party seller undercutting by 3%'],
        ['VitalBlend Supplements', 'B0BK2MN4P',
         'VitalBlend Magnesium Glycinate 400mg 120 Capsules',
         'Health', 'Minerals', 27.99, 290, 'Won', 96,
         4.5, 1102, 26, 30, 'Active', ''],
        ['VitalBlend Supplements', 'B0CL3MN5Q',
         'VitalBlend Vitamin D3 K2 5000IU 60 Softgels',
         'Health', 'Vitamins', 22.99, 410, 'Won', 98,
         4.3, 876, 25, 29, 'Active', ''],
        ['PeakGear Outdoors', 'B07WQ2KP4M',
         'PeakGear 20 Degree Sleeping Bag Lightweight Backpacking',
         'Sports', 'Sleeping Bags', 89.99, 210, 'Won', 97,
         4.6, 2312, 25, 32, 'Active',
         'Seasonal: ACOS target 35% Mar-Aug'],
        ['Lumina Home Lighting', 'B0CK2L9MNP',
         'Lumina Brushed Brass Pendant Light Mid-Century Modern',
         'Home', 'Ceiling Lights', 38.00, 155, 'Won', 88,
         4.5, 412, 30, 38, 'Active',
         'Image suppression Jan 2024 — resolved'],
    ]

    for row in products:
        ws.append(row)

    auto_width(ws)
    wb.save(path)
    print(f"  XLSX: {path.name}")


def make_acos_targets_csv(path: Path) -> None:
    """
    Per-client ACOS targets by account stage.
    Simple CSV — the kind a client ops team keeps in Google Sheets
    and exports as CSV before uploading to KIRA.
    """
    rows = [
        ['client_id', 'client_name', 'account_stage', 'days_range',
         'acos_target_min', 'acos_target_max', 'break_even_acos',
         'primary_objective', 'notes'],
        ['natura', 'Natura Skincare', 'Launch', '0-90',
         0.40, 0.55, 0.38, 'Ranking', 'New SKU launch'],
        ['natura', 'Natura Skincare', 'Growth', '90-365',
         0.28, 0.35, 0.38, 'Balanced', 'Standard growth target'],
        ['natura', 'Natura Skincare', 'Mature', '365+',
         0.20, 0.28, 0.38, 'Profitability', 'Maintain rank optimise margin'],
        ['vitalblend', 'VitalBlend Supplements', 'Launch', '0-90',
         0.38, 0.50, 0.28, 'Ranking', ''],
        ['vitalblend', 'VitalBlend Supplements', 'Growth', '90-365',
         0.22, 0.28, 0.28, 'Balanced',
         'Current state 34% — over target'],
        ['vitalblend', 'VitalBlend Supplements', 'Mature', '365+',
         0.18, 0.24, 0.28, 'Profitability', ''],
        ['peakgear', 'PeakGear Outdoors', 'Seasonal-Peak', 'Mar-Aug',
         0.28, 0.35, 0.32, 'Revenue',
         'Spring/summer hiking season'],
        ['peakgear', 'PeakGear Outdoors', 'Seasonal-Low', 'Sep-Feb',
         0.15, 0.20, 0.32, 'Profitability', 'Off-season reduce spend'],
        ['lumina', 'Lumina Home Lighting', 'Standard', 'All',
         0.25, 0.30, 0.38, 'Balanced',
         'Break-even 38% — never exceed without approval'],
    ]

    with open(path, 'w', newline='') as f:
        csv.writer(f).writerows(rows)

    print(f"  CSV:  {path.name} ({len(rows)-1} data rows)")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    raw_dir = Path('data/raw')

    dirs = {
        'sop':        raw_dir / 'documents' / 'sop',
        'compliance': raw_dir / 'documents' / 'compliance',
        'frameworks': raw_dir / 'documents' / 'frameworks',
        'transcripts':raw_dir / 'documents' / 'transcripts',
        'tabular':    raw_dir / 'tabular',
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    print("\n── SOPs → DOCX ──────────────────────────────────────────")
    for fn in [
        'sop_acos_management.txt',
        'sop_buy_box_recovery.txt',
        'sop_listing_optimisation.txt',
    ]:
        make_docx_from_txt(
            raw_dir / fn,
            dirs['sop'] / fn.replace('.txt', '.docx'),
        )

    print("\n── Frameworks → DOCX ────────────────────────────────────")
    for fn in [
        'framework_weekly_reporting.txt',
        'framework_anomaly_investigation.txt',
    ]:
        make_docx_from_txt(
            raw_dir / fn,
            dirs['frameworks'] / fn.replace('.txt', '.docx'),
        )

    print("\n── Compliance → PDF ─────────────────────────────────────")
    for fn in [
        'compliance_restricted_claims.txt',
        'compliance_image_standards.txt',
    ]:
        make_pdf_from_txt(
            raw_dir / fn,
            dirs['compliance'] / fn.replace('.txt', '.pdf'),
        )

    print("\n── Transcripts → TXT (copy) ─────────────────────────────")
    for fn in [
        'transcript_onboarding_natura.txt',
        'transcript_onboarding_vitalblend.txt',
        'transcript_onboarding_peakgear.txt',
        'transcript_onboarding_lumina.txt',
    ]:
        shutil.copy2(raw_dir / fn, dirs['transcripts'] / fn)
        print(f"  TXT:  {fn}")

    print("\n── Tabular data ─────────────────────────────────────────")
    make_campaign_performance_xlsx(
        dirs['tabular'] / 'campaign_performance_report.xlsx'
    )
    make_keyword_tracker_xlsx(
        dirs['tabular'] / 'keyword_tracker.xlsx'
    )
    make_product_catalogue_xlsx(
        dirs['tabular'] / 'product_catalogue.xlsx'
    )
    make_acos_targets_csv(
        dirs['tabular'] / 'acos_targets_by_client.csv'
    )

    print("\n── Output summary ───────────────────────────────────────")
    for category, d in dirs.items():
        files = sorted(d.iterdir())
        sizes = [f.stat().st_size for f in files]
        total = sum(sizes)
        print(f"  {category}/: {len(files)} files, {total:,} bytes total")
        for f, s in zip(files, sizes):
            print(f"    {f.name} ({s:,} bytes)")

    print("\nDone. Next step: build ingestion/pipeline.py")


if __name__ == "__main__":
    main()
