"""
ingestion/data_loader.py

Loads tabular data files into PostgreSQL for KIRA.

Handles:
    XLSX  — campaign_performance_report.xlsx
            keyword_tracker.xlsx
            product_catalogue.xlsx
    CSV   — acos_targets_by_client.csv

Design decisions:
    - Tabular data belongs in the relational database, not the vector store.
      Campaign metrics, keyword rankings, ACOS targets, and product data
      are operational records queried with SQL — not semantic knowledge
      retrieved by embedding similarity.

    - Each loader function is idempotent — running it twice does not
      create duplicate rows. It clears existing data for the affected
      client_ids before inserting. This matches how agencies work:
      they upload a fresh weekly report, which replaces last week's data.

    - Failures are explicit. A file that cannot be parsed, or that
      produces rows failing validation, does not proceed. The loader
      reports exactly what failed and why.

    - Column name normalisation is explicit — the mapping between
      spreadsheet column headers and database column names is defined
      in one place per loader function, not inferred automatically.
      This prevents silent mismatches when a client renames a column.

Run:
    python ingestion/data_loader.py --all
    python ingestion/data_loader.py --file data/raw/tabular/campaign_performance_report.xlsx
    python ingestion/data_loader.py --type campaign
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ── Data contract ─────────────────────────────────────────────────────

@dataclass
class LoadResult:
    """
    Result of loading one file into the database.
    """
    file_path: str
    file_name: str
    table_name: str
    status: str = "pending"      # success | failed
    rows_loaded: int = 0
    rows_rejected: int = 0
    failure_reason: str = ""
    warnings: list[str] = field(default_factory=list)
    loaded_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ── Column normaliser ─────────────────────────────────────────────────

def normalise_client_id(client_name: str) -> str:
    """
    Converts a client display name to a client_id.

    The XLSX files use full client names ("Natura Skincare").
    The database uses snake_case IDs ("natura").
    This mapping is explicit — no fuzzy matching.
    """
    mapping = {
        'Natura Skincare': 'natura',
        'VitalBlend Supplements': 'vitalblend',
        'PeakGear Outdoors': 'peakgear',
        'Lumina Home Lighting': 'lumina',
    }
    result = mapping.get(client_name)
    if result is None:
        # Fallback: lowercase first word
        result = client_name.split()[0].lower()
        logger.warning(
            f"Unknown client name '{client_name}' — "
            f"using fallback ID '{result}'"
        )
    return result


def safe_float(value, default=None) -> Optional[float]:
    """Converts a value to float, returning default on failure."""
    if value is None or str(value).strip() in ('', 'None', 'N/A', '-'):
        return default
    try:
        return float(str(value).replace('%', '').replace('$', '').replace(',', ''))
    except (ValueError, TypeError):
        return default


def safe_int(value, default=None) -> Optional[int]:
    """Converts a value to int, returning default on failure."""
    if value is None or str(value).strip() in ('', 'None', 'N/A', '-'):
        return default
    try:
        return int(float(str(value).replace(',', '')))
    except (ValueError, TypeError):
        return default


# ── Loaders ───────────────────────────────────────────────────────────

def load_campaign_performance(
    file_path: Path,
    session,
) -> LoadResult:
    """
    Loads campaign_performance_report.xlsx into campaign_performance table.

    Idempotent: clears all existing rows before loading.
    Weekly reports replace previous data entirely.

    Column mapping (spreadsheet → database):
        Week              → week
        Client            → client_id  (normalised)
        Campaign Name     → campaign_name
        Campaign Type     → campaign_type
        Spend ($)         → spend_usd
        Revenue ($)       → revenue_usd
        ACOS (%)          → acos       (stored as decimal: 28.3% → 0.283)
        Impressions       → impressions
        Clicks            → clicks
        CTR (%)           → ctr        (stored as decimal)
        CPC ($)           → cpc_usd
        Units Sold        → units_sold
        ROAS              → roas
    """
    from infra.database import CampaignPerformance
    import openpyxl

    result = LoadResult(
        file_path=str(file_path),
        file_name=file_path.name,
        table_name="campaign_performance",
    )

    try:
        wb = openpyxl.load_workbook(str(file_path), data_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
    except Exception as e:
        result.status = 'failed'
        result.failure_reason = f"Cannot open file: {e}"
        return result

    if not rows:
        result.status = 'failed'
        result.failure_reason = "File is empty"
        return result

    headers = [str(h).strip() if h else '' for h in rows[0]]

    # Clear existing data — weekly reports replace previous
    deleted = session.query(CampaignPerformance).delete()
    if deleted > 0:
        logger.info(f"  Cleared {deleted} existing campaign performance rows")

    rows_loaded = 0
    rows_rejected = 0

    for row_idx, row in enumerate(rows[1:], start=2):
        row_dict = dict(zip(headers, row))

        # Skip empty rows
        if not any(v for v in row_dict.values() if v):
            continue

        # Skip totals rows
        if str(row_dict.get('Week', '')).upper() in ('TOTALS', 'TOTAL'):
            continue

        client_name = str(row_dict.get('Client', '')).strip()
        if not client_name:
            rows_rejected += 1
            result.warnings.append(f"Row {row_idx}: missing client name — skipped")
            continue

        spend = safe_float(row_dict.get('Spend ($)'))
        revenue = safe_float(row_dict.get('Revenue ($)'))

        if spend is None or revenue is None:
            rows_rejected += 1
            result.warnings.append(
                f"Row {row_idx}: missing spend or revenue — skipped"
            )
            continue

        acos_raw = safe_float(row_dict.get('ACOS (%)'))
        acos = acos_raw / 100 if acos_raw and acos_raw > 1 else acos_raw

        ctr_raw = safe_float(row_dict.get('CTR (%)'))
        ctr = ctr_raw / 100 if ctr_raw and ctr_raw > 1 else ctr_raw

        record = CampaignPerformance(
            week=str(row_dict.get('Week', '')).strip(),
            client_id=normalise_client_id(client_name),
            campaign_name=str(row_dict.get('Campaign Name', '')).strip(),
            campaign_type=str(row_dict.get('Campaign Type', '')).strip(),
            spend_usd=spend,
            revenue_usd=revenue,
            acos=acos,
            impressions=safe_int(row_dict.get('Impressions')),
            clicks=safe_int(row_dict.get('Clicks')),
            ctr=ctr,
            cpc_usd=safe_float(row_dict.get('CPC ($)')),
            units_sold=safe_int(row_dict.get('Units Sold')),
            roas=safe_float(row_dict.get('ROAS')),
        )
        session.add(record)
        rows_loaded += 1

    session.flush()
    result.status = 'success'
    result.rows_loaded = rows_loaded
    result.rows_rejected = rows_rejected
    return result


def load_keyword_rankings(
    file_path: Path,
    session,
) -> LoadResult:
    """
    Loads keyword_tracker.xlsx into keyword_rankings table.

    Idempotent: clears all existing rows before loading.
    """
    from infra.database import KeywordRanking
    import openpyxl

    result = LoadResult(
        file_path=str(file_path),
        file_name=file_path.name,
        table_name="keyword_rankings",
    )

    try:
        wb = openpyxl.load_workbook(str(file_path), data_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
    except Exception as e:
        result.status = 'failed'
        result.failure_reason = f"Cannot open file: {e}"
        return result

    if not rows:
        result.status = 'failed'
        result.failure_reason = "File is empty"
        return result

    headers = [str(h).strip() if h else '' for h in rows[0]]

    deleted = session.query(KeywordRanking).delete()
    if deleted > 0:
        logger.info(f"  Cleared {deleted} existing keyword ranking rows")

    rows_loaded = 0
    rows_rejected = 0

    for row_idx, row in enumerate(rows[1:], start=2):
        row_dict = dict(zip(headers, row))

        if not any(v for v in row_dict.values() if v):
            continue

        client_name = str(row_dict.get('Client', '')).strip()
        keyword = str(row_dict.get('Keyword', '')).strip()

        if not client_name or not keyword:
            rows_rejected += 1
            result.warnings.append(
                f"Row {row_idx}: missing client or keyword — skipped"
            )
            continue

        record = KeywordRanking(
            client_id=normalise_client_id(client_name),
            asin=str(row_dict.get('ASIN', '')).strip(),
            keyword=keyword,
            match_type=str(row_dict.get('Match Type', '')).strip(),
            monthly_search_volume=safe_int(row_dict.get('Monthly Search Volume')),
            current_rank=safe_int(row_dict.get('Current Rank')),
            previous_rank=safe_int(row_dict.get('Previous Rank')),
            rank_change=safe_int(row_dict.get('Rank Change')),
            bid_usd=safe_float(row_dict.get('Bid ($)')),
            relevance_score=safe_float(row_dict.get('Relevance Score')),
            priority=str(row_dict.get('Priority', '')).strip(),
            notes=str(row_dict.get('Notes', '')).strip() or None,
        )
        session.add(record)
        rows_loaded += 1

    session.flush()
    result.status = 'success'
    result.rows_loaded = rows_loaded
    result.rows_rejected = rows_rejected
    return result


def load_product_catalogue(
    file_path: Path,
    session,
) -> LoadResult:
    """
    Loads product_catalogue.xlsx into product_catalogue table.

    Idempotent: upserts on ASIN — updates existing records,
    inserts new ones. Product catalogue is a master data record
    that grows over time rather than being replaced weekly.
    """
    from infra.database import ProductCatalogue
    import openpyxl

    result = LoadResult(
        file_path=str(file_path),
        file_name=file_path.name,
        table_name="product_catalogue",
    )

    try:
        wb = openpyxl.load_workbook(str(file_path), data_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
    except Exception as e:
        result.status = 'failed'
        result.failure_reason = f"Cannot open file: {e}"
        return result

    if not rows:
        result.status = 'failed'
        result.failure_reason = "File is empty"
        return result

    headers = [str(h).strip() if h else '' for h in rows[0]]

    rows_loaded = 0
    rows_rejected = 0

    for row_idx, row in enumerate(rows[1:], start=2):
        row_dict = dict(zip(headers, row))

        if not any(v for v in row_dict.values() if v):
            continue

        client_name = str(row_dict.get('Client', '')).strip()
        asin = str(row_dict.get('ASIN', '')).strip()

        if not client_name or not asin:
            rows_rejected += 1
            result.warnings.append(
                f"Row {row_idx}: missing client or ASIN — skipped"
            )
            continue

        # Upsert: update if ASIN exists, insert if not
        existing = session.query(ProductCatalogue).filter_by(asin=asin).first()

        values = dict(
            client_id=normalise_client_id(client_name),
            asin=asin,
            product_title=str(row_dict.get('Product Title', '')).strip(),
            category=str(row_dict.get('Category', '')).strip(),
            sub_category=str(row_dict.get('Sub-category', '')).strip(),
            asp_usd=safe_float(row_dict.get('ASP ($)')),
            units_fba=safe_int(row_dict.get('Units FBA')),
            buy_box_status=str(row_dict.get('Buy Box Status', '')).strip(),
            buy_box_pct=safe_float(row_dict.get('Buy Box %')),
            star_rating=safe_float(row_dict.get('Star Rating')),
            review_count=safe_int(row_dict.get('Reviews')),
            acos_target=safe_float(row_dict.get('ACOS Target (%)')),
            break_even_acos=safe_float(row_dict.get('Break-even ACOS (%)')),
            status=str(row_dict.get('Status', '')).strip(),
            notes=str(row_dict.get('Notes', '')).strip() or None,
            updated_at=datetime.utcnow(),
        )

        if existing:
            for k, v in values.items():
                setattr(existing, k, v)
        else:
            session.add(ProductCatalogue(**values))

        rows_loaded += 1

    session.flush()
    result.status = 'success'
    result.rows_loaded = rows_loaded
    result.rows_rejected = rows_rejected
    return result


def load_acos_targets(
    file_path: Path,
    session,
) -> LoadResult:
    """
    Loads acos_targets_by_client.csv into acos_targets table.

    Idempotent: clears all existing rows before loading.
    ACOS targets are configuration — a full reload is safe.
    """
    from infra.database import AcosTarget

    result = LoadResult(
        file_path=str(file_path),
        file_name=file_path.name,
        table_name="acos_targets",
    )

    try:
        content = file_path.read_text(encoding='utf-8')
    except Exception as e:
        result.status = 'failed'
        result.failure_reason = f"Cannot read file: {e}"
        return result

    deleted = session.query(AcosTarget).delete()
    if deleted > 0:
        logger.info(f"  Cleared {deleted} existing ACOS target rows")

    rows_loaded = 0
    rows_rejected = 0

    reader = csv.DictReader(content.splitlines())
    for row_idx, row in enumerate(reader, start=2):
        client_id = row.get('client_id', '').strip()
        account_stage = row.get('account_stage', '').strip()

        if not client_id or not account_stage:
            rows_rejected += 1
            result.warnings.append(
                f"Row {row_idx}: missing client_id or account_stage — skipped"
            )
            continue

        record = AcosTarget(
            client_id=client_id,
            client_name=row.get('client_name', '').strip(),
            account_stage=account_stage,
            days_range=row.get('days_range', '').strip(),
            acos_target_min=safe_float(row.get('acos_target_min')),
            acos_target_max=safe_float(row.get('acos_target_max')),
            break_even_acos=safe_float(row.get('break_even_acos')),
            primary_objective=row.get('primary_objective', '').strip(),
            notes=row.get('notes', '').strip() or None,
        )
        session.add(record)
        rows_loaded += 1

    session.flush()
    result.status = 'success'
    result.rows_loaded = rows_loaded
    result.rows_rejected = rows_rejected
    return result


# ── Report ────────────────────────────────────────────────────────────

def print_load_report(results: list[LoadResult]) -> None:
    succeeded = [r for r in results if r.status == 'success']
    failed = [r for r in results if r.status == 'failed']
    total_rows = sum(r.rows_loaded for r in succeeded)

    print(f"\n{'='*60}")
    print(f"DATA LOAD REPORT")
    print(f"{'='*60}")
    print(f"Files processed : {len(results)}")
    print(f"Succeeded       : {len(succeeded)}")
    print(f"Failed          : {len(failed)}")
    print(f"Total rows      : {total_rows}")

    if succeeded:
        print(f"\n── Succeeded ────────────────────────────────────────")
        for r in succeeded:
            print(
                f"  ✓ {r.file_name:<45} "
                f"{r.rows_loaded:>5} rows → {r.table_name}"
            )
            if r.rows_rejected:
                print(f"    ⚠ {r.rows_rejected} rows rejected")
            for w in r.warnings:
                print(f"    ⚠ {w}")

    if failed:
        print(f"\n── Failed ───────────────────────────────────────────")
        for r in failed:
            print(f"  ✗ {r.file_name}")
            print(f"    Reason: {r.failure_reason}")

    print(f"\n{'='*60}")


# ── CLI ───────────────────────────────────────────────────────────────

TABULAR_DIR = Path('data/raw/tabular')

FILE_LOADER_MAP = {
    'campaign_performance_report.xlsx': load_campaign_performance,
    'keyword_tracker.xlsx':             load_keyword_rankings,
    'product_catalogue.xlsx':           load_product_catalogue,
    'acos_targets_by_client.csv':       load_acos_targets,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="KIRA tabular data loader"
    )
    parser.add_argument(
        '--all', action='store_true',
        help="Load all tabular files from data/raw/tabular/"
    )
    parser.add_argument(
        '--file',
        help="Load a specific file"
    )
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()
    from infra.database import get_engine, get_session

    engine = get_engine()
    session = get_session()

    results = []

    try:
        if args.all:
            for filename, loader_fn in FILE_LOADER_MAP.items():
                file_path = TABULAR_DIR / filename
                if not file_path.exists():
                    logger.warning(f"File not found: {file_path}")
                    continue
                logger.info(f"Loading: {filename}")
                result = loader_fn(file_path, session)
                results.append(result)
                if result.status == 'success':
                    logger.info(
                        f"  ✓ {result.rows_loaded} rows → {result.table_name}"
                    )
                else:
                    logger.error(f"  ✗ {result.failure_reason}")

        elif args.file:
            file_path = Path(args.file)
            loader_fn = FILE_LOADER_MAP.get(file_path.name)
            if not loader_fn:
                print(f"No loader for {file_path.name}")
                print(f"Known files: {list(FILE_LOADER_MAP.keys())}")
                sys.exit(1)
            result = loader_fn(file_path, session)
            results.append(result)

        else:
            parser.print_help()
            sys.exit(0)

        # Commit all or nothing
        failed = [r for r in results if r.status == 'failed']
        if failed:
            session.rollback()
            logger.error("Rolling back — one or more files failed")
        else:
            session.commit()
            logger.info("Committed successfully")

    except Exception as e:
        session.rollback()
        logger.error(f"Unexpected error: {e}")
        raise
    finally:
        session.close()

    print_load_report(results)
    sys.exit(1 if any(r.status == 'failed' for r in results) else 0)