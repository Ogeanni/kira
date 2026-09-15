"""
infra/database.py

PostgreSQL layer for KIRA.
Replaces data/metrics/daily_metrics.csv with a proper database.

Tables:
  clients          — agency clients (natura, vitalblend, peakgear, lumina)
  daily_metrics    — time-series performance data per client per day
  anomalies        — detected anomalies with metadata
  pipeline_runs    — audit trail of every pipeline execution
  feedback         — thumbs up/down on generated reports

Why PostgreSQL over CSV:
  - SQL queries are more expressive than pandas for complex aggregations
  - Proper indexing — date range queries are instant at scale
  - Concurrent writes — multiple pipeline runs don't corrupt data
  - Audit trail — know exactly when data was inserted and by whom
  - Production standard — matches the SQL Server warehouse in the job description

Run directly to create tables and load synthetic data:
    python infra/database.py
"""

from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Integer, String, Text, create_engine, text
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import NullPool

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import get_settings

settings = get_settings()


# ── SQLAlchemy setup ──────────────────────────────────────────────────

def get_engine():
    """
    Creates SQLAlchemy engine.
    NullPool disables connection pooling — safe for scripts and
    serverless environments. Use pool_size=5 for long-running servers.
    """
    return create_engine(
        settings.database_url,
        poolclass=NullPool,
        echo=False,   # set True to log all SQL queries
    )


def get_session() -> Session:
    """Returns a new database session."""
    engine = get_engine()
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


# ── Models ────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class Client(Base):
    """Agency clients — one row per client."""
    __tablename__ = "clients"

    id = Column(String, primary_key=True)          # e.g. "natura"
    name = Column(String, nullable=False)           # e.g. "Natura Skincare"
    hero_asin = Column(String)                      # primary product ASIN
    acos_target = Column(Float)                     # target ACOS %
    break_even_acos = Column(Float)                 # break-even ACOS %
    created_at = Column(DateTime, default=datetime.utcnow)


class DailyMetric(Base):
    """
    Daily performance metrics per client.
    Replaces data/metrics/daily_metrics.csv.
    Indexed on (client_id, date) for fast range queries.
    """
    __tablename__ = "daily_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String, nullable=False, index=True)
    client_id = Column(String, ForeignKey("clients.id"), nullable=False, index=True)
    asin = Column(String)
    revenue_usd = Column(Float)
    ad_revenue_usd = Column(Float)
    organic_revenue_usd = Column(Float)
    units_sold = Column(Integer)
    sessions = Column(Integer)
    conversion_rate = Column(Float)
    acos = Column(Float)
    tacos = Column(Float)
    buy_box_pct = Column(Float)
    asp_usd = Column(Float)
    is_anomaly = Column(Boolean, default=False)
    anomaly_type = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class AnomalyRecord(Base):
    """
    Detected anomalies with full context.
    Written by the Analysis Agent after each pipeline run.
    """
    __tablename__ = "anomalies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, nullable=False, index=True)
    client_id = Column(String, ForeignKey("clients.id"), nullable=False, index=True)
    metric = Column(String, nullable=False)
    current_value = Column(Float)
    baseline_value = Column(Float)
    deviation_pct = Column(Float)
    description = Column(Text)
    detected_at = Column(DateTime, default=datetime.utcnow)


class PipelineRun(Base):
    """
    Audit trail of every pipeline execution.
    Records which agents ran, cost, latency, and outcome.
    """
    __tablename__ = "pipeline_runs"

    id = Column(String, primary_key=True)          # run_id
    client_id = Column(String, ForeignKey("clients.id"), nullable=False, index=True)
    query = Column(Text)
    completed_agents = Column(String)              # comma-separated
    anomalies_detected = Column(Integer, default=0)
    compliance_passed = Column(Boolean, nullable=True)
    compliance_confidence = Column(Float, nullable=True)
    llm_calls = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)
    latency_seconds = Column(Float, default=0.0)
    has_final_report = Column(Boolean, default=False)
    errors = Column(Text, nullable=True)           # JSON string
    s3_key = Column(String, nullable=True)
    started_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)


class Feedback(Base):
    """
    Thumbs up/down on generated reports.
    In production, negative feedback triggers retrieval reranking.
    """
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, ForeignKey("pipeline_runs.id"), nullable=False)
    rating = Column(String, nullable=False)        # "up" or "down"
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class CampaignPerformance(Base):
    """
    Weekly campaign performance metrics per client.
    Loaded from campaign_performance_report.xlsx.
    Queried by: query_metrics MCP tool, analysis agent.
    """
    __tablename__ = "campaign_performance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    week = Column(String(20), nullable=False)
    client_id = Column(String(50), nullable=False)
    campaign_name = Column(String(200))
    campaign_type = Column(String(50))
    spend_usd = Column(Float, nullable=False)
    revenue_usd = Column(Float, nullable=False)
    acos = Column(Float)
    impressions = Column(Integer)
    clicks = Column(Integer)
    ctr = Column(Float)
    cpc_usd = Column(Float)
    units_sold = Column(Integer)
    roas = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class KeywordRanking(Base):
    """
    Keyword ranking and search volume tracker per client.
    Loaded from keyword_tracker.xlsx.
    Queried by: analysis agent for SEO performance trends.
    """
    __tablename__ = "keyword_rankings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    client_id = Column(String(50), nullable=False)
    asin = Column(String(20), nullable=False)
    keyword = Column(String(200), nullable=False)
    match_type = Column(String(20))
    monthly_search_volume = Column(Integer)
    current_rank = Column(Integer)
    previous_rank = Column(Integer)
    rank_change = Column(Integer)
    bid_usd = Column(Float)
    relevance_score = Column(Float)
    priority = Column(String(20))
    notes = Column(Text)
    recorded_at = Column(DateTime, default=datetime.utcnow)


class ProductCatalogue(Base):
    """
    Product master data per client.
    Loaded from product_catalogue.xlsx.
    Queried by: analysis agent for Buy Box status, ASP, review counts.
    Notes field contains free-text operational commentary.
    """
    __tablename__ = "product_catalogue"

    id = Column(Integer, primary_key=True, autoincrement=True)
    client_id = Column(String(50), nullable=False)
    asin = Column(String(20), nullable=False)
    product_title = Column(String(500))
    category = Column(String(100))
    sub_category = Column(String(100))
    asp_usd = Column(Float)
    units_fba = Column(Integer)
    buy_box_status = Column(String(20))
    buy_box_pct = Column(Float)
    star_rating = Column(Float)
    review_count = Column(Integer)
    acos_target = Column(Float)
    break_even_acos = Column(Float)
    status = Column(String(20))
    notes = Column(Text)
    updated_at = Column(DateTime, default=datetime.utcnow)


class AcosTarget(Base):
    """
    Per-client ACOS targets by account stage.
    Loaded from acos_targets_by_client.csv.
    More granular than the single acos_target in the clients table —
    supports seasonal and stage-based target variations.
    """
    __tablename__ = "acos_targets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    client_id = Column(String(50), nullable=False)
    client_name = Column(String(100))
    account_stage = Column(String(50), nullable=False)
    days_range = Column(String(20))
    acos_target_min = Column(Float)
    acos_target_max = Column(Float)
    break_even_acos = Column(Float)
    primary_objective = Column(String(50))
    notes = Column(Text)

# ── Schema management ─────────────────────────────────────────────────

def create_tables() -> None:
    """Creates all tables if they don't exist."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    tables = [
        "clients", "daily_metrics", "anomalies",
        "pipeline_runs", "feedback",
        "campaign_performance", "keyword_rankings",
        "product_catalogue", "acos_targets",
    ]
    print(f"Tables created: {', '.join(tables)}")


def drop_tables() -> None:
    """Drops all tables — use only in development."""
    engine = get_engine()
    Base.metadata.drop_all(engine)
    print("All tables dropped.")


# ── Data loading ──────────────────────────────────────────────────────

def load_clients(session: Session) -> None:
    """Inserts client records if they don't exist."""
    clients = [
        Client(
            id="natura",
            name="Natura Skincare",
            hero_asin="B08XK9L2MN",
            acos_target=0.28,
            break_even_acos=0.28,
        ),
        Client(
            id="vitalblend",
            name="VitalBlend Supplements",
            hero_asin="B09KL3MN2P",
            acos_target=0.24,
            break_even_acos=0.28,
        ),
        Client(
            id="peakgear",
            name="PeakGear Outdoors",
            hero_asin="B07WQ2KP4M",
            acos_target=0.25,
            break_even_acos=0.30,
        ),
        Client(
            id="lumina",
            name="Lumina Home Lighting",
            hero_asin="B0CK2L9MNP",
            acos_target=0.30,
            break_even_acos=0.38,
        ),
    ]
    for client in clients:
        existing = session.get(Client, client.id)
        if not existing:
            session.add(client)
    session.commit()
    print(f"Clients loaded: {len(clients)}")


def load_metrics_from_csv(session: Session) -> None:
    """
    Loads daily_metrics.csv into the daily_metrics table.
    Idempotent — skips rows that already exist.
    """
    import csv
    from pathlib import Path

    csv_path = settings.metrics_dir / "daily_metrics.csv"
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        print("Run: python scripts/generate_data.py first")
        return

    # Check if data already loaded
    result = session.execute(text("SELECT COUNT(*) FROM daily_metrics"))
    count = result.scalar()
    if count > 0:
        print(f"Metrics already loaded: {count} rows exist")
        return

    rows_loaded = 0
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = DailyMetric(
                date=row["date"],
                client_id=row["client_id"],
                asin=row["asin"],
                revenue_usd=float(row["revenue_usd"]),
                ad_revenue_usd=float(row["ad_revenue_usd"]),
                organic_revenue_usd=float(row["organic_revenue_usd"]),
                units_sold=int(row["units_sold"]),
                sessions=int(row["sessions"]),
                conversion_rate=float(row["conversion_rate"]),
                acos=float(row["acos"]),
                tacos=float(row["tacos"]),
                buy_box_pct=float(row["buy_box_pct"]),
                asp_usd=float(row["asp_usd"]),
                is_anomaly=row["is_anomaly"].lower() == "true",
                anomaly_type=row["anomaly_type"] if row["anomaly_type"] else None,
            )
            session.add(metric)
            rows_loaded += 1

    session.commit()
    print(f"Metrics loaded: {rows_loaded} rows")


# ── Query helpers ─────────────────────────────────────────────────────

def get_client_metrics(
    session: Session,
    client_id: str,
    days: int = 90,
    metric: str = "all",
) -> list[dict]:
    """
    Returns recent metrics for a client as a list of dicts.
    Equivalent to what query_metrics() returns from CSV.
    This is the SQL version — more expressive and scalable.
    """
    base_cols = [
        "date", "client_id", "asin", "revenue_usd", "ad_revenue_usd",
        "organic_revenue_usd", "units_sold", "sessions", "conversion_rate",
        "acos", "tacos", "buy_box_pct", "asp_usd", "is_anomaly", "anomaly_type",
    ]

    query = text("""
        SELECT {cols}
        FROM daily_metrics
        WHERE client_id = :client_id
        ORDER BY date DESC
        LIMIT :days
    """.format(cols=", ".join(base_cols)))

    result = session.execute(query, {"client_id": client_id, "days": days})
    rows = [dict(zip(base_cols, row)) for row in result.fetchall()]

    # Return in ascending date order for time-series analysis
    return sorted(rows, key=lambda x: x["date"])


def get_anomaly_history(
    session: Session,
    client_id: str,
    limit: int = 10,
) -> list[dict]:
    """Returns recent detected anomalies for a client."""
    query = text("""
        SELECT metric, description, deviation_pct, detected_at
        FROM anomalies
        WHERE client_id = :client_id
        ORDER BY detected_at DESC
        LIMIT :limit
    """)
    result = session.execute(query, {"client_id": client_id, "limit": limit})
    cols = ["metric", "description", "deviation_pct", "detected_at"]
    return [dict(zip(cols, row)) for row in result.fetchall()]


def save_pipeline_run(session: Session, run_data: dict) -> None:
    """Saves a pipeline run record to the database."""
    import json
    run = PipelineRun(
        id=run_data["run_id"],
        client_id=run_data["client_id"],
        query=run_data.get("query", ""),
        completed_agents=",".join(run_data.get("completed_agents", [])),
        anomalies_detected=run_data.get("anomalies_detected", 0),
        compliance_passed=run_data.get("compliance_passed"),
        compliance_confidence=run_data.get("compliance_confidence"),
        llm_calls=run_data.get("llm_calls", 0),
        total_tokens=run_data.get("total_tokens", 0),
        cost_usd=run_data.get("cost_usd", 0.0),
        latency_seconds=run_data.get("latency_seconds", 0.0),
        has_final_report=bool(run_data.get("final_report")),
        errors=json.dumps(run_data.get("errors", [])),
        s3_key=run_data.get("s3_key"),
        started_at=datetime.utcnow(),
    )
    session.merge(run)   # upsert — safe to call multiple times
    session.commit()


if __name__ == "__main__":
    print("Setting up KIRA database...\n")

    create_tables()

    session = get_session()

    load_clients(session)
    load_metrics_from_csv(session)

    # Verify
    result = session.execute(text("SELECT COUNT(*) FROM daily_metrics"))
    count = result.scalar()
    print(f"\nVerification:")
    print(f"  daily_metrics rows : {count}")

    result = session.execute(text("SELECT id, name FROM clients"))
    print(f"  clients:")
    for row in result.fetchall():
        print(f"    {row[0]} — {row[1]}")

    # Test a SQL query — equivalent to what Analysis Agent will run
    print(f"\nSample SQL query — Lumina last 7 days:")
    rows = get_client_metrics(session, "lumina", days=7)
    for row in rows:
        print(f"  {row['date']} | revenue=${row['revenue_usd']:.2f} | acos={row['acos']:.3f}")

    session.close()
    print("\nDatabase setup complete.")