"""
scripts/pull_data_from_s3.py

Downloads synthetic data from S3 to local data/ directory.
Runs at container startup on Render — ensures data is available
without baking it into the Docker image.

Also checks if Pinecone needs ingestion and runs the full
ingestion pipeline if the index is empty.

Usage:
    python scripts/pull_data_from_s3.py
"""

import os
from dotenv import load_dotenv

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv()

from infra.s3 import get_s3_client
from config.settings import get_settings

settings = get_settings()


def pull_data() -> None:
    """Downloads all files from S3 kira-data/ prefix to local data/."""
    if not settings.has_s3:
        print("S3 not configured — skipping data pull.")
        return

    client = get_s3_client()
    prefix = "kira-data/"

    response = client.list_objects_v2(
        Bucket=settings.s3_bucket_name,
        Prefix=prefix,
    )

    objects = response.get("Contents", [])
    if not objects:
        print("No data found in S3. Run generate_data.py locally and upload first.")
        return

    print(f"Pulling {len(objects)} files from S3...")

    for obj in objects:
        s3_key = obj["Key"]
        local_path = Path(s3_key.replace(prefix, ""))
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if local_path.exists():
            print(f"  Exists   : {local_path}")
            continue

        body = client.get_object(
            Bucket=settings.s3_bucket_name,
            Key=s3_key,
        )["Body"].read()

        local_path.write_bytes(body)
        print(f"  Downloaded: {local_path}")

    print("Data pull complete.\n")


def needs_ingestion() -> bool:
    """
    Returns True if Pinecone index is empty and needs ingestion.
    Checks total vector count across all namespaces.
    Only ingests once — skips on subsequent container restarts.
    """
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=settings.pinecone_api_key)
        index = pc.Index(settings.pinecone_index_name)
        stats = index.describe_index_stats()
        total = sum(
            ns.vector_count
            for ns in stats.namespaces.values()
        )
        print(f"Pinecone vector count: {total}")
        return total == 0
    except Exception as e:
        print(f"Could not check Pinecone: {e}")
        return True


def run_ingestion() -> None:
    """Runs the full ingestion pipeline — chunk, embed, ingest to Pinecone."""
    print("Pinecone is empty — running ingestion pipeline...")
    os.system("python ingestion/chunker.py")
    os.system("python ingestion/embedder.py")
    os.system("python scripts/ingest_docs.py --backend pinecone --skip-spot-check")
    print("Ingestion complete.\n")


if __name__ == "__main__":
    pull_data()

    if settings.has_pinecone and needs_ingestion():
        run_ingestion()
    else:
        print("Pinecone already populated — skipping ingestion.")