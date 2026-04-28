"""
infra/s3.py

S3 helper for saving and retrieving KIRA reports.
Uses explicit regional endpoint — required on some networks
where boto3's default virtual-hosted endpoint fails DNS resolution.
"""

import json
import boto3
from pathlib import Path
from datetime import datetime
import sys

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import get_settings

settings = get_settings()


def get_s3_client():
    """Returns a boto3 S3 client."""
    return boto3.client(
        "s3",
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        region_name=settings.s3_region,
    )


def upload_report(report_data: dict) -> str:
    """
    Uploads a pipeline report to S3.
    Returns the S3 key the report was saved under.

    Key format: reports/{client_id}/{run_id}.json
    """
    if not settings.has_s3:
        return ""

    client = get_s3_client()
    key = f"{settings.s3_reports_prefix}/{report_data['client_id']}/{report_data['run_id']}.json"

    client.put_object(
        Bucket=settings.s3_bucket_name,
        Key=key,
        Body=json.dumps(report_data, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

    return key


def download_report(client_id: str, run_id: str) -> dict | None:
    """
    Downloads a report from S3 by client_id and run_id.
    Returns None if not found.
    """
    if not settings.has_s3:
        return None

    client = get_s3_client()
    key = f"{settings.s3_reports_prefix}/{client_id}/{run_id}.json"

    try:
        response = client.get_object(Bucket=settings.s3_bucket_name, Key=key)
        return json.loads(response["Body"].read().decode("utf-8"))
    except client.exceptions.NoSuchKey:
        return None


def list_reports(client_id: str | None = None) -> list[dict]:
    """
    Lists all reports in S3, optionally filtered by client_id.
    Returns list of {client_id, run_id, key, last_modified}.
    """
    if not settings.has_s3:
        return []

    s3 = get_s3_client()
    prefix = settings.s3_reports_prefix
    if client_id:
        prefix = f"{prefix}/{client_id}/"

    response = s3.list_objects_v2(
        Bucket=settings.s3_bucket_name,
        Prefix=prefix,
    )

    reports = []
    for obj in response.get("Contents", []):
        parts = obj["Key"].replace(settings.s3_reports_prefix + "/", "").split("/")
        if len(parts) == 2:
            reports.append({
                "client_id": parts[0],
                "run_id": parts[1].replace(".json", ""),
                "key": obj["Key"],
                "last_modified": obj["LastModified"].isoformat(),
                "size_bytes": obj["Size"],
            })

    return sorted(reports, key=lambda x: x["last_modified"], reverse=True)


if __name__ == "__main__":
    print(f"S3 enabled: {settings.has_s3}")
    print(f"Bucket    : {settings.s3_bucket_name}")
    print(f"Endpoint  : {settings.s3_endpoint_url}")

    # Upload a test report
    test_report = {
        "run_id": "test_20260325",
        "client_id": "natura",
        "query": "S3 connection test",
        "final_report": "## Test\nS3 upload confirmed.",
        "anomalies": [],
        "compliance_passed": True,
    }

    key = upload_report(test_report)
    print(f"\nUploaded  : {key}")

    # List reports
    reports = list_reports()
    print(f"Reports in S3: {len(reports)}")
    for r in reports:
        print(f"  {r['client_id']} / {r['run_id']} — {r['last_modified'][:10]}")

    # Download it back
    downloaded = download_report("natura", "test_20260325")
    print(f"\nDownloaded: {downloaded['run_id']} — {downloaded['final_report'][:30]}...")