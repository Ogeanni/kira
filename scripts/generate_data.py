"""
scripts/generate_data.py

Generates all synthetic data KIRA needs across every layer.
Run this once before building any layer beyond Layer 1.

Output:
  data/raw/          — 10 knowledge documents (SOPs, transcripts, compliance, frameworks)
  data/raw/metadata_index.json     — document metadata for ingestion pipeline
  data/raw/anomaly_ground_truth.json — known anomalies for eval
  data/metrics/daily_metrics.csv   — 90 days x 4 clients with injected anomalies
  data/compliance/train.json       — LoRA training examples (80%)
  data/compliance/test.json        — LoRA eval examples (20%)
"""

import csv
import json
import random
from datetime import datetime, timedelta
from pathlib import Path


# ── 1. KNOWLEDGE DOCUMENTS ───────────────────────────────────────────

KNOWLEDGE_DOCUMENTS = [
    {
        "filename": "sop_listing_optimisation.txt",
        "doc_type": "sop",
        "client": "global",
        "content": """
STANDARD OPERATING PROCEDURE — Listing Optimisation

Version: 2.1 | Owner: Strategy Team | Last updated: 2024-01

PURPOSE
This SOP defines the process for optimising Amazon product listings
across all client accounts.

WHEN TO APPLY
Apply this SOP when: (1) onboarding a new client, (2) ACOS exceeds
25% for 7 consecutive days, (3) organic rank drops below position 15
for primary keywords, (4) client requests a listing audit.

PROCESS

Step 1 — Keyword Research
Use Helium 10 Cerebro to extract competitor ASINs ranking for target
keywords. Pull search volume data for the past 90 days. Prioritise
keywords with search volume above 500 per month and relevance score
above 0.7. Document all keywords in the client keyword tracker.

Step 2 — Title Construction
Title must lead with primary keyword. Follow the format:
Brand, Primary Keyword, Key Feature, Size or Variant, Secondary Keyword.
Maximum 200 characters. Never use ALL CAPS. Never include promotional
language such as sale, discount, or best. Never mention competitor brands.

Step 3 — Bullet Points
Write exactly 5 bullet points. Each must begin with a capitalised
feature name followed by an em dash. Lead with the benefit, not the
feature. Example: FASTER CHARGING — Charges your device 3x faster
than standard cables, from 0 to 80% in under 30 minutes.

Step 4 — Backend Keywords
Add all non-duplicate keywords not used in title or bullets.
Include common misspellings. Never repeat words already in visible
copy — Amazon counts the full listing.

COMPLIANCE RULES
Never use the following in any listing copy: best, #1, greatest,
guaranteed, cure, treat, prevent. These trigger Amazon policy flags.
Any listing containing these terms must be revised before publishing.

APPROVAL
All listing changes require review by the account manager before
submission. Turnaround: 48 hours standard, 24 hours urgent.
""",
    },
    {
        "filename": "sop_acos_management.txt",
        "doc_type": "sop",
        "client": "global",
        "content": """
STANDARD OPERATING PROCEDURE — ACOS Management

Version: 1.4 | Owner: PPC Team | Last updated: 2024-02

PURPOSE
Define thresholds and response actions for Advertising Cost of Sale
(ACOS) management across all client accounts.

ACOS THRESHOLDS BY ACCOUNT STAGE

Launch phase (0 to 90 days): Target ACOS 40 to 60%.
Prioritise ranking over profitability.

Growth phase (90 to 365 days): Target ACOS 25 to 35%.
Balance ranking and margin.

Mature phase (365 days and above): Target ACOS 15 to 25%.
Profitability is primary objective.

RESPONSE ACTIONS BY DEVIATION

If ACOS exceeds target by more than 10% for 3 consecutive days:
- Reduce bids on keywords with ACOS above 60% by 15%
- Pause keywords with zero sales and more than 10 clicks
- Add converting search terms as exact match keywords
- Document changes in the campaign log with timestamp and rationale

If ACOS is below target by more than 15% for 7 consecutive days:
- Increase bids on top-converting keywords by 10%
- Expand to broad match on proven exact match keywords
- Raise daily budget by 20% if impression share is below 60%

NEVER do the following without account manager approval:
- Pause an entire campaign
- Change campaign structure across the board
- Increase total budget by more than 50% in a single change

REPORTING
Log every bid change in the PPC change log. Include: date, keyword,
old bid, new bid, rationale. Weekly ACOS report due every Monday 9am.
""",
    },
    {
        "filename": "sop_buy_box_recovery.txt",
        "doc_type": "sop",
        "client": "global",
        "content": """
STANDARD OPERATING PROCEDURE — Buy Box Loss and Recovery

Version: 2.0 | Owner: Operations Team | Last updated: 2024-03

PURPOSE
Define the response process when a client loses the Buy Box on any
active ASIN.

DETECTION
Buy Box status is checked every 4 hours via automated monitoring.
Loss detected when Buy Box percentage drops below 80% for any ASIN
with more than 50 daily sessions.

IMMEDIATE RESPONSE — within 2 hours of detection

Step 1 — Identify root cause. Common causes ranked by frequency:
  a) Price undercut by competitor or third-party seller
  b) Inventory below Amazon reorder threshold (under 30 units FBA)
  c) Seller performance metric drop
  d) Listing suppression due to policy violation
  e) Account health issue

Step 2 — Match response to cause:
  a) Price undercut — reprice to within 2% of lowest FBA offer.
     Do not go below client minimum price floor.
  b) Inventory — raise reorder alert immediately, notify client.
  c) Performance — escalate to account manager immediately.
  d) Suppression — fix violation within 4 hours.
  e) Account health — escalate to senior account manager same day.

ESCALATION
Any Buy Box loss lasting more than 24 hours requires client
notification. Any loss on a hero SKU requires immediate account
manager involvement regardless of duration.
""",
    },
    {
        "filename": "compliance_restricted_claims.txt",
        "doc_type": "compliance",
        "client": "global",
        "content": """
COMPLIANCE DOCUMENT — Restricted Claims and Language

Version: 3.2 | Owner: Compliance Team | Last updated: 2024-03

ABSOLUTE PROHIBITIONS
These terms trigger Amazon policy violations and must never appear
in any listing copy, A+ content, or sponsored brand creative.

Health and medical claims:
  cure, treat, prevent, diagnose, heal, therapeutic, clinically proven
  (unless accompanied by FDA-cleared study citation on file)

Superlatives without substantiation:
  best, #1, greatest, finest, top-rated, unrivalled, unmatched

Competitor references:
  Any competitor brand name, any phrase better than Brand,
  any direct comparison to a named competitor product.

Guarantee language:
  guaranteed, money-back guarantee, lifetime guarantee

Pricing in listing copy:
  sale, discount, percent off, limited time, act now, free
  (unless accurately describing a bundled item)

CONDITIONAL TERMS — require compliance review before use
  natural — only with relevant third-party certification
  organic — only with USDA Organic or equivalent certification
  safe — must specify: safe for what, for whom, under what conditions
  hypoallergenic — requires dermatologist testing documentation
  clinically tested — requires study citation and approval on file

COMPLIANCE REVIEW PROCESS
Submit draft copy to the compliance team with: the specific term,
the certification supporting its use, and the target ASIN.
SLA: 24 hours standard, 4 hours urgent.
""",
    },
    {
        "filename": "compliance_image_standards.txt",
        "doc_type": "compliance",
        "client": "global",
        "content": """
COMPLIANCE DOCUMENT — Amazon Image Standards

Version: 2.1 | Owner: Compliance Team | Last updated: 2024-02

MAIN IMAGE REQUIREMENTS
Background: pure white, RGB 255 255 255. No off-white, no gradients.
Product must occupy at least 85% of the image frame.
No props or accessories unless sold as a set.
No text overlays, watermarks, logos, or badges.
No borders or frames.
Minimum 1000 x 1000 pixels. 2000 x 2000 recommended.

SECONDARY IMAGE SLOTS
Lifestyle images permitted — product in use or in context.
Infographic images permitted with text overlays on features.
No explicit before and after images for health or beauty products.
No images that make medical or clinical claims visually.

COMMON SUPPRESSION TRIGGERS
Off-white background on main image — the most common cause.
Text on main image.
Competitor product visible in frame.
Watermark at any zoom level.
Image below 1000px on any side.

RESOLUTION PROCESS
Suppression notice appears in Seller Central under Manage Inventory.
Replace non-compliant image within 4 hours.
Suppression typically clears within 12 to 24 hours of correct upload.
""",
    },
    {
        "filename": "framework_weekly_reporting.txt",
        "doc_type": "framework",
        "client": "global",
        "content": """
ANALYSIS FRAMEWORK — Weekly Client Performance Report

Version: 2.0 | Owner: Analytics Team | Last updated: 2024-01

REPORT STRUCTURE

Section 1 — Executive Summary (3 to 5 sentences)
Lead with the single most important metric movement this week.
State direction, magnitude, and one sentence of causal explanation.
Example: Total revenue grew 18% week on week driven by a 22% increase
in organic sales following the listing optimisation on Day 3.

Section 2 — Key Metrics Table
Always include: Total Revenue, Ad Revenue, Organic Revenue, ACOS,
TACOS, Units Sold, Sessions, Conversion Rate, Buy Box %, Average ASP.
Show current week, prior week, and percent change.
Flag any metric that moved more than 15% in either direction.

Section 3 — Campaign Performance
Top 5 campaigns by spend. For each: spend, revenue, ACOS, week on
week ACOS change. Highlight any campaign where ACOS improved more
than 5 points or deteriorated more than 5 points.

Section 4 — Anomaly Commentary
Any metric deviating more than 2 standard deviations from the 4-week
rolling average must be explained. Possible causes: seasonality,
competitor activity, listing change, buy box loss, inventory issue.

Section 5 — Next Week Actions
Maximum 3 action items. Each must have: owner, deadline, expected
impact. No actions without a clear owner.

TONE
Professional but direct. No filler phrases.
Quantify everything. If you cannot put a number on it, leave it out.
""",
    },
    {
        "filename": "framework_anomaly_investigation.txt",
        "doc_type": "framework",
        "client": "global",
        "content": """
ANALYSIS FRAMEWORK — Performance Anomaly Investigation

Version: 1.0 | Owner: Analytics Team | Last updated: 2024-01

ANOMALY DEFINITION
A metric is anomalous when it deviates more than 2 standard deviations
from its 4-week rolling average.

INVESTIGATION SEQUENCE

Step 1 — Confirm the anomaly is real
Amazon data can lag 24 to 48 hours. A sudden drop on Monday often
reflects incomplete Sunday data. If data is incomplete, revisit Tuesday.

Step 2 — Classify the anomaly type
  Spike (sudden increase): ad spend increase, viral event, price drop.
  Drop (sudden decrease): Buy Box loss, suppression, stockout.
  Gradual drift: keyword rank decline, bid erosion, seasonal effect.

Step 3 — Check correlated metrics
Never explain an anomaly from a single metric.
  Revenue drops — check sessions (traffic) vs conversion (listing).
  ACOS spikes — check if spend increased or revenue dropped or both.
  Sessions drop — check organic rank, sponsored impressions, Buy Box.

Step 4 — Timeline matching
Find when the anomaly started. Match to known events in the change log,
campaign logs, Amazon announcements, or competitor price history.

Step 5 — Draft the explanation
Structure: [Metric] [direction] [magnitude] starting [date].
Root cause: [cause]. Evidence: [correlated metric]. Action: [response].
""",
    },
    {
        "filename": "transcript_onboarding_natura.txt",
        "doc_type": "transcript",
        "client": "natura",
        "content": """
MEETING TRANSCRIPT — Client Onboarding Call
Client: Natura Skincare | Date: 2024-03-15 | Duration: 47 minutes
Attendees: Omotolani (Account Manager), Ann (Client CEO), Abiodun (Client Marketing)

[00:03] Omotolani: How do you want customers to feel when they encounter
your products on Amazon?

[00:04] Ann: Premium but accessible. Our core customer is a woman
aged 28 to 45 who cares about ingredients and sustainability but
does not want to spend Tatcha money.

[00:05] Abiodun: The word natural is very important to us but we use
it carefully. All our products are EWG verified. If you write natural
it needs to be followed by our certification, not used as a vague term.

[00:08] Omotolani: What terms should we never use for Natura?

[00:09] Ann: Never anti-aging — we say supports skin renewal instead.
Also never chemical-free because everything is a chemical and it
undermines our credibility. Never compare us to any competitor by name.

[00:12] Abiodun: We cannot make treatment claims. Say formulated for
sensitive skin but not treats sensitive skin.

[00:18] Omotolani: What are your primary goals for the next 90 days?

[00:19] Ann: Three things. First, get our Vitamin C serum ASIN
B08XK9L2MN into the top 10 for vitamin c serum for face — we are
at position 23. Second, reduce ACOS from 38% to under 28%. Third,
fix our main image on the eye cream — it is not white background
compliant and we keep getting suppressed.
""",
    },
    {
        "filename": "transcript_onboarding_vitalblend.txt",
        "doc_type": "transcript",
        "client": "vitalblend",
        "content": """
MEETING TRANSCRIPT — Client Onboarding Call
Client: VitalBlend Supplements | Date: 2024-02-08 | Duration: 52 minutes
Attendees: Oge (Account Manager), Diana (Client CEO), Junaid (Client Ops)

[00:04] Diana: Three SKUs account for about 70% of revenue. The collagen
peptides powder ASIN B09KL3MN2P is our hero. Then magnesium glycinate
and vitamin D3 with K2. We cannot afford to lose Buy Box on any of them.

[00:07] Junaid: The collagen has had trouble. We were at 94% Buy Box six
months ago, now at 71%. A third-party seller appeared and has been
undercutting us by about 3%.

[00:11] Diana: Our break-even ACOS is 28%. We are currently at 34%
so we are losing money on ads. The goal within 90 days is 24% or below.

[00:15] Diana: Never use treat, cure, prevent, or diagnose. Beyond the
Amazon rules: never say pharmaceutical grade — we cannot substantiate it.
Never say doctor recommended without a documented quote. Never use
detox or cleanse — we want to stay away from that positioning entirely.

[00:19] Junaid: Our collagen is bovine sourced. Never hide that. Always
disclose bovine collagen clearly in the bullets.
""",
    },
    {
        "filename": "transcript_onboarding_peakgear.txt",
        "doc_type": "transcript",
        "client": "peakgear",
        "content": """
MEETING TRANSCRIPT — Client Onboarding Call
Client: PeakGear Outdoors | Date: 2024-01-22 | Duration: 38 minutes
Attendees: Sarah (Account Manager), Maryann (Client Founder)

[00:03] Maryann: We are functional outdoor gear, mid-price point. We compete
with brands like Teton Sports but do not ever mention them by name.

[00:07] Maryann: The 20-degree sleeping bag ASIN B07WQ2KP4M is 40% of our
revenue. It has a 4.6 star rating on 2300 reviews. I want it in the
top 3 for 20 degree sleeping bag — we are at position 8.

[00:11] Maryann: We are seasonal. March through August ACOS up to 35% is
acceptable. September through February we want under 20%. Do not apply
a flat ACOS target — you need to know the time of year.

[00:15] Maryann: Never say waterproof unless the product is certified.
Our bags are water-resistant. Never claim made in USA — they are
manufactured in Vietnam. Never say military-grade. It attracts the
wrong customer who then returns the product.

[00:20] Maryann: Our brand name is PeakGear — capital P capital G, two
words. Never peakgear as one lowercase word.
""",
    },
]


# ── 2. STRUCTURED METRICS ─────────────────────────────────────────────

CLIENTS = [
    {
        "client_id": "natura",
        "client_name": "Natura Skincare",
        "hero_asin": "B08XK9L2MN",
        "baseline": {
            "revenue": 18000,
            "ad_revenue": 7200,
            "units": 420,
            "sessions": 5800,
            "conversion_rate": 0.072,
            "acos": 0.38,
            "buy_box_pct": 0.91,
        },
    },
    {
        "client_id": "vitalblend",
        "client_name": "VitalBlend Supplements",
        "hero_asin": "B09KL3MN2P",
        "baseline": {
            "revenue": 31000,
            "ad_revenue": 10500,
            "units": 890,
            "sessions": 9200,
            "conversion_rate": 0.097,
            "acos": 0.34,
            "buy_box_pct": 0.71,
        },
    },
    {
        "client_id": "peakgear",
        "client_name": "PeakGear Outdoors",
        "hero_asin": "B07WQ2KP4M",
        "baseline": {
            "revenue": 24000,
            "ad_revenue": 8400,
            "units": 300,
            "sessions": 4100,
            "conversion_rate": 0.073,
            "acos": 0.31,
            "buy_box_pct": 0.97,
        },
    },
    {
        "client_id": "lumina",
        "client_name": "Lumina Home Lighting",
        "hero_asin": "B0CK2L9MNP",
        "baseline": {
            "revenue": 14000,
            "ad_revenue": 6700,
            "units": 380,
            "sessions": 6200,
            "conversion_rate": 0.061,
            "acos": 0.41,
            "buy_box_pct": 0.88,
        },
    },
]

# Known anomalies — ground truth for KIRA detection eval
ANOMALIES = [
    {
        "client_id": "natura",
        "start_day": 45,
        "duration_days": 5,
        "type": "buy_box_loss",
        "description": "Competitor undercut Natura price by 8%. Buy Box dropped to 34%.",
        "effects": {"buy_box_pct": -0.57, "revenue": -0.31, "sessions": -0.12},
    },
    {
        "client_id": "vitalblend",
        "start_day": 30,
        "duration_days": 3,
        "type": "acos_spike",
        "description": "Overbid on broad match campaign. ACOS spiked to 61%.",
        "effects": {"acos": 0.27, "ad_revenue": 0.22},
    },
    {
        "client_id": "peakgear",
        "start_day": 60,
        "duration_days": 7,
        "type": "seasonal_spike",
        "description": "Spring hiking season. Sessions and revenue surged.",
        "effects": {"sessions": 0.45, "revenue": 0.38, "units": 0.41},
    },
    {
        "client_id": "lumina",
        "start_day": 20,
        "duration_days": 2,
        "type": "listing_suppression",
        "description": "Main image flagged. Sessions dropped to near zero.",
        "effects": {"sessions": -0.89, "revenue": -0.84, "buy_box_pct": -0.88},
    },
]


def _noise(value: float, pct: float = 0.04) -> float:
    return value * (1 + random.uniform(-pct, pct))


def generate_metrics(num_days: int = 90) -> list[dict]:
    rows = []
    start_date = datetime(2024, 1, 1)
    anomaly_map = {a["client_id"]: a for a in ANOMALIES}

    for client in CLIENTS:
        cid = client["client_id"]
        base = client["baseline"]
        anomaly = anomaly_map.get(cid)

        for day in range(num_days):
            date = start_date + timedelta(days=day)
            mults = {k: 1.0 for k in base}

            in_anomaly = (
                anomaly
                and anomaly["start_day"] <= day < anomaly["start_day"] + anomaly["duration_days"]
            )
            if in_anomaly:
                for metric, effect in anomaly["effects"].items():
                    mults[metric] = 1 + effect

            revenue = _noise(base["revenue"] / 30 * mults.get("revenue", 1))
            ad_rev = _noise(base["ad_revenue"] / 30 * mults.get("ad_revenue", 1))
            sessions = int(_noise(base["sessions"] / 30 * mults.get("sessions", 1)))
            units = max(1, int(_noise(base["units"] / 30 * mults.get("units", 1))))
            buy_box = round(min(1.0, max(0.0,
                _noise(base["buy_box_pct"] * mults.get("buy_box_pct", 1), 0.02)
            )), 3)
            acos = round(min(1.0, max(0.05,
                _noise(base["acos"] * mults.get("acos", 1), 0.03)
            )), 3)
            conversion = round(min(0.3, max(0.01,
                _noise(base["conversion_rate"] * mults.get("conversion_rate", 1), 0.03)
            )), 4)

            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "client_id": cid,
                "client_name": client["client_name"],
                "asin": client["hero_asin"],
                "revenue_usd": round(revenue, 2),
                "ad_revenue_usd": round(ad_rev, 2),
                "organic_revenue_usd": round(max(0, revenue - ad_rev), 2),
                "units_sold": units,
                "sessions": sessions,
                "conversion_rate": conversion,
                "acos": acos,
                "tacos": round(acos * (ad_rev / max(revenue, 1)), 3),
                "buy_box_pct": buy_box,
                "asp_usd": round(revenue / max(units, 1), 2),
                "is_anomaly": bool(in_anomaly),
                "anomaly_type": anomaly["type"] if in_anomaly else None,
            })

    return rows


# ── 3. COMPLIANCE TRAINING DATA ───────────────────────────────────────

COMPLIANCE_EXAMPLES = [
    # Compliant
    {"text": "FAST CHARGING — Powers your device from 0 to 80% in 35 minutes using GaN technology.", "label": "compliant", "violated_rule": None},
    {"text": "PREMIUM MATERIALS — Constructed with aircraft-grade aluminium for lasting durability.", "label": "compliant", "violated_rule": None},
    {"text": "WIDE COMPATIBILITY — Works with iPhone 12 and later, Samsung Galaxy S21 and later.", "label": "compliant", "violated_rule": None},
    {"text": "ENERGY EFFICIENT — Uses 9W to produce the equivalent of a 60W incandescent bulb.", "label": "compliant", "violated_rule": None},
    {"text": "BOVINE COLLAGEN — Sourced from grass-fed cattle. Each serving delivers 10g of Type I and III collagen peptides.", "label": "compliant", "violated_rule": None},
    {"text": "TEMPERATURE RATED — Tested and rated for comfort at temperatures as low as 20°F (-7°C).", "label": "compliant", "violated_rule": None},
    {"text": "EWG VERIFIED — All ingredients meet Environmental Working Group safety standards. Certificate on file.", "label": "compliant", "violated_rule": None},
    {"text": "WATER RESISTANT — IPX4 rated. Withstands splashes and light rain. Not suitable for submersion.", "label": "compliant", "violated_rule": None},
    {"text": "Supports healthy joints and skin elasticity when used as part of a balanced diet.", "label": "compliant", "violated_rule": None},
    {"text": "Formulated for sensitive skin. Free from parabens, sulphates, and synthetic fragrances.", "label": "compliant", "violated_rule": None},
    {"text": "Manufactured in Vietnam to ISO 9001 quality standards. Imported by PeakGear LLC, USA.", "label": "compliant", "violated_rule": None},
    {"text": "Supports skin renewal and helps maintain a youthful appearance.", "label": "compliant", "violated_rule": None},
    {"text": "App-controlled smart lighting. Compatible with Alexa and Google Home.", "label": "compliant", "violated_rule": None},
    {"text": "ALEXA AND GOOGLE HOME COMPATIBLE — Control with voice commands or the Lumina app. HomeKit not supported.", "label": "compliant", "violated_rule": None},
    {"text": "Natural ingredients sourced from certified organic farms. USDA Organic certified.", "label": "compliant", "violated_rule": None},
    # Non-compliant
    {"text": "The best collagen supplement on the market. Guaranteed to transform your skin in 30 days.", "label": "non_compliant", "violated_rule": "superlative_best + guarantee_language"},
    {"text": "Clinically proven to cure joint pain and treat inflammation. Results guaranteed.", "label": "non_compliant", "violated_rule": "medical_claim + guarantee"},
    {"text": "Better than Vital Proteins — same quality at half the price.", "label": "non_compliant", "violated_rule": "competitor_reference"},
    {"text": "#1 rated sleeping bag. Unrivalled warmth. The greatest outdoor gear you will ever buy.", "label": "non_compliant", "violated_rule": "superlatives"},
    {"text": "Chemical-free formula. Completely natural and safe for all skin types.", "label": "non_compliant", "violated_rule": "chemical_free + safe_unqualified"},
    {"text": "On sale now — 40% off limited time only! Act now before price goes back up.", "label": "non_compliant", "violated_rule": "pricing_language_in_listing"},
    {"text": "Pharmaceutical grade magnesium. Doctor recommended formula. Clinically tested.", "label": "non_compliant", "violated_rule": "unsubstantiated_claims"},
    {"text": "DETOX AND CLEANSE — Flushes toxins from your body and purifies your bloodstream.", "label": "non_compliant", "violated_rule": "detox_cleanse_claim"},
    {"text": "Treats sensitive skin conditions including eczema, psoriasis, and rosacea.", "label": "non_compliant", "violated_rule": "medical_claim_treat"},
    {"text": "Military grade materials. Built to the same standards used by US Special Forces.", "label": "non_compliant", "violated_rule": "unsubstantiated_military_grade"},
    {"text": "WATERPROOF — Fully waterproof in all conditions.", "label": "non_compliant", "violated_rule": "false_waterproof_claim"},
    {"text": "Saves 80% on your electricity bill guaranteed. Best smart bulb available anywhere.", "label": "non_compliant", "violated_rule": "unsubstantiated_savings + guarantee + superlative"},
    {"text": "Made in USA with American materials and American labour.", "label": "non_compliant", "violated_rule": "false_country_of_origin"},
    {"text": "Anti-aging serum. Reverses the signs of aging and prevents future wrinkles permanently.", "label": "non_compliant", "violated_rule": "anti_aging + prevent_medical_claim"},
    # Borderline — tests classifier on ambiguous cases
    {"text": "Top seller in the outdoor sleeping bag category.", "label": "non_compliant", "violated_rule": "implied_superlative_unsubstantiated"},
    {"text": "Clinically tested formula — see results in 4 weeks.", "label": "non_compliant", "violated_rule": "clinically_tested_no_citation"},
    {"text": "This product is safe when used as directed. Keep out of reach of children.", "label": "compliant", "violated_rule": None},
    {"text": "Our collagen is sourced from the highest quality bovine sources available.", "label": "compliant", "violated_rule": None},
]


# ── WRITER ────────────────────────────────────────────────────────────

def write_all():
    random.seed(42)
    base = Path("data")

    # 1. Knowledge documents
    raw_dir = base / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata = []

    for doc in KNOWLEDGE_DOCUMENTS:
        path = raw_dir / doc["filename"]
        path.write_text(doc["content"].strip())
        metadata.append({
            "filename": doc["filename"],
            "doc_type": doc["doc_type"],
            "client": doc["client"],
            "char_count": len(doc["content"]),
        })

    with open(raw_dir / "metadata_index.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(raw_dir / "anomaly_ground_truth.json", "w") as f:
        json.dump(ANOMALIES, f, indent=2)

    print(f"Knowledge documents  : {len(KNOWLEDGE_DOCUMENTS)} files → data/raw/")
    print(f"Anomaly ground truth : {len(ANOMALIES)} anomalies → data/raw/anomaly_ground_truth.json")

    # 2. Structured metrics
    metrics_dir = base / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    rows = generate_metrics(num_days=90)

    with open(metrics_dir / "daily_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Structured metrics   : {len(rows)} rows → data/metrics/daily_metrics.csv")
    print(f"  Clients: {len(CLIENTS)} | Days: 90 | Anomalies injected: {len(ANOMALIES)}")

    # 3. Compliance training data
    compliance_dir = base / "compliance"
    compliance_dir.mkdir(exist_ok=True)

    examples = COMPLIANCE_EXAMPLES.copy()
    random.shuffle(examples)
    split = int(len(examples) * 0.8)

    with open(compliance_dir / "train.json", "w") as f:
        json.dump(examples[:split], f, indent=2)
    with open(compliance_dir / "test.json", "w") as f:
        json.dump(examples[split:], f, indent=2)

    compliant = sum(1 for e in COMPLIANCE_EXAMPLES if e["label"] == "compliant")
    non_compliant = sum(1 for e in COMPLIANCE_EXAMPLES if e["label"] == "non_compliant")

    print(f"Compliance examples  : {len(COMPLIANCE_EXAMPLES)} total → data/compliance/")
    print(f"  Train: {split} | Test: {len(examples) - split}")
    print(f"  Compliant: {compliant} | Non-compliant: {non_compliant}")
    print(f"\nDone. Run: python config/settings.py to verify paths.")


if __name__ == "__main__":
    write_all()