"""Build ReviewSense AI's single normalized reviews + laptop-catalog CSV.

The source files are inputs to this build only. The Streamlit app reads only
data/reviewsense_master.csv. No marketplace pages are scraped by this script.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import date
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = DATA / "reviewsense_master.csv"
BUILT_ON = date.today().isoformat()
KAGGLE_URL = "https://www.kaggle.com/datasets/gitadityamaddali/flipkart-laptop-reviews"
GITHUB_URL = "https://github.com/rj285/Module9Task1-flipkart-laptop-reviews-eda"

FIELDS = [
    "record_type", "record_id", "built_on", "provider_name", "marketplace",
    "laptop", "product_id", "title", "review", "rating", "sentiment",
    "aspect_terms", "price", "price_currency", "price_as_of", "price_is_live",
    "available", "recommendation_eligible", "ram_gb", "storage_gb",
    "screen_inches", "display_resolution", "processor", "gpu", "use_case",
    "catalog_group", "review_source_url", "price_source_url", "source_note",
]


def text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def number(value: object) -> float | None:
    parsed = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(parsed) else float(parsed)


def stable_id(*parts: object) -> str:
    joined = "\x1f".join(text(part) for part in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]


def canonical(value: object) -> str:
    value = unicodedata.normalize("NFKC", text(value)).casefold()
    return re.sub(r"\s+", " ", value).strip()


def sentiment_from_rating(value: object) -> str:
    rating = number(value)
    if rating is None:
        return ""
    return "positive" if rating >= 4 else ("neutral" if rating == 3 else "negative")


def base_record(**values: object) -> dict[str, object]:
    record = {field: "" for field in FIELDS}
    record.update(values)
    record["built_on"] = BUILT_ON
    return record


def build_reviews() -> tuple[list[dict[str, object]], dict[str, int]]:
    primary_path = DATA / "flipkart_laptop_reviews.csv"
    additional_path = DATA / "additional_review_sources.csv"
    primary = pd.read_csv(primary_path, low_memory=False).rename(
        columns={"product_name": "laptop"}
    )
    additional = pd.read_csv(additional_path, low_memory=False)
    for frame in (primary, additional):
        for column, default in {"laptop": "", "review": "", "title": "", "rating": None,
                               "data_source": "", "aspect_terms": ""}.items():
            if column not in frame:
                frame[column] = default

    # Exact text/rating/product repeats are collapsed across all review inputs.
    # Reviewer IDs are absent, so near-duplicates are deliberately retained.
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, str, str]] = set()
    primary_count = 0
    added_count = 0
    duplicate_count = 0

    sources = [
        (primary, "Flipkart Laptop Reviews (Kaggle via public GitHub mirror)",
         "Flipkart review dataset; no exact marketplace listing match is implied.", KAGGLE_URL),
        (additional, "User-provided additional review sentences", "Review input; source/license details not supplied for some rows.", ""),
    ]
    for frame, provider, note, url in sources:
        for row in frame.to_dict(orient="records"):
            laptop = text(row.get("laptop"))
            review = text(row.get("review"))
            if not laptop or not review:
                continue
            rating = number(row.get("rating"))
            title = text(row.get("title"))
            key = (canonical(laptop), "" if rating is None else f"{rating:g}",
                   canonical(title), canonical(review))
            if key in seen:
                duplicate_count += 1
                continue
            seen.add(key)
            row_source = text(row.get("data_source"))
            if row_source.startswith("Provided Parquet:"):
                provider = row_source
                marketplace = "Unspecified"
                source_note = "Laptop model, star rating, upstream source URL, and license were not supplied."
                review_url = ""
            elif frame is primary:
                marketplace = "Flipkart review dataset"
                source_note = "Rating-derived sentiment; historical review data, not a current listing."
                review_url = url
            else:
                marketplace = "Unspecified"
                source_note = note
                review_url = ""
            rows.append(base_record(
                record_type="review", record_id=stable_id("review", *key),
                provider_name=provider, marketplace=marketplace, laptop=laptop,
                title=title, review=review, rating="" if rating is None else rating,
                sentiment=sentiment_from_rating(rating), aspect_terms=text(row.get("aspect_terms")),
                available="", recommendation_eligible=False,
                review_source_url=review_url, source_note=source_note,
            ))
            if frame is primary:
                primary_count += 1
            else:
                added_count += 1
    return rows, {"flipkart_review_rows": primary_count, "additional_unique_rows": added_count,
                  "exact_duplicates_removed": duplicate_count}


def build_catalog() -> tuple[list[dict[str, object]], dict[str, int]]:
    records: list[dict[str, object]] = []
    prices_path = DATA / "laptop_prices_2026.csv"
    prices = pd.read_csv(prices_path, low_memory=False)
    for row in prices.to_dict(orient="records"):
        laptop = text(row.get("laptop"))
        price = number(row.get("price_inr"))
        if not laptop or price is None:
            continue
        records.append(base_record(
            record_type="catalog", record_id=stable_id("price", laptop, price, row.get("price_as_of")),
            provider_name="Gadgets 360 India / Gaming Laptop Deals India",
            marketplace="Price-list sources; not a full Amazon/Flipkart feed", laptop=laptop,
            price=price, price_currency="INR", price_as_of=text(row.get("price_as_of")),
            price_is_live=False, available="Unknown", recommendation_eligible=True,
            ram_gb=number(row.get("ram_gb")), storage_gb=number(row.get("storage_gb")),
            screen_inches=number(row.get("screen_inches")),
            display_resolution=text(row.get("display_resolution")),
            processor=text(row.get("processor")), gpu=text(row.get("gpu")),
            use_case=text(row.get("use_case")), catalog_group="dated_inr_price_snapshot",
            price_source_url=text(row.get("source_url")),
            source_note="Dated snapshot; verify price and stock with the seller before purchase.",
        ))

    return records, {"dated_inr_price_rows": len(prices)}


def main() -> None:
    reviews, review_counts = build_reviews()
    catalog, catalog_counts = build_catalog()
    frame = pd.DataFrame(reviews + catalog, columns=FIELDS)
    frame.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print({
        "output": str(OUTPUT), "total_rows": len(frame),
        "review_rows": int(frame.record_type.eq("review").sum()),
        "catalog_rows": int(frame.record_type.eq("catalog").sum()),
        "recommendation_eligible_inr_rows": int(frame.recommendation_eligible.astype(str).str.lower().eq("true").sum()),
        "review_details": review_counts, "catalog_details": catalog_counts,
    })


if __name__ == "__main__":
    main()
