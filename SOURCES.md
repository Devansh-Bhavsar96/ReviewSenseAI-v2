# Dataset provenance

The default data file, `data/flipkart_laptop_reviews.csv`, is a real laptop customer-review dataset mirrored in the public GitHub repository [rj285/Module9Task1-flipkart-laptop-reviews-eda](https://github.com/rj285/Module9Task1-flipkart-laptop-reviews-eda). Its README attributes the dataset to [Flipkart Laptop Reviews on Kaggle](https://www.kaggle.com/datasets/gitadityamaddali/flipkart-laptop-reviews) and describes more than 24,000 reviews across nearly 600 laptop models. The checked-in CSV contains 24,113 rows and fields including product name, individual rating, review title, and review text.

The app maps the individual 1–5 star rating into a coarse sentiment label: 1–2 negative, 3 neutral, and 4–5 positive. These are rating-derived labels, not human-annotated sentiment labels. The app retains the rating and shows the machine-learning prediction separately.

The checked-in CSV was downloaded from the [public GitHub mirror's raw data file](https://raw.githubusercontent.com/rj285/Module9Task1-flipkart-laptop-reviews-eda/main/laptops_dataset_final_600.csv).

The mirrored repository page does not state a dataset license. The reviews are included for this local academic project and source attribution; verify the Kaggle/source terms before redistributing them or using them commercially. This dataset does not provide current prices, battery benchmarks, or complete specifications. The app does not infer those values as facts.

`data/laptops_dataset_final_600.csv` is a user-provided review-file copy retained in `data/` for provenance. It is not read by the running app or by the master-dataset builder. Earlier inspection found the same product/rating/title/review combinations as the bundled review file, so it was excluded from the master to avoid importing the duplicate copy. The data has no reviewer IDs; identical text may sometimes be a repeated submission rather than an accidental duplicate.

## Additional Parquet review sentences

The project also includes five user-provided Parquet shards (`data/datasethf.parquet`, `datasethf3.parquet`, `datasethf4.parquet`, `datasethf5.parquet`, and `datasethf6.parquet`) and the accompanying `data/hf_dataset_gitattributes` metadata. The shards contain tokenized laptop-review sentences and numeric token labels, but no product-model field, star rating, dataset citation, repository ID, or license metadata was supplied. The exact originating dataset therefore cannot be confirmed from the files alone.

The shards contain 4,795 rows in total. Two shard pairs repeat one another exactly; removing repeated sentences leaves 3,826 unique review sentences. These are combined into `data/additional_review_sources.csv` for the app. Because product names are absent, the app groups them under “Laptop model not specified”; because star ratings are absent, it uses the app's predicted sentiment and keeps these separate from rating-derived labels. The token labels are retained as extracted `aspect_terms`, using the observed 0/1/2 begin/inside/outside pattern. Since no label dictionary accompanied the files, treat extracted terms as a best-effort interpretation, not verified ground truth. No original source URL or license is inferred from the filename or `.gitattributes` file.

## 2026 India laptop price snapshot

`data/laptop_prices_2026.csv` is a curated snapshot of 30 current laptop listings and prices in INR. The main entries were transcribed from the [Gadgets 360 India laptop price list](https://www.gadgets360.com/laptops/price-list-india), which reported its list last updated on 17 October 2026. One older TUF F16 price is from a dated 24 September 2026 list on the same site. Two gaming listings were taken from [Gaming Laptop Deals India](https://gaminglaptop.deals/india), which reported prices checked about two weeks before this snapshot. Each row includes its price-check date and source URL.

This is a small catalog snapshot, not a complete national price database. Prices and stock can change by seller, configuration, discounts, and location. Specification cells are left blank when the cited listing did not provide the value used by the app. The `use_case` labels are simple product-family groupings for recommendation ranking, not manufacturer claims. Newer 2026 products generally do not have exact matches in the older Flipkart reviews dataset; the app therefore ranks these catalog entries by price, listed specs, and intended use without presenting unrelated reviews as if they were for the same laptop.

## Additional laptop specification catalogs

`data/Cleaned_laptop_dataset.csv` and `data/laptop_dataset.csv` were added locally by the project owner. They are retained as source archives but excluded from the active master dataset and recommendation UI. Their prices lack source, currency, and collection-date metadata, and their generated names such as “catalog item 979” were not useful shopper-facing product names.

These files do not include source, collection date, or price currency metadata. Their supplied price values are excluded from the active app because they cannot be safely compared with the INR budget. They do not enter the sentiment model, review summaries, complaint counts, or recommendations.

## Shared normalized dataset used by the app

`data/reviewsense_master.csv` is the only data file loaded by the running Streamlit app. Its `record_type` column distinguishes review rows from catalog rows, and `catalog_group` identifies the dated INR price snapshot. Reviews and Recommendations use filtered views of this same file; review records are not fabricated for catalog-only laptops, and catalog prices are not inferred for reviewed models.

The generated file currently contains 20,800 review rows: 16,974 exact-unique rows from the main Flipkart review source and 3,826 deduplicated sentences from the additional Parquet source. Exact duplicates are removed using normalized product, rating, title, and review text. Because reviewer IDs are absent, identical text from different people cannot be distinguished and may be collapsed. Review sentiment is derived from ratings when available; unrated Parquet sentences receive model-predicted sentiment in the app.

It also contains 30 dated INR price snapshot entries, for 20,830 total rows. The 1,272 supplemental laptop specifications are excluded from this active file because their names and prices are not trustworthy enough for shopper-facing recommendations.

To rebuild the master file from the retained local source files, run `python build_master_dataset.py` from the project folder. The builder does not fetch or scrape retailer websites. No Amazon listing feed or live Flipkart/Amazon price API is connected, so this file is not a complete, current list of products available on those marketplaces.
