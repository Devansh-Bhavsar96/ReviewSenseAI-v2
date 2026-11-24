"""Create data/sample_master.csv - SYNTHETIC demo data so the app runs without your real dataset.

The app automatically prefers data/reviewsense_master.csv (your real data) when it exists.
"""
import random
from pathlib import Path

import pandas as pd

random.seed(7)
OUT = Path(__file__).resolve().parent / "data" / "sample_master.csv"

LAPTOPS = ["Demo Laptop Air 14", "Demo Laptop Pro 16", "Demo Gamer X15", "Demo Student Book 15",
           "Demo Creator Studio 16", "Demo Budget Lite 15"]
BIAS = [0.7, 0.75, 0.55, 0.5, 0.7, 0.35]  # chance a review is positive
POS = ["The battery lasts all day and charging is quick.", "Display is bright and colors look sharp.",
       "Performance is smooth even with many tabs open.", "Runs cool and the fan stays quiet.",
       "Great build quality and a comfortable keyboard.", "Fast boot and very responsive for the price.",
       "Screen is crisp and perfect for watching movies.", "Excellent value, I am very happy with it."]
NEG = ["The battery drains quickly, barely a few hours.", "It heats up a lot and the fan is loud.",
       "The screen is dim and colors look washed out.", "Performance is slow and it lags when multitasking.",
       "Poor battery backup, needs the charger all the time.", "Gets hot while gaming and stutters.",
       "Display is average and the laptop feels heavy.", "Not worth the money, I regret buying it."]
NEU = ["It is okay for the price, nothing special.", "Average performance, battery is fine.",
       "Does the job but the display is just average."]

reviews = []
for name, bias in zip(LAPTOPS, BIAS):
    for _ in range(120):
        r = random.random()
        if r < bias:
            rating, pool = random.choice([4, 5, 5]), POS
        elif r < bias + 0.12:
            rating, pool = 3, NEU
        else:
            rating, pool = random.choice([1, 2]), NEG
        text = " ".join(random.sample(pool, k=min(len(pool), random.choice([1, 2, 2, 3]))))
        reviews.append({"record_type": "review", "laptop": name, "title": "", "review": text,
                        "rating": rating, "sentiment": ""})

CATALOG = [  # name, price, ram, storage, inches, resolution, gpu, use_case
    ("Sample Everyday 14", 42990, 8, 512, 14, "1920x1080", "Integrated", "office"),
    ("Sample Student 15", 36990, 8, 512, 15.6, "1920x1080", "Integrated", "study_office"),
    ("Sample Slim Pro 14", 79990, 16, 512, 14, "2880x1800", "Integrated", "creator"),
    ("Sample Creator 16", 129990, 32, 1024, 16, "2560x1600", "NVIDIA GeForce RTX 4060", "creator"),
    ("Sample Gamer 15", 89990, 16, 512, 15.6, "1920x1080", "NVIDIA GeForce RTX 4050", "gaming"),
    ("Sample Gamer Max 17", 149990, 32, 1024, 17.3, "2560x1440", "NVIDIA GeForce RTX 4070", "gaming"),
    ("Sample Budget 15", 29990, 8, 256, 15.6, "1366x768", "Integrated", "study_office"),
    ("Sample Ultralight 13", 94990, 16, 1024, 13.3, "2560x1600", "Integrated", "office"),
]
catalog = [{"record_type": "catalog", "laptop": n, "price": p, "price_currency": "INR",
            "price_as_of": "SAMPLE DATA", "ram_gb": ram, "storage_gb": st, "screen_inches": inch,
            "display_resolution": res, "gpu": gpu, "use_case": uc, "recommendation_eligible": True,
            "catalog_group": "sample"} for n, p, ram, st, inch, res, gpu, uc in CATALOG]

OUT.parent.mkdir(exist_ok=True)
pd.DataFrame(reviews + catalog).to_csv(OUT, index=False)
print(f"Wrote {OUT} ({len(reviews)} reviews, {len(catalog)} catalog rows)")
