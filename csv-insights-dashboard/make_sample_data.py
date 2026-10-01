"""Generate sample_data/sales.csv: a year of made-up online/store orders.

The data is deliberately a little messy (missing ratings, stray spaces,
duplicate rows) so the dashboard's cleaning step has something to fix.

Run:  python make_sample_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "sample_data" / "sales.csv"

PRODUCTS = {
    "Wireless Earbuds": 49.0,
    "Smart Watch": 129.0,
    "Laptop Stand": 35.0,
    "USB-C Hub": 29.0,
    "Bluetooth Speaker": 69.0,
}


def make_sales(n: int = 1200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    order_date = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    product = rng.choice(list(PRODUCTS), n, p=[0.3, 0.15, 0.2, 0.2, 0.15])
    region = rng.choice(["North", "South", "East", "West"], n, p=[0.3, 0.25, 0.25, 0.2])
    channel = rng.choice(["Online", "Store"], n, p=[0.6, 0.4])

    # More units per order in the Nov/Dec holiday season.
    holiday = np.asarray(order_date.month >= 11)
    units = rng.integers(1, 5, n) + holiday * rng.integers(0, 3, n)

    discount_pct = rng.choice([0, 5, 10, 15, 20, 30], n, p=[0.35, 0.2, 0.18, 0.12, 0.1, 0.05])
    unit_price = np.array([PRODUCTS[p] for p in product]) * rng.normal(1, 0.03, n)
    revenue = units * unit_price * (1 - discount_pct / 100)

    online = channel == "Online"
    delivery_days = np.where(online, rng.integers(1, 9, n), 0)

    # Slow deliveries and smart watches get worse ratings; bad ratings get returned.
    rating = 4.4 - 0.22 * delivery_days - 0.5 * (product == "Smart Watch") + rng.normal(0, 0.6, n)
    customer_rating = np.clip(np.round(rating), 1, 5)
    return_odds = -1.2 - 2.4 * (customer_rating - 3) + 0.4 * (discount_pct >= 20)
    returned = np.where(rng.random(n) < 1 / (1 + np.exp(-return_odds)), "Yes", "No")

    df = pd.DataFrame(
        {
            "order_id": [f"ORD-{10001 + i}" for i in range(n)],
            "order_date": order_date.strftime("%Y-%m-%d"),
            "region": region,
            "channel": channel,
            "product": product,
            "units": units,
            "unit_price": unit_price.round(2),
            "discount_pct": discount_pct.astype(float),
            "revenue": revenue.round(2),
            "delivery_days": delivery_days,
            "customer_rating": customer_rating,
            "returned": returned,
        }
    )

    # Real-world mess: unrated orders, missing discounts, stray spaces, duplicate rows.
    df.loc[rng.random(n) < 0.05, "customer_rating"] = np.nan
    df.loc[rng.random(n) < 0.02, "discount_pct"] = np.nan
    spaced = rng.random(n) < 0.03
    df.loc[spaced, "region"] = " " + df.loc[spaced, "region"] + " "
    duplicates = df.sample(15, random_state=seed)
    df = pd.concat([df, duplicates]).sort_values("order_date", kind="stable")
    return df.reset_index(drop=True)


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    make_sales().to_csv(OUT, index=False)
    print(f"Wrote {OUT}")
