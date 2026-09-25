# Module 1 — Data Pipeline: Catalog Intelligence

## 1. Overview
This module extracts competitive product pricing and stock availability from `books.toscrape.com`, transforms raw HTML string fields into normalized data types, maps items to a relational 3NF SQLite schema, and exposes analytical query interfaces via SQL and pandas.

## 2. Fixed Baseline Conversion Rate
All conversions from GBP to INR use the project-defined constant:
$$\text{Conversion Rate}: 1\text{ GBP} = 105.50\text{ INR}$$
* **Formula:** `price_inr = round(price_gbp * 105.50, 2)`
* This calculation operates as an isolated, keyless transformation with no external network dependency.

## 3. Data Cleaning & Imputation Strategy
* **`price_gbp`**: Extracted using regular expression `(\d+\.\d+|\d+)` to strip currency symbols. Any unparseable row is imputed using the column **median** to prevent pipeline termination while avoiding skew from extreme values.
* **`rating`**: Extracted from CSS classes (`One` through `Five`) mapped directly to integers `1` through `5`. Non-matches default to the median integer rating.
* **`in_stock`**: Evaluated as a boolean flag (`True`/`False`) based on the substring `"In stock"`.
* **Missing Categories**: Fallbacks default to `"Default"` if the breadcrumb element is absent.

## 4. Relational Schema Design
The SQLite database (`zepto_catalog.db`) enforces a normalized Two-Table Primary/Foreign Key structure:

```sql
categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
);

books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    price_gbp REAL NOT NULL,
    price_inr REAL NOT NULL,
    rating INTEGER NOT NULL,
    in_stock INTEGER NOT NULL,
    category_id INTEGER NOT NULL,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);