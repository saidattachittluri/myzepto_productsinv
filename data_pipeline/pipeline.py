import sqlite3
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd


# CONSTANTS & CONFIGURATION
BASE_URL = "https://books.toscrape.com/catalogue/page-{}.html"
DB_NAME = "zepto_catalog.db"
GBP_TO_INR_RATE = 105.50

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

# SCRAPING STAGE
def scrape_catalog(max_pages: int = 5) -> list[dict]:
    scraped_data = []
    headers = {"User-Agent": "Zepto-Analytics-Guild/1.0"}

    for page in range(1, max_pages + 1):
        url = BASE_URL.format(page)
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"Warning: Page {page} returned status {response.status_code}. Skipping.")
            continue

        soup = BeautifulSoup(response.content, "html.parser")
        articles = soup.find_all("article", class_="product_pod")

        for article in articles:
            title_tag = article.h3.find("a")
            raw_title = title_tag.get("title", "").strip() if title_tag else "Unknown Title"

            price_tag = article.find("p", class_="price_color")
            raw_price = price_tag.text.strip() if price_tag else None

            rating_p = article.find("p", class_="star-rating")
            raw_rating = "Zero"
            if rating_p:
                classes = rating_p.get("class", [])
                rating_classes = [c for c in classes if c != "star-rating"]
                if rating_classes:
                    raw_rating = rating_classes[0]

            avail_p = article.find("p", class_="instock availability")
            raw_avail = avail_p.text.strip() if avail_p else ""

            detail_rel_url = article.h3.find("a")["href"]
            detail_url = f"https://books.toscrape.com/catalogue/{detail_rel_url}"
            category = fetch_category(detail_url, headers)

            scraped_data.append({
                "title": raw_title,
                "raw_price": raw_price,
                "raw_rating": raw_rating,
                "raw_avail": raw_avail,
                "category": category
            })

    print(f"Scraped {len(scraped_data)} records across {max_pages} pages.")
    return scraped_data

def fetch_category(product_url: str, headers: dict) -> str:
    try:
        res = requests.get(product_url, headers=headers, timeout=5)
        if res.status_code == 200:
            sub_soup = BeautifulSoup(res.content, "html.parser")
            breadcrumb = sub_soup.find("ul", class_="breadcrumb")
            if breadcrumb:
                items = breadcrumb.find_all("li")
                if len(items) >= 3:
                    return items[2].text.strip()
    except Exception:
        pass
    return "Default"

# CLEANING & ENRICHMENT STAGE
def clean_and_transform(raw_records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(raw_records)

    def parse_price(val):
        if not val:
            return None
        match = re.search(r"(\d+\.\d+|\d+)", val)
        return float(match.group(1)) if match else None

    df["price_gbp"] = df["raw_price"].apply(parse_price)

    if df["price_gbp"].isnull().any():
        median_price = df["price_gbp"].median()
        df["price_gbp"] = df["price_gbp"].fillna(median_price)

    def parse_rating(val):
        return RATING_MAP.get(val, None)

    df["rating"] = df["raw_rating"].apply(parse_rating)
    if df["rating"].isnull().any():
        med_rating = int(df["rating"].median())
        df["rating"] = df["rating"].fillna(med_rating).astype(int)
    else:
        df["rating"] = df["rating"].astype(int)

    df["in_stock"] = df["raw_avail"].str.contains("In stock", case=False, na=False)

    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR_RATE).round(2)

    cleaned_df = df[["title", "category", "price_gbp", "price_inr", "rating", "in_stock"]].copy()
    return cleaned_df

# 3. RELATIONAL SCHEMA & LOADING
def init_db_and_load(df: pd.DataFrame, db_path: str = DB_NAME):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("PRAGMA foreign_keys = ON;")

    cursor.execute("DROP TABLE IF EXISTS books;")
    cursor.execute("DROP TABLE IF EXISTS categories;")

    cursor.execute("""
    CREATE TABLE categories (
        category_id INTEGER PRIMARY KEY AUTOINCREMENT,
        category_name TEXT UNIQUE NOT NULL
    );
    """)

    cursor.execute("""
    CREATE TABLE books (
        book_id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        price_gbp REAL NOT NULL,
        price_inr REAL NOT NULL,
        rating INTEGER NOT NULL,
        in_stock INTEGER NOT NULL,
        category_id INTEGER NOT NULL,
        FOREIGN KEY (category_id) REFERENCES categories(category_id)
    );
    """)

    categories = sorted(df["category"].unique())
    category_map = {}
    for cat in categories:
        cursor.execute("INSERT INTO categories (category_name) VALUES (?);", (cat,))
        category_map[cat] = cursor.lastrowid

    books_data = []
    for _, row in df.iterrows():
        books_data.append((
            row["title"],
            row["price_gbp"],
            row["price_inr"],
            row["rating"],
            1 if row["in_stock"] else 0,
            category_map[row["category"]]
        ))

    cursor.executemany("""
    INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
    VALUES (?, ?, ?, ?, ?, ?);
    """, books_data)

    conn.commit()
    conn.close()
    print(f"Loaded {len(categories)} categories and {len(books_data)} books into {db_path}.")

# SQL QUERY SUITE & PANDAS VERIFICATION
def run_sql_demonstrations(db_path: str = DB_NAME):
    conn = sqlite3.connect(db_path)

    queries = {
        "Query 1 (SELECT, WHERE, ORDER BY, LIMIT)": """
            SELECT title, price_gbp, price_inr, rating 
            FROM books 
            WHERE rating = 5 
            ORDER BY price_inr DESC 
            LIMIT 5;
        """,
        "Query 2 (DISTINCT categories)": """
            SELECT DISTINCT category_name 
            FROM categories 
            ORDER BY category_name ASC;
        """,
        "Query 3 (BETWEEN filter)": """
            SELECT title, price_inr, rating 
            FROM books 
            WHERE price_inr BETWEEN 2000 AND 3500 
            ORDER BY price_inr ASC 
            LIMIT 5;
        """,
        "Query 4 (IN filter)": """
            SELECT title, rating, in_stock 
            FROM books 
            WHERE rating IN (1, 5) 
            LIMIT 5;
        """,
        "Query 5 (JOIN between books and categories)": """
            SELECT b.book_id, b.title, c.category_name, b.price_inr, b.rating
            FROM books b
            INNER JOIN categories c ON b.category_id = c.category_id
            WHERE b.rating >= 4
            ORDER BY b.price_inr DESC
            LIMIT 5;
        """
    }

    print("\n" + "="*80)
    print("EXECUTING MANDATORY SQL DEMONSTRATIONS")
    print("="*80)

    for label, sql in queries.items():
        print(f"\n--- {label} ---")
        print(f"SQL:\n{sql.strip()}\n")
        df_res = pd.read_sql_query(sql, conn)
        print(df_res.to_string(index=False))

    # PANDAS VS SQL EQUIVALENCE TEST
    print("\n" + "="*80)
    print("Verification: pd.read_sql VS pd.merge (Side-by-Side Equivalence)")
    print("="*80)

    join_sql = queries["Query 5 (JOIN between books and categories)"]
    sql_joined_df = pd.read_sql_query(join_sql, conn)

    books_df = pd.read_sql_query("SELECT * FROM books;", conn)
    categories_df = pd.read_sql_query("SELECT * FROM categories;", conn)

    merged_df = pd.merge(
        books_df,
        categories_df,
        on="category_id",
        how="inner"
    )
    filtered_merged_df = (
        merged_df[merged_df["rating"] >= 4][["book_id", "title", "category_name", "price_inr", "rating"]]
        .sort_values(by="price_inr", ascending=False)
        .head(5)
        .reset_index(drop=True)
    )

    print("\n[Method A: pd.read_sql with INNER JOIN]")
    print(sql_joined_df)

    print("\n[Method B: pd.merge in Memory]")
    print(filtered_merged_df)

    is_equivalent = sql_joined_df.equals(filtered_merged_df)
    print(f"\nExact Equivalence Match: {is_equivalent}")

    conn.close()

if __name__ == "__main__":
    records = scrape_catalog(max_pages=5)
    cleaned = clean_and_transform(records)
    init_db_and_load(cleaned)
    run_sql_demonstrations()