from __future__ import annotations

import sqlite3

from tools.generate_million_products import categories, generate


def test_taxonomy_contains_256_unique_categories() -> None:
    taxonomy = categories()
    assert len(taxonomy) == 256
    assert len({item["category"] for item in taxonomy}) == 256
    assert len({item["category_id"] for item in taxonomy}) == 256


def test_generator_writes_valid_csv_and_sqlite(tmp_path) -> None:
    summary = generate(
        tmp_path,
        row_count=512,
        category_count=256,
        batch_size=73,
        seed=42,
        force=False,
    )
    assert summary["product_count"] == 512
    assert summary["category_count"] == 256
    validation = summary["validation"]
    assert validation["csv_product_count"] == 512
    assert validation["sqlite_product_count"] == 512
    assert validation["sqlite_unique_product_ids"] == 512
    assert validation["products_per_category_min"] == 2
    assert validation["products_per_category_max"] == 2
    with sqlite3.connect(summary["sqlite_path"]) as connection:
        first = connection.execute(
            "SELECT product_id, category_id FROM products ORDER BY product_id LIMIT 1"
        ).fetchone()
        last = connection.execute(
            "SELECT product_id, category_id FROM products ORDER BY product_id DESC LIMIT 1"
        ).fetchone()
    assert first == ("P0000001", 1)
    assert last == ("P0000512", 256)
