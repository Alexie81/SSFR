from __future__ import annotations

from pathlib import Path

import pytest

from ssfr.catalog import CatalogIndex
from ssfr.csv_loader import ProductCSVLoader, build_semantic_text


def test_csv_loader_reads_utf8_catalog() -> None:
    loader = ProductCSVLoader()
    products = loader.load(Path("data/products.csv"))
    assert len(products) == 20
    assert loader.last_report is not None
    assert loader.last_report.rows_invalid == 0
    assert "Adidași" in build_semantic_text(products[0])


def test_csv_loader_streams_real_batches() -> None:
    loader = ProductCSVLoader()
    batches = list(loader.iter_batches(Path("data/products.csv"), batch_size=7))
    assert [len(batch) for batch in batches] == [7, 7, 6]
    assert loader.last_report is not None
    assert loader.last_report.rows_valid == 20


def test_csv_loader_missing_columns_is_explicit(tmp_path) -> None:
    path = tmp_path / "invalid.csv"
    path.write_text("product_id,title\nP1,Item\n", encoding="utf-8")
    with pytest.raises(ValueError, match="description"):
        ProductCSVLoader().load(path)


def test_tolerant_csv_reports_duplicate_and_invalid_rows(tmp_path) -> None:
    path = tmp_path / "mixed.csv"
    path.write_text(
        "product_id,title,description,price_ron,in_stock\n"
        "P1,One,Valid,12.5,da\n"
        "P1,Duplicate,Valid,13,1\n"
        "P2,Two,Valid,invalid,maybe\n",
        encoding="utf-8",
    )
    loader = ProductCSVLoader(tolerant=True)
    products = loader.load(path)
    assert len(products) == 1
    assert loader.last_report is not None
    assert loader.last_report.rows_invalid == 2
    assert loader.last_report.duplicate_ids == 1


def test_catalog_build_load_and_search_without_rebuild(tmp_path) -> None:
    artifact = tmp_path / "catalog"
    catalog, report = CatalogIndex.build(
        "data/products.csv",
        artifact,
        shard_count=4,
        bands=(1, 2),
        probe_shards=2,
        embedding_provider="hash",
        embedding_dimension=96,
        local_index_backend="exact",
    )
    assert report["products_loaded"] == 20
    first = catalog.search_text(
        "telefon cu baterie mare și cameră bună",
        top_k=3,
        probe_shards=4,
        report_path=None,
    )
    loaded = CatalogIndex.load(artifact)
    second = loaded.search_text(
        "telefon cu baterie mare și cameră bună",
        top_k=3,
        probe_shards=4,
        report_path=None,
    )
    assert first.search.item_ids.tolist() == second.search.item_ids.tolist()
    assert "P0008" in second.search.item_ids.tolist()


def test_pre_and_post_filters_are_supported(tmp_path) -> None:
    catalog, _ = CatalogIndex.build(
        "data/products.csv",
        tmp_path / "catalog",
        shard_count=4,
        bands=(1, 2),
        probe_shards=4,
        embedding_provider="hash",
        embedding_dimension=64,
        local_index_backend="exact",
    )
    for strategy in ("pre", "post"):
        result = catalog.search_text(
            "adidași pentru alergare",
            top_k=5,
            probe_shards=4,
            color="negru",
            price_max=500,
            in_stock_only=True,
            filter_strategy=strategy,
            report_path=None,
        )
        assert all(product.color == "negru" for product in result.products)
        assert all(
            product.price_ron is not None and product.price_ron <= 500
            for product in result.products
        )
