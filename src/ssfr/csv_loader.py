"""Validated product CSV ingestion."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

from .types import ImportReport, ProductRecord


class ProductCSVLoader:
    REQUIRED_COLUMNS = {"product_id", "title", "description"}
    OPTIONAL_COLUMNS = {
        "category",
        "brand",
        "price_ron",
        "color",
        "audience",
        "in_stock",
    }

    def __init__(self, *, tolerant: bool = True) -> None:
        self.tolerant = tolerant
        self.last_report: ImportReport | None = None

    @staticmethod
    def _encoding(path: Path) -> str:
        with path.open("rb") as handle:
            prefix = handle.read(3)
        return "utf-8-sig" if prefix == b"\xef\xbb\xbf" else "utf-8"

    @staticmethod
    def _optional_text(row: dict[str, str | None], key: str) -> str | None:
        value = row.get(key)
        if value is None:
            return None
        value = value.strip()
        return value or None

    @staticmethod
    def _parse_price(value: str | None) -> float | None:
        if value is None or not value.strip():
            return None
        parsed = float(value.strip().replace(" ", "").replace(",", "."))
        if parsed < 0:
            raise ValueError("price_ron cannot be negative")
        return parsed

    @staticmethod
    def _parse_stock(value: str | None) -> bool:
        if value is None or not value.strip():
            return True
        normalized = value.strip().casefold()
        truthy = {"1", "true", "yes", "y", "da", "d"}
        falsy = {"0", "false", "no", "n", "nu"}
        if normalized in truthy:
            return True
        if normalized in falsy:
            return False
        raise ValueError("in_stock must be one of 1/0, true/false, yes/no, or da/nu")

    def load(self, path: str | Path) -> list[ProductRecord]:
        products = [
            product
            for batch in self.iter_batches(path, batch_size=10_000)
            for product in batch
        ]
        if not products:
            raise ValueError("CSV contains no valid product rows")
        return products

    def iter_batches(
        self,
        path: str | Path,
        batch_size: int = 10_000,
    ) -> Iterator[list[ProductRecord]]:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"CSV file does not exist: {source}")
        encoding = self._encoding(source)
        batch: list[ProductRecord] = []
        invalid_rows: list[dict[str, object]] = []
        seen_ids: set[str] = set()
        duplicate_ids = 0
        rows_read = 0
        rows_valid = 0

        with source.open("r", encoding=encoding, newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError("CSV has no header")
            fieldnames = {name.strip() for name in reader.fieldnames if name is not None}
            missing = sorted(self.REQUIRED_COLUMNS - fieldnames)
            if missing:
                raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

            for line_number, row in enumerate(reader, start=2):
                if not row or all(value is None or not str(value).strip() for value in row.values()):
                    continue
                rows_read += 1
                try:
                    product_id = (row.get("product_id") or "").strip()
                    title = (row.get("title") or "").strip()
                    description = (row.get("description") or "").strip()
                    if not product_id:
                        raise ValueError("product_id is empty")
                    if not title:
                        raise ValueError("title is empty")
                    if not description:
                        raise ValueError("description is empty")
                    if product_id in seen_ids:
                        duplicate_ids += 1
                        raise ValueError(f"duplicate product_id: {product_id}")
                    record = ProductRecord(
                        product_id=product_id,
                        title=title,
                        description=description,
                        category=self._optional_text(row, "category"),
                        brand=self._optional_text(row, "brand"),
                        price_ron=self._parse_price(row.get("price_ron")),
                        color=self._optional_text(row, "color"),
                        audience=self._optional_text(row, "audience"),
                        in_stock=self._parse_stock(row.get("in_stock")),
                    )
                    batch.append(record)
                    rows_valid += 1
                    seen_ids.add(product_id)
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
                except (TypeError, ValueError) as exc:
                    invalid = {
                        "line_number": line_number,
                        "product_id": (row.get("product_id") or "").strip(),
                        "error": str(exc),
                    }
                    invalid_rows.append(invalid)
                    if not self.tolerant:
                        raise ValueError(
                            f"invalid CSV row at line {line_number}: {exc}"
                        ) from exc

        if batch:
            yield batch
        self.last_report = ImportReport(
            source=str(source),
            rows_read=rows_read,
            rows_valid=rows_valid,
            rows_invalid=len(invalid_rows),
            duplicate_ids=duplicate_ids,
            encoding=encoding,
            invalid_rows=tuple(invalid_rows),
        )


def build_semantic_text(product: ProductRecord) -> str:
    fields = [
        f"Titlu: {product.title}.",
        f"Descriere: {product.description}.",
    ]
    if product.category:
        fields.append(f"Categorie: {product.category}.")
    if product.brand:
        fields.append(f"Brand: {product.brand}.")
    if product.color:
        fields.append(f"Culoare: {product.color}.")
    if product.audience:
        fields.append(f"Public: {product.audience}.")
    return " ".join(fields)
