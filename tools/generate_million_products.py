"""Generate a physical million-product catalog for SSFR load testing.

The generator writes both the CSV consumed by SSFR and an indexed SQLite database
that can be inspected independently. Rows are deterministic and streamed in
batches; the full catalog is never held in memory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter


@dataclass(frozen=True)
class Department:
    name: str
    subcategories: tuple[str, ...]
    brands: tuple[str, ...]
    price_min: float
    price_max: float


DEPARTMENTS = (
    Department(
        "Electronice",
        (
            "Telefoane",
            "Laptopuri",
            "Tablete",
            "Televizoare",
            "Monitoare",
            "Căști audio",
            "Boxe portabile",
            "Camere foto",
            "Routere Wi-Fi",
            "Ceasuri inteligente",
            "Console gaming",
            "Imprimante",
            "Stocare externă",
            "Componente PC",
            "Accesorii mobile",
            "Proiectoare",
        ),
        ("Nova", "DevPro", "Visionix", "AudioMax", "PixelOne", "NexTech", "Quantum", "Smartix"),
        49,
        14_999,
    ),
    Department(
        "Electrocasnice",
        (
            "Frigidere",
            "Mașini de spălat",
            "Uscătoare",
            "Mașini de spălat vase",
            "Cuptoare",
            "Plite",
            "Hote",
            "Aspiratoare",
            "Roboți de aspirare",
            "Aparate de aer condiționat",
            "Espressoare",
            "Blendere",
            "Friteuze cu aer cald",
            "Fiare de călcat",
            "Purificatoare",
            "Boilere",
        ),
        ("HomeFlow", "ArcticPlus", "Cleanix", "CookMaster", "FreshAir", "AquaLine", "ThermoMax", "RoboClean"),
        79,
        19_999,
    ),
    Department(
        "Casă și mobilier",
        (
            "Canapele",
            "Fotolii",
            "Mese",
            "Scaune",
            "Paturi",
            "Saltele",
            "Dulapuri",
            "Comode",
            "Biblioteci",
            "Mobilier bucătărie",
            "Corpuri de iluminat",
            "Covoare",
            "Perdele",
            "Lenjerii",
            "Decorațiuni",
            "Organizare",
        ),
        ("NordicHome", "CasaNova", "WoodCraft", "ComfortLine", "UrbanNest", "Decora", "Lumina", "SoftLiving"),
        29,
        12_999,
    ),
    Department(
        "Grădină și exterior",
        (
            "Mașini de tuns iarba",
            "Trimmere",
            "Drujbe",
            "Suflante",
            "Unelte de grădină",
            "Sisteme de irigații",
            "Furtunuri",
            "Ghivece",
            "Mobilier de grădină",
            "Grătare",
            "Foișoare",
            "Piscine",
            "Iluminat exterior",
            "Semințe",
            "Îngrășăminte",
            "Sere",
        ),
        ("GreenPro", "Gardenix", "TerraMax", "AquaGarden", "OutdoorLife", "PlantCare", "GrillMaster", "EcoGrow"),
        19,
        9_999,
    ),
    Department(
        "Modă femei",
        (
            "Rochii",
            "Bluze",
            "Cămăși",
            "Tricouri",
            "Pulovere",
            "Cardigane",
            "Pantaloni",
            "Blugi",
            "Fuste",
            "Sacouri",
            "Geci",
            "Paltoane",
            "Lenjerie",
            "Pijamale",
            "Costume de baie",
            "Accesorii",
        ),
        ("MaisonElle", "Velvet", "UrbanChic", "LunaWear", "Elegance", "ModaViva", "SilkRoad", "DailyStyle"),
        29,
        2_499,
    ),
    Department(
        "Modă bărbați",
        (
            "Tricouri",
            "Cămăși",
            "Pulovere",
            "Hanorace",
            "Pantaloni",
            "Blugi",
            "Sacouri",
            "Costume",
            "Geci",
            "Paltoane",
            "Lenjerie",
            "Pijamale",
            "Șorturi",
            "Echipament business",
            "Curele",
            "Accesorii",
        ),
        ("Gentleman", "UrbanMan", "NorthLine", "ClassicFit", "DenimLab", "StreetCore", "BusinessOne", "DailyMen"),
        29,
        2_999,
    ),
    Department(
        "Încălțăminte",
        (
            "Pantofi sport",
            "Pantofi casual",
            "Pantofi eleganți",
            "Ghete",
            "Bocanci",
            "Cizme",
            "Sandale",
            "Papuci",
            "Espadrile",
            "Mocasini",
            "Pantofi alergare",
            "Pantofi trekking",
            "Încălțăminte copii",
            "Încălțăminte protecție",
            "Branțuri",
            "Îngrijire încălțăminte",
        ),
        ("TrailRun", "UrbanFlex", "Aero", "MountainShield", "StepOne", "ComfortWalk", "StreetSole", "ProTrek"),
        19,
        1_999,
    ),
    Department(
        "Sport și fitness",
        (
            "Fitness",
            "Alergare",
            "Ciclism",
            "Fotbal",
            "Baschet",
            "Tenis",
            "Înot",
            "Drumeții",
            "Camping",
            "Schi",
            "Pescuit",
            "Yoga",
            "Arte marțiale",
            "Sporturi de echipă",
            "Nutriție sportivă",
            "Recuperare",
        ),
        ("PowerSet", "PulseTrack", "ActivePro", "RunMax", "FitCore", "OutdoorX", "TeamSport", "ZenMotion"),
        19,
        8_999,
    ),
    Department(
        "Auto și moto",
        (
            "Anvelope",
            "Jante",
            "Baterii auto",
            "Uleiuri",
            "Filtre",
            "Frâne",
            "Suspensie",
            "Iluminat auto",
            "Camere de bord",
            "Navigație",
            "Compresoare",
            "Scule auto",
            "Îngrijire auto",
            "Accesorii interior",
            "Accesorii moto",
            "Echipamente protecție",
        ),
        ("AutoAir", "DriveGuard", "MotoMax", "RoadPro", "CarTech", "SpeedLine", "AutoCare", "SafeDrive"),
        15,
        12_999,
    ),
    Department(
        "Copii și bebeluși",
        (
            "Cărucioare",
            "Pătuțuri",
            "Scaune auto",
            "Scaune de masă",
            "Mobilier copii",
            "Haine bebeluși",
            "Haine copii",
            "Încălțăminte copii",
            "Jucării educative",
            "Jucării exterior",
            "Construcții",
            "Păpuși",
            "Igienă bebeluși",
            "Hrănire",
            "Siguranță",
            "Rechizite copii",
        ),
        ("BabyGo", "SafeSleep", "BabySecure", "Kiddo", "TinySteps", "PlaySmart", "MiniJoy", "CareBaby"),
        9,
        4_999,
    ),
    Department(
        "Frumusețe",
        (
            "Îngrijire ten",
            "Îngrijire corp",
            "Îngrijire păr",
            "Machiaj ten",
            "Machiaj ochi",
            "Machiaj buze",
            "Parfumuri femei",
            "Parfumuri bărbați",
            "Manichiură",
            "Aparate coafat",
            "Aparate epilat",
            "Bărbierit",
            "Cosmetice naturale",
            "Protecție solară",
            "Seturi cadou",
            "Accesorii beauty",
        ),
        ("HydraSkin", "PureGlow", "BeautyLab", "SilkHair", "ColorMuse", "NatureCare", "Aroma", "DermaPlus"),
        9,
        1_999,
    ),
    Department(
        "Sănătate",
        (
            "Vitamine",
            "Minerale",
            "Suplimente",
            "Imunitate",
            "Digestie",
            "Somn",
            "Tensiometre",
            "Termometre",
            "Glucometre",
            "Nebulizatoare",
            "Ortopedie",
            "Prim ajutor",
            "Igienă orală",
            "Măști protecție",
            "Aparate masaj",
            "Consumabile medicale",
        ),
        ("VitaPlus", "MediCare", "HealthOne", "BioBalance", "WellnessPro", "SafeMed", "OrthoFit", "CarePoint"),
        5,
        3_999,
    ),
    Department(
        "Alimente și băuturi",
        (
            "Cafea",
            "Ceai",
            "Băuturi",
            "Apă",
            "Dulciuri",
            "Gustări",
            "Cereale",
            "Paste",
            "Sosuri",
            "Conserve",
            "Produse bio",
            "Produse fără gluten",
            "Condimente",
            "Uleiuri alimentare",
            "Cadouri gourmet",
            "Ingrediente patiserie",
        ),
        ("Gusto", "BioFarm", "DailyFood", "AromaCafe", "GreenLeaf", "Dolce", "Granoro", "FreshChoice"),
        2,
        999,
    ),
    Department(
        "Animale de companie",
        (
            "Hrană câini",
            "Hrană pisici",
            "Hrană păsări",
            "Hrană pești",
            "Recompense",
            "Litiere",
            "Igienă animale",
            "Paturi animale",
            "Cuști",
            "Acvaristică",
            "Jucării animale",
            "Lese și zgărzi",
            "Transport animale",
            "Îmbrăcăminte animale",
            "Suplimente animale",
            "Dresaj",
        ),
        ("PetJoy", "AnimalCare", "HappyPaws", "AquaPet", "VetLife", "PetHome", "WildTaste", "PawPro"),
        5,
        2_999,
    ),
    Department(
        "Birotică și școală",
        (
            "Caiete",
            "Agende",
            "Instrumente de scris",
            "Hârtie",
            "Dosare",
            "Organizare birou",
            "Calculatoare",
            "Distrugătoare documente",
            "Laminatoare",
            "Table de scris",
            "Ghiozdane",
            "Penare",
            "Rechizite desen",
            "Mobilier birou",
            "Consumabile imprimante",
            "Etichete",
        ),
        ("OfficePro", "SchoolMate", "WriteLine", "PaperMax", "DeskOne", "ColorPen", "SmartOffice", "OrganizeIt"),
        1,
        4_999,
    ),
    Department(
        "Cărți, jocuri și hobby",
        (
            "Literatură",
            "Dezvoltare personală",
            "Business",
            "Știință",
            "Istorie",
            "Cărți pentru copii",
            "Benzi desenate",
            "Manuale",
            "Jocuri de societate",
            "Puzzle",
            "Jocuri educative",
            "Instrumente muzicale",
            "Pictură",
            "Modelism",
            "Colecționabile",
            "Artizanat",
        ),
        ("BookVerse", "EduPress", "GameNight", "CreativeLab", "MusicBox", "ArtHouse", "PuzzleWorld", "HobbyPro"),
        5,
        5_999,
    ),
)

FEATURES = (
    "construcție durabilă",
    "design ergonomic",
    "utilizare intuitivă",
    "consum eficient",
    "materiale atent selecționate",
    "finisaj rezistent",
    "întreținere ușoară",
    "format compact",
    "performanță stabilă",
    "protecție îmbunătățită",
    "confort pentru utilizare zilnică",
    "compatibilitate extinsă",
    "control precis",
    "autonomie ridicată",
    "montaj simplu",
    "depozitare practică",
    "greutate redusă",
    "funcționare silențioasă",
    "rezistență la uzură",
    "garanție comercială",
    "ambalaj reciclabil",
    "componente testate",
    "raport bun calitate-preț",
    "instrucțiuni în limba română",
)

USE_CASES = (
    "acasă",
    "birou",
    "călătorii",
    "utilizare zilnică",
    "activități în aer liber",
    "proiecte personale",
    "familii active",
    "începători",
    "utilizatori avansați",
    "spații mici",
    "cadouri",
    "sezonul rece",
    "sezonul cald",
    "activități profesionale",
    "școală și studiu",
    "timp liber",
)

COLORS = (
    "negru",
    "alb",
    "gri",
    "albastru",
    "roșu",
    "verde",
    "bej",
    "maro",
    "argintiu",
    "auriu",
    "roz",
    "mov",
    "portocaliu",
    "bleumarin",
    "transparent",
    "multicolor",
)

SERIES = (
    "Essential",
    "Plus",
    "Pro",
    "Max",
    "Ultra",
    "Prime",
    "Active",
    "Comfort",
    "Smart",
    "Eco",
    "Urban",
    "Classic",
    "Premium",
    "Compact",
    "Advance",
    "Expert",
)

AUDIENCES = ("unisex", "femei", "bărbați", "copii", "familie")
CSV_FIELDS = (
    "product_id",
    "title",
    "description",
    "category",
    "brand",
    "price_ron",
    "color",
    "audience",
    "in_stock",
)


def categories(limit: int = 256) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for department in DEPARTMENTS:
        if len(department.subcategories) != 16:
            raise RuntimeError(f"{department.name} must define exactly 16 subcategories")
        for subcategory in department.subcategories:
            result.append(
                {
                    "category_id": len(result) + 1,
                    "department": department.name,
                    "subcategory": subcategory,
                    "category": f"{department.name} > {subcategory}",
                    "price_min": department.price_min,
                    "price_max": department.price_max,
                    "brands": department.brands,
                }
            )
    if len(result) != 256:
        raise RuntimeError("the catalog taxonomy must contain exactly 256 categories")
    if not 1 <= limit <= len(result):
        raise ValueError("category count must be between 1 and 256")
    return result[:limit]


def product_values(
    zero_based_index: int,
    taxonomy: list[dict[str, object]],
    seed: int,
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    category_position = zero_based_index % len(taxonomy)
    category = taxonomy[category_position]
    category_id = int(category["category_id"])
    sequence = zero_based_index // len(taxonomy) + 1
    mixed = (
        zero_based_index * 1_103_515_245
        + seed * 12_345
        + category_id * 2_654_435_761
    ) & 0xFFFFFFFF
    brands = category["brands"]
    if not isinstance(brands, tuple):
        raise RuntimeError("category brands are invalid")
    brand = brands[mixed % len(brands)]
    series = SERIES[(mixed >> 4) % len(SERIES)]
    color = COLORS[(mixed >> 8) % len(COLORS)]
    audience = AUDIENCES[(mixed >> 12) % len(AUDIENCES)]
    feature_one = FEATURES[(mixed >> 16) % len(FEATURES)]
    feature_two = FEATURES[(mixed >> 21) % len(FEATURES)]
    use_case = USE_CASES[(mixed >> 25) % len(USE_CASES)]
    price_min = float(category["price_min"])
    price_max = float(category["price_max"])
    fraction = ((mixed % 10_000) / 9_999) ** 1.35
    price = round(price_min + (price_max - price_min) * fraction, 2)
    in_stock = 0 if (mixed + category_id) % 13 == 0 else 1
    product_id = f"P{zero_based_index + 1:07d}"
    subcategory = str(category["subcategory"])
    title = f"{subcategory} {brand} {series} {sequence:04d}"
    description = (
        f"{subcategory} {brand} din seria {series}, recomandat pentru {use_case}. "
        f"Oferă {feature_one} și {feature_two}, variantă {color}. "
        f"Produs testat pentru utilizare constantă și livrare din România."
    )
    csv_row = (
        product_id,
        title,
        description,
        category["category"],
        brand,
        f"{price:.2f}",
        color,
        audience,
        in_stock,
    )
    database_row = (
        product_id,
        title,
        description,
        category_id,
        category["category"],
        brand,
        price,
        color,
        audience,
        in_stock,
    )
    return csv_row, database_row


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_database(path: Path, taxonomy: list[dict[str, object]]) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA cache_size=-131072")
    connection.executescript(
        """
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY,
            department TEXT NOT NULL,
            subcategory TEXT NOT NULL,
            category TEXT NOT NULL UNIQUE,
            price_min REAL NOT NULL,
            price_max REAL NOT NULL
        );

        CREATE TABLE products (
            product_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            category_id INTEGER NOT NULL REFERENCES categories(category_id),
            category TEXT NOT NULL,
            brand TEXT NOT NULL,
            price_ron REAL NOT NULL CHECK(price_ron >= 0),
            color TEXT NOT NULL,
            audience TEXT NOT NULL,
            in_stock INTEGER NOT NULL CHECK(in_stock IN (0, 1))
        );
        """
    )
    connection.executemany(
        """
        INSERT INTO categories (
            category_id, department, subcategory, category, price_min, price_max
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                item["category_id"],
                item["department"],
                item["subcategory"],
                item["category"],
                item["price_min"],
                item["price_max"],
            )
            for item in taxonomy
        ],
    )
    return connection


def validate_outputs(
    csv_path: Path,
    database_path: Path,
    *,
    expected_rows: int,
    expected_categories: int,
) -> dict[str, object]:
    with sqlite3.connect(database_path) as connection:
        product_count, unique_ids, category_count = connection.execute(
            """
            SELECT
                COUNT(*),
                COUNT(DISTINCT product_id),
                COUNT(DISTINCT category_id)
            FROM products
            """
        ).fetchone()
        minimum, maximum = connection.execute(
            """
            SELECT MIN(item_count), MAX(item_count)
            FROM (
                SELECT category_id, COUNT(*) AS item_count
                FROM products
                GROUP BY category_id
            )
            """
        ).fetchone()
        foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()
    csv_count = 0
    csv_categories: Counter[str] = Counter()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise RuntimeError("generated CSV header is invalid")
        for row in reader:
            csv_count += 1
            csv_categories[row["category"]] += 1
    checks = {
        "sqlite_product_count": int(product_count),
        "sqlite_unique_product_ids": int(unique_ids),
        "sqlite_category_count": int(category_count),
        "csv_product_count": csv_count,
        "csv_category_count": len(csv_categories),
        "products_per_category_min": int(minimum),
        "products_per_category_max": int(maximum),
        "foreign_key_errors": len(foreign_key_errors),
    }
    expected = (
        product_count == expected_rows
        and unique_ids == expected_rows
        and category_count == expected_categories
        and csv_count == expected_rows
        and len(csv_categories) == expected_categories
        and not foreign_key_errors
    )
    if not expected:
        raise RuntimeError(f"generated catalog validation failed: {checks}")
    return checks


def generate(
    output_directory: Path,
    *,
    row_count: int,
    category_count: int,
    batch_size: int,
    seed: int,
    force: bool,
) -> dict[str, object]:
    if row_count < 1:
        raise ValueError("row_count must be at least 1")
    if row_count < category_count:
        raise ValueError(
            "row_count must be at least category_count so every category is represented"
        )
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    taxonomy = categories(category_count)
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = "products_1m" if row_count == 1_000_000 else f"products_{row_count}"
    csv_path = output_directory / f"{stem}.csv"
    database_path = output_directory / f"{stem}.sqlite"
    categories_path = output_directory / f"categories_{category_count}.csv"
    summary_path = output_directory / f"{stem}_summary.json"
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    temporary_database = database_path.with_suffix(".sqlite.tmp")
    destinations = (csv_path, database_path, categories_path, summary_path)
    if not force and any(path.exists() for path in destinations):
        existing = ", ".join(str(path) for path in destinations if path.exists())
        raise FileExistsError(f"output already exists; use --force: {existing}")
    for path in (*destinations, temporary_csv, temporary_database):
        if path.exists():
            path.unlink()

    started = perf_counter()
    with categories_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("category_id", "department", "subcategory", "category", "price_min", "price_max")
        )
        for item in taxonomy:
            writer.writerow(
                (
                    item["category_id"],
                    item["department"],
                    item["subcategory"],
                    item["category"],
                    item["price_min"],
                    item["price_max"],
                )
            )

    connection = create_database(temporary_database, taxonomy)
    insert_sql = """
        INSERT INTO products (
            product_id, title, description, category_id, category, brand,
            price_ron, color, audience, in_stock
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    batch: list[tuple[object, ...]] = []
    try:
        with temporary_csv.open("w", encoding="utf-8", newline="", buffering=4 * 1024 * 1024) as handle:
            writer = csv.writer(handle)
            writer.writerow(CSV_FIELDS)
            for index in range(row_count):
                csv_row, database_row = product_values(index, taxonomy, seed)
                writer.writerow(csv_row)
                batch.append(database_row)
                if len(batch) >= batch_size:
                    connection.executemany(insert_sql, batch)
                    batch.clear()
                completed = index + 1
                if completed % 100_000 == 0 or completed == row_count:
                    print(
                        f"[SSFR-DATA] {completed:,}/{row_count:,} products generated",
                        flush=True,
                    )
            if batch:
                connection.executemany(insert_sql, batch)
                batch.clear()
        connection.commit()
        print("[SSFR-DATA] Creating SQLite indexes...", flush=True)
        connection.executescript(
            """
            CREATE INDEX idx_products_category ON products(category_id);
            CREATE INDEX idx_products_category_stock_price
                ON products(category_id, in_stock, price_ron);
            CREATE INDEX idx_products_brand ON products(brand);
            CREATE INDEX idx_products_price ON products(price_ron);
            """
        )
        connection.commit()
    finally:
        connection.close()

    os.replace(temporary_csv, csv_path)
    os.replace(temporary_database, database_path)
    print("[SSFR-DATA] Validating CSV and SQLite row by row...", flush=True)
    checks = validate_outputs(
        csv_path,
        database_path,
        expected_rows=row_count,
        expected_categories=category_count,
    )
    summary = {
        "dataset": "SSFR realistic synthetic product catalog",
        "physical_dataset": True,
        "synthetic_data": True,
        "generated_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "product_count": row_count,
        "category_count": category_count,
        "csv_path": str(csv_path.resolve()),
        "csv_bytes": csv_path.stat().st_size,
        "csv_sha256": file_sha256(csv_path),
        "sqlite_path": str(database_path.resolve()),
        "sqlite_bytes": database_path.stat().st_size,
        "categories_path": str(categories_path.resolve()),
        "generation_and_validation_seconds": perf_counter() - started,
        "validation": checks,
        "warning": (
            "The rows are physically generated and validated, but product names and "
            "descriptions are synthetic rather than transactions from a real merchant."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a physical SSFR product catalog as CSV and SQLite."
    )
    parser.add_argument("--output", type=Path, default=Path("data/generated"))
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--categories", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = generate(
        args.output,
        row_count=args.rows,
        category_count=args.categories,
        batch_size=args.batch_size,
        seed=args.seed,
        force=args.force,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
