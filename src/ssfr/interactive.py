"""Friendly interactive terminal search for a persistent SSFR catalog."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from time import perf_counter

from .catalog import CatalogIndex
from .types import CatalogSearchResult

InputFunction = Callable[[str], str]
OutputFunction = Callable[[str], None]


@dataclass
class InteractiveState:
    """Mutable settings controlled by slash commands in an interactive session."""

    all_results: bool = True
    top_k: int = 10
    probe_shards: int | None = None
    page_size: int = 10
    category: str | None = None
    brand: str | None = None
    color: str | None = None
    audience: str | None = None
    price_min: float | None = None
    price_max: float | None = None
    in_stock_only: bool = False


HELP_TEXT = """Comenzi disponibile:
  /toate                 returnează toate produsele compatibile din toate shardurile
  /top 5                 returnează numai primele 5 rezultate
  /sharduri 4            caută top-ul în 4 sharduri
  /sharduri toate        caută top-ul în toate shardurile
  /pagina 10             afișează 10 rezultate pe pagină
  /categorie Electronice aplică filtrul de categorie
  /brand DevPro          aplică filtrul de brand
  /culoare negru         aplică filtrul de culoare
  /audienta unisex       aplică filtrul de audiență
  /pret-min 100          setează prețul minim
  /pret-max 1000         setează prețul maxim
  /stoc da               afișează numai produse în stoc
  /fara-filtre           elimină toate filtrele
  /filtre                afișează filtrele și modul curent
  /ajutor                afișează acest ajutor
  /iesire                închide programul

Pentru a elimina un singur filtru textual sau de preț, folosește '-' ca valoare."""


def _format_price(value: float | None) -> str:
    return "preț indisponibil" if value is None else f"{value:,.2f} RON"


def _format_status(state: InteractiveState, catalog: CatalogIndex) -> str:
    mode = (
        f"TOATE rezultatele, {catalog.router.shard_count}/{catalog.router.shard_count} sharduri"
        if state.all_results
        else (
            f"TOP {state.top_k}, "
            f"{state.probe_shards or catalog.router.config.probe_shards}/"
            f"{catalog.router.shard_count} sharduri"
        )
    )
    filters = [
        f"categorie={state.category}" if state.category else None,
        f"brand={state.brand}" if state.brand else None,
        f"culoare={state.color}" if state.color else None,
        f"audiență={state.audience}" if state.audience else None,
        f"preț minim={state.price_min:g}" if state.price_min is not None else None,
        f"preț maxim={state.price_max:g}" if state.price_max is not None else None,
        "doar în stoc" if state.in_stock_only else None,
    ]
    active = ", ".join(item for item in filters if item) or "niciunul"
    return f"Mod: {mode}\nFiltre: {active}\nRezultate/pagină: {state.page_size}"


def _format_page(
    result: CatalogSearchResult,
    *,
    page: int,
    page_size: int,
    perceived_latency_ms: float,
) -> str:
    total = len(result.products)
    page_count = max(1, ceil(total / page_size))
    start = page * page_size
    end = min(total, start + page_size)
    route = result.search.route
    lines = [
        "",
        (
            f"Rezultate {start + 1 if total else 0}-{end} din {total} | "
            f"pagina {page + 1}/{page_count}"
        ),
        (
            f"Sharduri {result.search.shards_accessed}/{route.approximate_scores.size} | "
            f"Recall {result.recall_at_k:.3f} | "
            f"Motor {result.search.total_latency_ms:.3f} ms | "
            f"Perceput {perceived_latency_ms:.3f} ms"
        ),
        "-" * 72,
    ]
    if not total:
        lines.append("Nu există produse care să corespundă filtrelor.")
        return "\n".join(lines)
    for rank in range(start, end):
        product = result.products[rank]
        score = float(result.scores[rank])
        details = " | ".join(
            value
            for value in (
                product.category,
                product.brand,
                _format_price(product.price_ron),
                "în stoc" if product.in_stock else "stoc epuizat",
            )
            if value
        )
        lines.extend(
            [
                f"{rank + 1}. {product.title} [{product.product_id}] — scor {score:.4f}",
                f"   {details}",
                f"   {product.description}",
            ]
        )
    return "\n".join(lines)


def _read(
    input_fn: InputFunction,
    prompt: str,
) -> str | None:
    try:
        return input_fn(prompt)
    except (EOFError, KeyboardInterrupt):
        return None


def _show_paginated(
    result: CatalogSearchResult,
    state: InteractiveState,
    *,
    input_fn: InputFunction,
    output_fn: OutputFunction,
    perceived_latency_ms: float,
) -> bool:
    total = len(result.products)
    page_count = max(1, ceil(total / state.page_size))
    page = 0
    while True:
        output_fn(
            _format_page(
                result,
                page=page,
                page_size=state.page_size,
                perceived_latency_ms=perceived_latency_ms,
            )
        )
        if page_count <= 1:
            return True
        action = _read(
            input_fn,
            "\n[Enter/următoarea, p=precedenta, a=toate, n=căutare nouă, x=ieșire] > ",
        )
        if action is None:
            return False
        command = action.strip().casefold()
        if command in {"x", "exit", "iesire"}:
            return False
        if command in {"n", "noua", "nouă", "q"}:
            return True
        if command in {"p", "precedenta", "precedentă"}:
            page = max(0, page - 1)
            continue
        if command in {"a", "toate"}:
            for remaining_page in range(page + 1, page_count):
                output_fn(
                    _format_page(
                        result,
                        page=remaining_page,
                        page_size=state.page_size,
                        perceived_latency_ms=perceived_latency_ms,
                    )
                )
            return True
        if command in {"", "u", "urmatoarea", "următoarea"}:
            if page + 1 >= page_count:
                return True
            page += 1
            continue
        output_fn("Comandă de navigare necunoscută.")


def _optional_text(value: str) -> str | None:
    cleaned = value.strip()
    return None if cleaned.casefold() in {"", "-", "nu", "off"} else cleaned


def _optional_number(value: str) -> float | None:
    cleaned = value.strip()
    if cleaned.casefold() in {"", "-", "nu", "off"}:
        return None
    return float(cleaned.replace(",", "."))


def _handle_command(
    raw: str,
    state: InteractiveState,
    catalog: CatalogIndex,
    output_fn: OutputFunction,
) -> bool:
    command, _, value = raw[1:].strip().partition(" ")
    command = command.casefold()
    value = value.strip()
    if command in {"iesire", "exit", "quit", "q"}:
        return False
    if command in {"ajutor", "help", "h", "?"}:
        output_fn(HELP_TEXT)
        return True
    if command in {"toate", "all"}:
        state.all_results = True
        output_fn("Mod complet activat: toate rezultatele din toate shardurile.")
        return True
    if command == "top":
        try:
            count = int(value)
            if count < 1:
                raise ValueError
        except ValueError:
            output_fn("Folosește /top urmat de un număr pozitiv, de exemplu /top 5.")
            return True
        state.top_k = count
        state.all_results = False
        output_fn(f"Mod rapid activat: primele {count} rezultate.")
        return True
    if command in {"sharduri", "shards"}:
        if value.casefold() in {"toate", "all"}:
            state.probe_shards = catalog.router.shard_count
            output_fn(f"Vor fi căutate toate cele {catalog.router.shard_count} sharduri.")
            return True
        try:
            count = int(value)
            if not 1 <= count <= catalog.router.shard_count:
                raise ValueError
        except ValueError:
            output_fn(
                f"Alege un număr între 1 și {catalog.router.shard_count} sau "
                "folosește /sharduri toate."
            )
            return True
        state.probe_shards = count
        state.all_results = False
        output_fn(f"Mod rapid activat: se caută în {count} sharduri.")
        return True
    if command in {"pagina", "page"}:
        try:
            count = int(value)
            if count < 1:
                raise ValueError
        except ValueError:
            output_fn("Folosește /pagina urmat de un număr pozitiv.")
            return True
        state.page_size = count
        output_fn(f"Vor fi afișate {count} rezultate pe pagină.")
        return True
    text_filters = {
        "categorie": "category",
        "category": "category",
        "brand": "brand",
        "culoare": "color",
        "color": "color",
        "audienta": "audience",
        "audiență": "audience",
        "audience": "audience",
    }
    if command in text_filters:
        setattr(state, text_filters[command], _optional_text(value))
        output_fn(_format_status(state, catalog))
        return True
    if command in {"pret-min", "preț-min", "price-min", "pret-max", "preț-max", "price-max"}:
        attribute = "price_min" if command.endswith("min") else "price_max"
        try:
            setattr(state, attribute, _optional_number(value))
        except ValueError:
            output_fn("Prețul trebuie să fie un număr sau '-'.")
            return True
        output_fn(_format_status(state, catalog))
        return True
    if command in {"stoc", "stock"}:
        normalized = value.casefold()
        if normalized not in {"da", "nu", "on", "off", "true", "false", "1", "0"}:
            output_fn("Folosește /stoc da sau /stoc nu.")
            return True
        state.in_stock_only = normalized in {"da", "on", "true", "1"}
        output_fn(_format_status(state, catalog))
        return True
    if command in {"fara-filtre", "fără-filtre", "clear-filters"}:
        state.category = None
        state.brand = None
        state.color = None
        state.audience = None
        state.price_min = None
        state.price_max = None
        state.in_stock_only = False
        output_fn("Toate filtrele au fost eliminate.")
        return True
    if command in {"filtre", "filters", "status"}:
        output_fn(_format_status(state, catalog))
        return True
    output_fn("Comandă necunoscută. Folosește /ajutor.")
    return True


def run_interactive(
    catalog: CatalogIndex,
    *,
    state: InteractiveState | None = None,
    input_fn: InputFunction = input,
    output_fn: OutputFunction = print,
    report_path: str | Path | None = None,
    evaluate: bool = False,
) -> int:
    """Run a reusable interactive search session over an already loaded catalog."""

    current = state or InteractiveState()
    if current.top_k < 1:
        raise ValueError("top_k must be at least 1")
    if current.page_size < 1:
        raise ValueError("page_size must be at least 1")
    if current.probe_shards is not None and not (
        1 <= current.probe_shards <= catalog.router.shard_count
    ):
        raise ValueError("probe_shards is outside the catalog shard range")

    output_fn("=" * 72)
    output_fn("SSFR — Căutare interactivă")
    output_fn(
        f"Index: {catalog.path.resolve()}\n"
        f"Produse: {len(catalog.products)} | Sharduri: {catalog.router.shard_count}"
    )
    output_fn(_format_status(current, catalog))
    if catalog.manifest.get("embedding_provider") in {"hash", "fast-hash"}:
        output_fn(
            "Notă: indexul folosește embeddingul demonstrativ hash; pentru relevanță "
            "semantică mai bună reconstruiește-l cu sentence-transformers."
        )
    output_fn("Scrie ce cauți. Pentru comenzi folosește /ajutor.")
    output_fn("=" * 72)

    while True:
        raw = _read(input_fn, "\nCaută > ")
        if raw is None:
            output_fn("\nProgram închis.")
            return 0
        query = raw.strip()
        if not query:
            continue
        if query.startswith("/"):
            if not _handle_command(query, current, catalog, output_fn):
                output_fn("Program închis.")
                return 0
            continue
        output_fn("Se caută...")
        perceived_started = perf_counter()
        try:
            result = catalog.search_text(
                query,
                top_k=current.top_k,
                probe_shards=current.probe_shards,
                all_results=current.all_results,
                evaluate=evaluate,
                filter_strategy="pre",
                report_path=report_path,
                category=current.category,
                brand=current.brand,
                color=current.color,
                audience=current.audience,
                price_min=current.price_min,
                price_max=current.price_max,
                in_stock_only=current.in_stock_only,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            output_fn(f"Eroare la căutare: {exc}")
            continue
        perceived_latency_ms = (perf_counter() - perceived_started) * 1000.0
        if not _show_paginated(
            result,
            current,
            input_fn=input_fn,
            output_fn=output_fn,
            perceived_latency_ms=perceived_latency_ms,
        ):
            output_fn("Program închis.")
            return 0
