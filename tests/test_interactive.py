from __future__ import annotations

from ssfr.catalog import CatalogIndex
from ssfr.cli import build_parser
from ssfr.interactive import InteractiveState, run_interactive


def _input_sequence(*values: str):
    responses = iter(values)
    return lambda _prompt: next(responses)


def test_interactive_defaults_to_all_results(tmp_path, monkeypatch) -> None:
    catalog, _ = CatalogIndex.build(
        "data/products.csv",
        tmp_path / "catalog",
        shard_count=4,
        bands=(1, 2),
        probe_shards=1,
        embedding_provider="hash",
        embedding_dimension=64,
        local_index_backend="exact",
    )

    def fail_if_oracle_runs(*_args, **_kwargs):
        raise AssertionError("interactive complete search must not rescan the oracle")

    monkeypatch.setattr(
        "ssfr.catalog.exact_global_search",
        fail_if_oracle_runs,
    )
    output: list[str] = []
    status = run_interactive(
        catalog,
        state=InteractiveState(page_size=50),
        input_fn=_input_sequence("laptop pentru programare", "/iesire"),
        output_fn=output.append,
        report_path=None,
    )
    rendered = "\n".join(output)
    assert status == 0
    assert "TOATE rezultatele" in rendered
    assert "Rezultate 1-20 din 20" in rendered
    assert "Sharduri 4/4" in rendered
    assert "Laptop DevPro 16" in rendered
    assert "Recall 1.000" in rendered


def test_interactive_top_command_switches_to_limited_mode(tmp_path) -> None:
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
    output: list[str] = []
    run_interactive(
        catalog,
        state=InteractiveState(page_size=50, probe_shards=4),
        input_fn=_input_sequence("/top 3", "telefon cu baterie mare", "/iesire"),
        output_fn=output.append,
        report_path=None,
    )
    rendered = "\n".join(output)
    assert "Mod rapid activat: primele 3 rezultate." in rendered
    assert "Rezultate 1-3 din 3" in rendered


def test_cli_exposes_interactive_and_all_results_options() -> None:
    parser = build_parser()
    interactive = parser.parse_args(["interactive"])
    assert interactive.index == "artifacts/products"
    assert interactive.top_only is False
    assert interactive.report is None
    search = parser.parse_args(
        [
            "search",
            "--index",
            "artifacts/products",
            "--query",
            "laptop",
            "--all-results",
        ]
    )
    assert search.all_results is True
