"""quantcore.mcp_server — Model Context Protocol server exposing quantcore to agents.

Opt-in: install the ``mcp`` extra (``uv sync --extra mcp`` / ``pip install
quantcore[mcp]``). Runs as the ``quant-mcp`` console script over stdio, exposing the
factor/feature catalog as typed MCP tools an agent calls directly:

    list_factors(category?, tag?) -> list of factor-spec dicts
    search_factors(query)         -> matching factor-spec dicts
    describe_factor(name)         -> one factor-spec dict
    list_categories()             -> distinct categories
    catalog_json()                -> the whole catalog as a JSON string
    screen_factor_file(path, ...) -> factors ranked by cross-sectional IC (HAC t, BH-FDR)
    run_factory_file(path, ...)   -> the full screen->deflate->survivors verdicts
    run_factory_and_register(...) -> run the factory and append survivors to a JSON ledger
    list_discoveries(ledger_path) -> the validated factors recorded in a ledger
    list_manifests(ledger_path)   -> the provenance manifests recorded in a ledger
    verify_discovery(...)         -> re-derive a manifest against a panel; is it reproducible?

Tools return JSON-serializable structures (the same ``FactorSpec`` / result dicts the
``quant-catalog`` / ``quant-screen`` / ``quant-factory`` CLIs emit). As data-query /
backtest capabilities land, they register here as additional tools — this is the
agentic-native control surface.

The ``mcp`` import is guarded for the type-checker (``reportMissingImports``) because
``mcp`` is an optional extra, not a core dependency; importing this module without the
extra raises ``ModuleNotFoundError`` by design.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP  # pyright: ignore[reportMissingImports]

from quantcore import catalog, discoveries, factory, provenance, screening

mcp = FastMCP("quantcore")


@mcp.tool()
def list_factors(category: str | None = None, tag: str | None = None) -> list[dict[str, object]]:
    """List catalog factors/features, optionally filtered by category and/or tag."""
    return [s.to_dict() for s in catalog.list_factors(category=category, tag=tag)]


@mcp.tool()
def search_factors(query: str) -> list[dict[str, object]]:
    """Substring search over factor name, summary, and tags."""
    return [s.to_dict() for s in catalog.search(query)]


@mcp.tool()
def describe_factor(name: str) -> dict[str, object]:
    """Return the full spec for one factor by name (errors if unknown)."""
    return catalog.get(name).to_dict()


@mcp.tool()
def list_categories() -> list[str]:
    """List the distinct factor categories in the catalog."""
    return catalog.categories()


@mcp.tool()
def catalog_json() -> str:
    """Return the entire catalog as a JSON string."""
    return catalog.to_json()


@mcp.tool()
def screen_factor_file(
    path: str,
    return_col: str = "forward_return",
    date_col: str = "date",
    asset_col: str = "asset",
    hac_lags: int = 5,
    fdr: float = 0.10,
) -> list[dict[str, object]]:
    """Screen a long-format parquet/CSV of factor panels by cross-sectional IC.

    The file has one row per (date, asset) with a column per factor plus a forward-return
    column. Returns factors ranked by |IC| with a Newey-West HAC t-stat and BH-FDR
    significance — the agent-callable discovery screen over the catalog's factors.
    """
    frame = screening.read_panel_frame(path)
    results = screening.screen_long_frame(
        frame,
        date_col=date_col,
        asset_col=asset_col,
        return_col=return_col,
        hac_lags=hac_lags,
        fdr=fdr,
    )
    return [r.to_dict() for r in results]


@mcp.tool()
def run_factory_file(
    path: str,
    return_col: str = "forward_return",
    date_col: str = "date",
    asset_col: str = "asset",
    hac_lags: int = 5,
    fdr: float = 0.10,
    dsr_threshold: float = 0.95,
) -> list[dict[str, object]]:
    """Run the full alpha-factory loop on a long-format parquet/CSV of factor panels.

    Screens every factor column by cross-sectional IC (HAC t-stat, BH-FDR), then scores each
    factor's dollar-neutral long-short returns by the Deflated Sharpe (corrected for the number
    of candidates). Returns ranked verdicts; a factor ``passed`` only if its IC is
    FDR-significant AND its Deflated Sharpe clears ``dsr_threshold`` — the agent-callable
    end-to-end discovery gate.
    """
    frame = screening.read_panel_frame(path)
    verdicts = factory.run_factory_frame(
        frame,
        date_col=date_col,
        asset_col=asset_col,
        return_col=return_col,
        hac_lags=hac_lags,
        fdr=fdr,
        dsr_threshold=dsr_threshold,
    )
    return [v.to_dict() for v in verdicts]


@mcp.tool()
def run_factory_and_register(
    path: str,
    ledger_path: str,
    return_col: str = "forward_return",
    date_col: str = "date",
    asset_col: str = "asset",
    hac_lags: int = 5,
    fdr: float = 0.10,
    dsr_threshold: float = 0.95,
) -> dict[str, object]:
    """Run the factory on a panel file and append the survivors to a JSON discoveries ledger.

    Returns the verdicts, the survivors just registered, and the ledger's new size. The ledger
    (created if absent) is the desk's accumulating record of validated factors, so an agent can
    screen panel after panel and build up a research memory instead of re-discovering.
    """
    frame = screening.read_panel_frame(path)
    params: dict[str, object] = {"hac_lags": hac_lags, "fdr": fdr, "dsr_threshold": dsr_threshold}
    verdicts = factory.run_factory_frame(
        frame,
        date_col=date_col,
        asset_col=asset_col,
        return_col=return_col,
        hac_lags=hac_lags,
        fdr=fdr,
        dsr_threshold=dsr_threshold,
    )
    ledger = discoveries.DiscoveryLedger.load(ledger_path)
    manifest = provenance.build_manifest(frame, params=params, source=path)
    added = ledger.register_survivors(
        verdicts, source=path, screen_params=params, manifest=manifest
    )
    ledger.save(ledger_path)
    return {
        "run_id": manifest.run_id,
        "registered": [d.to_dict() for d in added],
        "ledger_size": len(ledger),
        "verdicts": [v.to_dict() for v in verdicts],
    }


@mcp.tool()
def list_discoveries(ledger_path: str) -> list[dict[str, object]]:
    """List the validated factors recorded in a JSON discoveries ledger (ranked by Deflated Sharpe)."""
    return [d.to_dict() for d in discoveries.DiscoveryLedger.load(ledger_path).records]


@mcp.tool()
def list_manifests(ledger_path: str) -> list[dict[str, object]]:
    """List the provenance manifests (data/code/params lineage) recorded in a discoveries ledger."""
    return [m.to_dict() for m in discoveries.DiscoveryLedger.load(ledger_path).manifests]


@mcp.tool()
def verify_discovery(ledger_path: str, run_id: str, panel_path: str) -> dict[str, object]:
    """Re-derive a recorded manifest against a panel file; report whether the finding reproduces.

    Loads the manifest for ``run_id``, re-fingerprints ``panel_path``, re-captures the code
    version, and returns a check with ``reproducible`` true only if data, code commit, and the
    recomputed run_id all match a clean working tree.
    """
    ledger = discoveries.DiscoveryLedger.load(ledger_path)
    manifest = ledger.get_manifest(run_id)
    frame = screening.read_panel_frame(panel_path)
    return provenance.verify_manifest(manifest, frame).to_dict()


def main() -> None:
    """Run the MCP server over stdio (the ``quant-mcp`` console script)."""
    mcp.run()


if __name__ == "__main__":
    main()
