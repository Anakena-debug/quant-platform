"""quantcore.mcp_server — MCP tool surface (skipped unless the ``mcp`` extra is installed)."""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mcp", reason="MCP server is an opt-in extra (install quantcore[mcp])")

from quantcore import catalog  # noqa: E402  (after importorskip by design)
from quantcore.mcp_server import (  # noqa: E402
    catalog_json,
    describe_factor,
    list_categories,
    list_discoveries,
    list_factors,
    list_manifests,
    mcp,
    run_factory_and_register,
    run_factory_file,
    screen_factor_file,
    search_factors,
    verify_discovery,
)

_EXPECTED_TOOLS = {
    "list_factors",
    "search_factors",
    "describe_factor",
    "list_categories",
    "catalog_json",
    "screen_factor_file",
    "run_factory_file",
    "run_factory_and_register",
    "list_discoveries",
    "list_manifests",
    "verify_discovery",
}


def test_all_tools_registered():
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert _EXPECTED_TOOLS <= names


def test_list_factors_tool_matches_catalog():
    assert len(list_factors()) == len(catalog.list_factors())
    micro = list_factors(category="microstructure")
    assert micro and all(r["category"] == "microstructure" for r in micro)


def test_search_and_describe_tools():
    assert any(r["name"] == "roll_spread" for r in search_factors("spread"))
    assert describe_factor("vpin")["category"] == "microstructure"


def test_catalog_json_tool_is_valid_json():
    payload = json.loads(catalog_json())
    assert len(payload) == len(catalog.list_factors())


def test_list_categories_tool():
    assert "entropy" in list_categories()


def test_screen_factor_file_tool(tmp_path):
    rng = np.random.default_rng(2)
    dates = pd.bdate_range("2021-01-01", periods=70)
    assets = [f"A{i}" for i in range(12)]
    rows = []
    for d in dates:
        r = rng.standard_normal(len(assets))
        sig = r * 0.5 + rng.standard_normal(len(assets))
        for a, rr, ss in zip(assets, r, sig):
            rows.append({"date": d.isoformat(), "asset": a, "forward_return": rr, "sig": ss})
    path = tmp_path / "panel.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    out = screen_factor_file(str(path), hac_lags=5)
    assert isinstance(out, list) and out
    sig = next(r for r in out if r["name"] == "sig")
    assert sig["mean_ic"] > 0 and sig["significant"] and sig["rank"] == 1


def test_run_factory_file_tool(tmp_path):
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2021-01-01", periods=160)
    assets = [f"A{i}" for i in range(15)]
    rows = []
    for d in dates:
        r = rng.standard_normal(len(assets))
        sig = r * 0.5 + rng.standard_normal(len(assets))
        noise0 = rng.standard_normal(len(assets))
        noise1 = rng.standard_normal(len(assets))
        for j, a in enumerate(assets):
            rows.append(
                {
                    "date": d.isoformat(),
                    "asset": a,
                    "forward_return": r[j],
                    "sig": sig[j],
                    "noise0": noise0[j],
                    "noise1": noise1[j],
                }
            )
    path = tmp_path / "panel.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    out = run_factory_file(str(path), hac_lags=5, dsr_threshold=0.6)
    assert isinstance(out, list) and out
    keys = {"name", "mean_ic", "deflated_sharpe", "passed", "reason", "rank"}
    assert keys <= set(out[0])
    sig = next(r for r in out if r["name"] == "sig")
    # The real signal is FDR-significant, scores a positive Deflated Sharpe, and ranks first.
    assert sig["ic_significant"] and sig["mean_ic"] > 0 and sig["rank"] == 1


def test_run_factory_and_register_then_list(tmp_path):
    rng = np.random.default_rng(8)
    dates = pd.bdate_range("2021-01-01", periods=140)
    assets = [f"A{i}" for i in range(15)]
    rows = []
    for d in dates:
        r = rng.standard_normal(len(assets))
        sig = r * 0.5 + rng.standard_normal(len(assets))
        noise = rng.standard_normal(len(assets))
        for j, a in enumerate(assets):
            rows.append(
                {
                    "date": d.isoformat(),
                    "asset": a,
                    "forward_return": r[j],
                    "sig": sig[j],
                    "noise": noise[j],
                }
            )
    panel = tmp_path / "panel.csv"
    pd.DataFrame(rows).to_csv(panel, index=False)
    ledger = tmp_path / "ledger.json"

    out = run_factory_and_register(str(panel), str(ledger), hac_lags=5, dsr_threshold=0.5)
    assert {"run_id", "registered", "ledger_size", "verdicts"} <= set(out)
    assert ledger.exists()
    listed = list_discoveries(str(ledger))
    # list_discoveries reflects exactly what the ledger holds after registration.
    assert isinstance(listed, list) and len(listed) == out["ledger_size"]
    assert all(e["source"] == str(panel) for e in listed)

    # A manifest is always recorded (even with zero survivors), and it re-derives against the panel.
    manifests = list_manifests(str(ledger))
    assert len(manifests) == 1 and manifests[0]["run_id"] == out["run_id"]
    check = verify_discovery(str(ledger), out["run_id"], str(panel))
    assert check["data_matches"] and check["code_matches"] and check["run_id_matches"]
