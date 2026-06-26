"""quantcore.cli — the quant-catalog console surface (JSON + human output)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quantcore import catalog
from quantcore.cli import factory_main, main, provenance_main, screen_main
from quantcore.discoveries import DiscoveryLedger


def test_list_json_matches_catalog(capsys):
    rc = main(["list", "--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert rc == 0
    assert len(payload) == len(catalog.list_factors())
    assert {"name", "category", "module", "summary"} <= set(payload[0])


def test_list_category_filter(capsys):
    main(["list", "--category", "microstructure", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload and all(r["category"] == "microstructure" for r in payload)


def test_search_finds_factor(capsys):
    rc = main(["search", "spread"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "roll_spread" in out and "corwin_schultz_spread" in out


def test_describe_json(capsys):
    rc = main(["describe", "vpin", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload[0]["name"] == "vpin" and payload[0]["category"] == "microstructure"


def test_describe_unknown_returns_1(capsys):
    rc = main(["describe", "not-a-factor"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "not-a-factor" in err


def test_human_output_is_tabular(capsys):
    main(["list"])
    out = capsys.readouterr().out
    assert "factor(s)" in out
    assert "amihud_illiquidity" in out


def test_missing_subcommand_errors():
    with pytest.raises(SystemExit):
        main([])


def _write_panel_csv(path, seed: int = 1) -> None:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=80)
    assets = [f"A{i}" for i in range(15)]
    rows = []
    for d in dates:
        r = rng.standard_normal(len(assets))
        sig = r * 0.5 + rng.standard_normal(len(assets))
        noise = rng.standard_normal(len(assets))
        for a, rr, ss, nn in zip(assets, r, sig, noise):
            rows.append(
                {
                    "date": d.date().isoformat(),
                    "asset": a,
                    "forward_return": rr,
                    "sig": ss,
                    "noise": nn,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_screen_cli_json(tmp_path, capsys):
    path = tmp_path / "panel.csv"
    _write_panel_csv(path)
    rc = screen_main([str(path), "--json", "--hac-lags", "5"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    by_name = {r["name"]: r for r in payload}
    # The real signal (t~20) is robustly caught and ranks first. (We don't assert the
    # noise factor is non-significant: with 2 factors at FDR 0.10 a pure null passes ~10%
    # of the time by design — FDR bounds *expected* false discoveries; family control is
    # tested in test_screening's pure-nulls case.)
    assert by_name["sig"]["mean_ic"] > 0
    assert by_name["sig"]["significant"] and by_name["sig"]["rank"] == 1


def test_screen_cli_human_table(tmp_path, capsys):
    path = tmp_path / "panel.csv"
    _write_panel_csv(path)
    assert screen_main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "significant at FDR target" in out and "sig" in out


def test_factory_cli_json(tmp_path, capsys):
    path = tmp_path / "panel.csv"
    _write_panel_csv(path)
    rc = factory_main([str(path), "--json", "--hac-lags", "5", "--dsr-threshold", "0.6"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    by_name = {r["name"]: r for r in payload}
    sig = by_name["sig"]
    assert {"name", "deflated_sharpe", "passed", "reason", "rank"} <= set(sig)
    # The real signal is FDR-significant, has positive IC, and ranks first.
    assert sig["ic_significant"] and sig["mean_ic"] > 0 and sig["rank"] == 1


def test_factory_cli_human_table(tmp_path, capsys):
    path = tmp_path / "panel.csv"
    _write_panel_csv(path)
    assert factory_main([str(path), "--dsr-threshold", "0.6"]) == 0
    out = capsys.readouterr().out
    assert "survived both gates" in out and "sig" in out


def test_factory_cli_register_writes_ledger(tmp_path, capsys):
    path = tmp_path / "panel.csv"
    _write_panel_csv(path)
    ledger_path = tmp_path / "discoveries.json"
    rc = factory_main([str(path), "--dsr-threshold", "0.5", "--register", str(ledger_path)])
    assert rc == 0
    err = capsys.readouterr().err
    assert "registered" in err and str(ledger_path) in err
    assert ledger_path.exists()
    payload = json.loads(ledger_path.read_text())
    # v2 ledger: a manifest is recorded for the run even if nothing survived.
    assert payload["schema_version"] == 2 and len(payload["manifests"]) == 1
    # Whatever survived is a valid Discovery record sourced from this panel (vacuous if none did).
    for e in payload["discoveries"]:
        assert e["name"] in {"sig", "noise"} and e["source"] == str(path)
        assert {"in_catalog", "deflated_sharpe", "screen_params", "run_id"} <= set(e)


def test_provenance_cli_verify_and_list(tmp_path, capsys):
    panel = tmp_path / "panel.csv"
    _write_panel_csv(panel)
    ledger = tmp_path / "ledger.json"
    factory_main([str(panel), "--dsr-threshold", "0.5", "--register", str(ledger)])
    capsys.readouterr()  # drain the factory output
    run_id = DiscoveryLedger.load(ledger).manifests[0].run_id

    assert provenance_main(["list", str(ledger), "--json"]) == 0
    manifests = json.loads(capsys.readouterr().out)
    assert len(manifests) == 1 and manifests[0]["run_id"] == run_id

    # Verify against the SAME panel: data, code, and run_id all re-derive identically.
    # (reproducible is True only on a clean tree, so rc may be 0 or 1 in a dirty dev checkout.)
    rc = provenance_main(["verify", str(ledger), run_id, str(panel), "--json"])
    check = json.loads(capsys.readouterr().out)
    assert check["data_matches"] and check["code_matches"] and check["run_id_matches"]
    assert rc in (0, 1)


def test_provenance_cli_detects_data_drift(tmp_path, capsys):
    panel = tmp_path / "panel.csv"
    _write_panel_csv(panel, seed=1)
    ledger = tmp_path / "ledger.json"
    factory_main([str(panel), "--dsr-threshold", "0.5", "--register", str(ledger)])
    capsys.readouterr()
    run_id = DiscoveryLedger.load(ledger).manifests[0].run_id

    other = tmp_path / "other.csv"
    _write_panel_csv(other, seed=99)  # different data -> different fingerprint
    rc = provenance_main(["verify", str(ledger), run_id, str(other), "--json"])
    check = json.loads(capsys.readouterr().out)
    assert not check["data_matches"] and not check["reproducible"] and rc == 1


def test_provenance_cli_unknown_run_id_returns_1(tmp_path, capsys):
    panel = tmp_path / "panel.csv"
    _write_panel_csv(panel)
    ledger = tmp_path / "ledger.json"
    factory_main([str(panel), "--register", str(ledger)])
    capsys.readouterr()
    assert provenance_main(["show", str(ledger), "deadbeefdeadbeef"]) == 1
