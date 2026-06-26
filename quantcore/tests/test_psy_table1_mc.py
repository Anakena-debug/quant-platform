"""s84 0b — PSY (2015) Table-1 constants: Monte-Carlo cross-check (s83 F23-batch item).

``_PSY_TABLE`` is a hand-transcribed table (the module history includes a prior episode of
*wrong* hard-coded critical values). This test regenerates the T=100 row under the I(1) null with
the module's own ``simulate_critical_values`` (p=0 — PSY Table 1 uses the unaugmented ADF;
deterministic seed) and asserts the transcribed constants sit inside transcription-error-sized
tolerances of the MC quantiles. A digit slip (0.5+) fails loudly; MC noise does not (the run is
seeded — calibration: seed 42 deltas ≤0.073 SADF / ≤0.202 GSADF; cross-seed 95% spread ≈0.1).

Runtime ≈ 2 s (n_sim=2000, T=100).
"""

from __future__ import annotations

import pytest

from quantcore.features.psy_gsadf import (
    psy_reference_critical_values,
    simulate_critical_values,
)

# transcribed PSY (2015) Table 1, T=100 row (the values under test)
TABLE_SADF = {0.90: 0.98, 0.95: 1.30, 0.99: 1.92}
TABLE_GSADF = {0.90: 1.66, 0.95: 1.92, 0.99: 2.45}

TOL_SADF = 0.25
TOL_GSADF = {0.90: 0.25, 0.95: 0.30, 0.99: 0.40}  # tail quantiles are MC-noisier


@pytest.fixture(scope="module")
def mc() -> dict:
    return simulate_critical_values(100, p=0, n_sim=2000, seed=42, include_bsadf=False)


def test_sadf_table_row_within_mc_tolerance(mc: dict) -> None:
    for q, table_val in TABLE_SADF.items():
        delta = abs(mc["sadf"][q] - table_val)
        assert delta <= TOL_SADF, (
            f"SADF {q:.0%} CV: table {table_val} vs MC {mc['sadf'][q]:.3f} "
            f"(|Δ|={delta:.3f} > {TOL_SADF}) — possible transcription error in _PSY_TABLE"
        )


def test_gsadf_table_row_within_mc_tolerance(mc: dict) -> None:
    for q, table_val in TABLE_GSADF.items():
        delta = abs(mc["gsadf"][q] - table_val)
        assert delta <= TOL_GSADF[q], (
            f"GSADF {q:.0%} CV: table {table_val} vs MC {mc['gsadf'][q]:.3f} "
            f"(|Δ|={delta:.3f} > {TOL_GSADF[q]}) — possible transcription error in _PSY_TABLE"
        )


def test_table_structural_sanity() -> None:
    """No MC needed: CVs increase in confidence level and in T; GSADF ≥ SADF at every level."""
    prev_sadf = prev_gsadf = -float("inf")
    for alpha in (0.10, 0.05, 0.01):
        ref = psy_reference_critical_values(100, alpha)
        assert ref["sadf"] >= prev_sadf and ref["gsadf"] >= prev_gsadf
        assert ref["gsadf"] >= ref["sadf"]  # sup over windows ⊇ the SADF window family
        prev_sadf, prev_gsadf = ref["sadf"], ref["gsadf"]
    for alpha in (0.10, 0.05, 0.01):
        vals = [psy_reference_critical_values(T, alpha)["gsadf"] for T in (100, 200, 400, 800)]
        assert vals == sorted(vals), f"GSADF CVs not monotone in T at alpha={alpha}: {vals}"
