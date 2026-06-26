"""quantcore.catalog — registry mechanics + every spec resolves to a real callable."""

from __future__ import annotations

import json

import pytest

from quantcore import catalog
from quantcore.catalog import FactorSpec


def test_every_registered_spec_resolves_to_a_callable():
    # The load-bearing correctness gate: a typo in any module/name fails here.
    specs = catalog.list_factors()
    assert len(specs) >= 25
    for s in specs:
        fn = s.resolve()
        assert callable(fn), f"{s.name} -> {s.module}.{s.name} is not callable"
        assert fn.__name__ == s.name


def test_list_filters_by_category_and_tag():
    micro = catalog.list_factors(category="microstructure")
    assert micro and all(s.category == "microstructure" for s in micro)
    flow = catalog.list_factors(tag="flow")
    assert flow and all("flow" in s.tags for s in flow)
    assert "microstructure" in catalog.categories()


def test_search_matches_name_summary_and_tags():
    assert any(s.name == "roll_spread" for s in catalog.search("spread"))
    assert any(s.name == "amihud_illiquidity" for s in catalog.search("liquidity"))  # via summary/tag
    assert catalog.search("definitely-not-a-factor-xyz") == []


def test_get_returns_spec_and_raises_with_hint():
    assert catalog.get("vpin").category == "microstructure"
    with pytest.raises(KeyError, match="list_factors"):
        catalog.get("nope")


def test_to_json_is_valid_and_round_trips():
    payload = json.loads(catalog.to_json())
    assert isinstance(payload, list) and len(payload) == len(catalog.list_factors())
    row = payload[0]
    assert {"name", "category", "module", "summary", "inputs", "tags", "reference"} <= set(row)


def test_register_rejects_duplicate_unless_overwrite():
    spec = FactorSpec("vpin", "microstructure", "quantcore.features.microstructure", "dup")
    with pytest.raises(ValueError, match="already registered"):
        catalog.register(spec)
    catalog.register(spec, overwrite=True)  # explicit overwrite is allowed
    catalog.register(catalog.get("vpin"), overwrite=True)  # restore canonical spec


def test_catalog_exposed_on_top_level_surface():
    import quantcore

    assert "catalog" in quantcore.__all__
    assert quantcore.catalog is catalog
