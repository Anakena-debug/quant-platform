"""quantcore.cli — shell- and agent-callable entry point over the catalog.

Installed as the ``quant-catalog`` console script (see ``[project.scripts]``). Every
command supports ``--json`` for machine / agent consumption; without it, output is a
human-readable table.

    quant-catalog list [--category C] [--tag T] [--json]
    quant-catalog search QUERY [--json]
    quant-catalog describe NAME [--json]

This is the first agent-callable surface for quantcore — the same capabilities the
MCP server will expose as typed tools.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from quantcore import catalog, discoveries, factory, provenance, screening
from quantcore.catalog import FactorSpec


def _emit(specs: list[FactorSpec], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps([s.to_dict() for s in specs], indent=2))
        return
    for s in specs:
        ref = f"  [{s.reference}]" if s.reference else ""
        print(f"{s.category:<17} {s.name:<28} {s.summary}{ref}")
    print(f"\n{len(specs)} factor(s)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quant-catalog",
        description="Browse the quantcore factor/feature catalog.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="list factors (optionally filtered)")
    p_list.add_argument("--category", help="restrict to a category")
    p_list.add_argument("--tag", help="restrict to a tag")
    p_list.add_argument("--json", action="store_true", help="emit JSON")

    p_search = sub.add_parser("search", help="substring search over name/summary/tags")
    p_search.add_argument("query")
    p_search.add_argument("--json", action="store_true", help="emit JSON")

    p_desc = sub.add_parser("describe", help="describe one factor")
    p_desc.add_argument("name")
    p_desc.add_argument("--json", action="store_true", help="emit JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "list":
        _emit(catalog.list_factors(category=args.category, tag=args.tag), as_json=args.json)
    elif args.cmd == "search":
        _emit(catalog.search(args.query), as_json=args.json)
    elif args.cmd == "describe":
        try:
            spec = catalog.get(args.name)
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        _emit([spec], as_json=args.json)
    return 0


def _emit_screen(results: list[screening.FactorScreenResult], *, as_json: bool) -> None:
    if as_json:
        print(screening.to_json(results))
        return
    print(f"{'rank':>4}  {'factor':<26} {'mean_ic':>9} {'t_stat':>8} {'q_value':>9}  sig")
    print("-" * 64)
    for r in results:
        flag = "*" if r.significant else ""
        print(
            f"{r.rank:>4}  {r.name:<26} {r.mean_ic:>9.4f} {r.t_stat:>8.2f} {r.q_value:>9.4f}  {flag}"
        )
    print(f"\n{sum(1 for r in results if r.significant)}/{len(results)} significant at FDR target")


def screen_main(argv: Sequence[str] | None = None) -> int:
    """``quant-screen`` console script: screen factor panels from a parquet/CSV file."""
    parser = argparse.ArgumentParser(
        prog="quant-screen",
        description="Screen factor panels by cross-sectional IC (HAC t-stat, BH-FDR).",
    )
    parser.add_argument(
        "path", help="long-format parquet/CSV: date, asset, <factor cols>, forward_return"
    )
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--asset-col", default="asset")
    parser.add_argument("--return-col", default="forward_return")
    parser.add_argument("--factor-cols", nargs="*", default=None, help="default: all non-key cols")
    parser.add_argument("--method", default="spearman", choices=["spearman", "pearson"])
    parser.add_argument("--hac-lags", type=int, default=5)
    parser.add_argument("--fdr", type=float, default=0.10)
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args(argv)

    frame = screening.read_panel_frame(args.path)
    results = screening.screen_long_frame(
        frame,
        date_col=args.date_col,
        asset_col=args.asset_col,
        return_col=args.return_col,
        factor_cols=args.factor_cols,
        method=args.method,
        hac_lags=args.hac_lags,
        fdr=args.fdr,
    )
    _emit_screen(results, as_json=args.json)
    return 0


def _emit_factory(verdicts: list[factory.FactoryVerdict], *, as_json: bool) -> None:
    if as_json:
        print(factory.to_json(verdicts))
        return
    print(
        f"{'rank':>4}  {'factor':<26} {'mean_ic':>9} {'sharpe':>8} {'defl_sr':>8}  {'pass':>4}  reason"
    )
    print("-" * 78)
    for v in verdicts:
        flag = "yes" if v.passed else ""
        print(
            f"{v.rank:>4}  {v.name:<26} {v.mean_ic:>9.4f} {v.ann_sharpe:>8.2f} "
            f"{v.deflated_sharpe:>8.3f}  {flag:>4}  {v.reason}"
        )
    print(f"\n{sum(1 for v in verdicts if v.passed)}/{len(verdicts)} survived both gates")


def factory_main(argv: Sequence[str] | None = None) -> int:
    """``quant-factory`` console script: run the full screen->deflate->survivors loop on a file."""
    parser = argparse.ArgumentParser(
        prog="quant-factory",
        description="Run the alpha-factory loop (IC+FDR screen, then Deflated Sharpe) on a panel.",
    )
    parser.add_argument(
        "path", help="long-format parquet/CSV: date, asset, <factor cols>, forward_return"
    )
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--asset-col", default="asset")
    parser.add_argument("--return-col", default="forward_return")
    parser.add_argument("--factor-cols", nargs="*", default=None, help="default: all non-key cols")
    parser.add_argument("--method", default="spearman", choices=["spearman", "pearson"])
    parser.add_argument("--hac-lags", type=int, default=5)
    parser.add_argument("--fdr", type=float, default=0.10)
    parser.add_argument("--dsr-threshold", type=float, default=0.95)
    parser.add_argument(
        "--register", metavar="LEDGER.json", help="append survivors to a JSON discoveries ledger"
    )
    parser.add_argument("--source", help="provenance recorded with survivors (default: PATH)")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args(argv)

    frame = screening.read_panel_frame(args.path)
    verdicts = factory.run_factory_frame(
        frame,
        date_col=args.date_col,
        asset_col=args.asset_col,
        return_col=args.return_col,
        factor_cols=args.factor_cols,
        method=args.method,
        hac_lags=args.hac_lags,
        fdr=args.fdr,
        dsr_threshold=args.dsr_threshold,
    )
    _emit_factory(verdicts, as_json=args.json)
    if args.register:
        ledger = discoveries.DiscoveryLedger.load(args.register)
        params: dict[str, object] = {
            "method": args.method,
            "hac_lags": args.hac_lags,
            "fdr": args.fdr,
            "dsr_threshold": args.dsr_threshold,
        }
        manifest = provenance.build_manifest(frame, params=params, source=args.source or args.path)
        added = ledger.register_survivors(
            verdicts,
            source=args.source or args.path,
            screen_params=params,
            manifest=manifest,
        )
        ledger.save(args.register)
        # Note to stderr so --json stdout stays a clean verdict array.
        print(
            f"registered {len(added)} survivor(s) [run {manifest.run_id}] -> "
            f"{args.register} ({len(ledger)} total)",
            file=sys.stderr,
        )
    return 0


def provenance_main(argv: Sequence[str] | None = None) -> int:
    """``quant-provenance`` console script: inspect / verify reproducibility manifests in a ledger."""
    parser = argparse.ArgumentParser(
        prog="quant-provenance",
        description="Inspect and verify research provenance recorded in a discoveries ledger.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="list the run manifests in a ledger")
    p_list.add_argument("ledger", help="discoveries ledger JSON")
    p_list.add_argument("--json", action="store_true", help="emit JSON")

    p_show = sub.add_parser("show", help="print one manifest by run id")
    p_show.add_argument("ledger")
    p_show.add_argument("run_id")

    p_verify = sub.add_parser(
        "verify", help="re-derive a manifest against a panel file and report reproducibility"
    )
    p_verify.add_argument("ledger")
    p_verify.add_argument("run_id")
    p_verify.add_argument("panel", help="the data file to re-fingerprint")
    p_verify.add_argument("--json", action="store_true", help="emit JSON")

    args = parser.parse_args(argv)
    ledger = discoveries.DiscoveryLedger.load(args.ledger)

    if args.cmd == "list":
        manifests = ledger.manifests
        if args.json:
            print(json.dumps([m.to_dict() for m in manifests], indent=2))
        else:
            for m in manifests:
                print(f"{m.run_id}  {m.created_utc}  rows={m.n_rows}  {m.source or ''}")
            print(f"\n{len(manifests)} manifest(s)")
        return 0

    try:
        manifest = ledger.get_manifest(args.run_id)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.cmd == "show":
        print(json.dumps(manifest.to_dict(), indent=2))
        return 0

    frame = screening.read_panel_frame(args.panel)
    check = provenance.verify_manifest(manifest, frame)
    if args.json:
        print(json.dumps(check.to_dict(), indent=2))
    else:
        status = "REPRODUCIBLE" if check.reproducible else "NOT REPRODUCIBLE"
        print(f"{status}  run {manifest.run_id}")
        print(f"  {check.detail}")
    return 0 if check.reproducible else 1


if __name__ == "__main__":
    raise SystemExit(main())
