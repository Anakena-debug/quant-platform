"""The ``__init__.pyi`` stub must mirror the runtime lazy surface exactly.

The top-level convenience surface is resolved at runtime by ``__getattr__`` over the
``_EXPORTS`` / ``_SUBPACKAGES`` tables (``quantcore/__init__.py``). The companion
``__init__.pyi`` re-declares that surface so static tooling sees real types. The two can
drift silently — add to ``_EXPORTS`` but forget the stub and the new symbol is typed as
``Any`` to every consumer. This test makes "stub == runtime surface" a checkable property:
it parses the stub with ``ast`` and asserts the re-exported names (and the module each is
sourced from) match the runtime tables byte-for-byte.
"""

from __future__ import annotations

import ast
import pathlib

import quantcore


def _stub_path() -> pathlib.Path:
    path = pathlib.Path(quantcore.__file__).with_name("__init__.pyi")
    assert path.is_file(), f"missing type stub: {path}"
    return path


def _parse_reexports() -> tuple[set[str], dict[str, str]]:
    """Return ``(subpackage_names, {symbol: providing_module})`` declared by the stub.

    Subpackages come from ``from . import X as X`` (no submodule); symbols from
    ``from .mod[.sub] import Y as Y``. Every alias must be an explicit re-export
    (``asname == name``) or it would not be part of the public interface.
    """
    tree = ast.parse(_stub_path().read_text())
    subpackages: set[str] = set()
    symbols: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level != 1:
            continue
        for alias in node.names:
            assert alias.asname == alias.name, (
                f"{alias.name!r} must be re-exported explicitly as "
                f"`{alias.name} as {alias.name}` for the stub to export it"
            )
            if node.module is None:  # from . import X as X  -> subpackage
                subpackages.add(alias.name)
            else:  # from .mod import Y as Y  -> curated symbol
                symbols[alias.name] = f"quantcore.{node.module}"
    return subpackages, symbols


def test_stub_subpackages_match_runtime():
    subpackages, _ = _parse_reexports()
    assert subpackages == set(quantcore._SUBPACKAGES)  # pyright: ignore[reportPrivateUsage]


def test_stub_symbols_and_their_modules_match_runtime():
    # Dict equality pins both the symbol set AND the module each is typed from, so a stub
    # that imports `sharpe_ratio` from the wrong module also fails here.
    _, symbols = _parse_reexports()
    assert symbols == dict(quantcore._EXPORTS)  # pyright: ignore[reportPrivateUsage]


def test_py_typed_marker_ships():
    marker = pathlib.Path(quantcore.__file__).with_name("py.typed")
    assert marker.is_file(), "PEP 561 py.typed marker is required for the stub to be honored"
