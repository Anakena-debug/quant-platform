"""Minimal pytest-compat runner — pytest unavailable in this sandbox.

Supports: plain test functions, ``pytest.raises``, ``pytest.parametrize``.
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Iterable, List, Tuple

import re


# -------------------- pytest shim --------------------
class _PytestShim(ModuleType):
    def __init__(self) -> None:
        super().__init__("pytest")
        self._param_stack: List[Tuple[str, Iterable]] = []

    @contextmanager
    def raises(self, exc_type, match: str | None = None):
        try:
            yield
        except exc_type as e:
            if match is not None and not re.search(match, str(e)):
                raise AssertionError(f"Exception message {e!r} does not match /{match}/")
            return
        except BaseException as e:
            raise AssertionError(f"Expected {exc_type.__name__}, got {type(e).__name__}: {e}")
        else:
            raise AssertionError(f"{exc_type.__name__} was not raised")

    class mark:  # noqa: N801
        @staticmethod
        def parametrize(argnames, argvalues):
            """Decorator: attach a list of param kwarg-dicts to the function."""
            if isinstance(argnames, str):
                if "," in argnames:
                    names = [n.strip() for n in argnames.split(",")]
                else:
                    names = [argnames]
            else:
                names = list(argnames)

            def decorator(fn):
                params = []
                for vals in argvalues:
                    if not isinstance(vals, (tuple, list)):
                        vals = (vals,)
                    params.append(dict(zip(names, vals)))
                existing = getattr(fn, "_parametrize_params", [])
                fn._parametrize_params = existing + params  # type: ignore[attr-defined]
                return fn

            return decorator

        @staticmethod
        def xfail(fn=None, *, strict: bool = False, reason: str = ""):
            """Decorator: mark a test as expected-failure.

            If strict=True, an unexpected pass (XPASS) counts as a failure.
            """

            def decorator(f):
                f._xfail = {"strict": strict, "reason": reason}  # type: ignore[attr-defined]
                return f

            if fn is not None:
                # Called as @pytest.mark.xfail (no parens)
                return decorator(fn)
            return decorator


pytest_shim = _PytestShim()
sys.modules["pytest"] = pytest_shim


# -------------------- runner --------------------
def load(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def run_module(path: Path) -> Tuple[int, int]:
    print(f"\n=== {path.name} ===")
    mod = load(path)
    passed, failed = 0, 0
    for name, obj in sorted(vars(mod).items()):
        if not (name.startswith("test_") and callable(obj)):
            continue
        params_list = getattr(obj, "_parametrize_params", [None])
        for i, params in enumerate(params_list):
            label = f"{name}[{i}]" if params is not None else name
            xfail = getattr(obj, "_xfail", None)
            try:
                if params is None:
                    obj()
                else:
                    obj(**params)
                if xfail is not None:
                    # Test passed but was expected to fail
                    if xfail["strict"]:
                        print(f"  XPASS {label} (strict xfail — counts as FAIL)")
                        failed += 1
                    else:
                        print(f"  XPASS {label}")
                        passed += 1
                else:
                    print(f"  PASS  {label}")
                    passed += 1
            except (AssertionError, Exception) as e:
                if xfail is not None:
                    # Test failed as expected
                    print(f"  XFAIL {label}")
                    passed += 1
                elif isinstance(e, AssertionError):
                    print(f"  FAIL  {label}: {e}")
                    traceback.print_exc()
                    failed += 1
                else:
                    print(f"  ERROR {label}: {type(e).__name__}: {e}")
                    traceback.print_exc()
                    failed += 1
    print(f"--> {passed} passed / {failed} failed")
    return passed, failed


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    files = sorted(root.glob("test_*_regressions.py"))
    tot_p = tot_f = 0
    for f in files:
        p, f_ = run_module(f)
        tot_p += p
        tot_f += f_
    print()
    print(f"TOTAL: {tot_p} passed, {tot_f} failed")
    sys.exit(0 if tot_f == 0 else 1)
