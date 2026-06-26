"""Module entry point for ``python -m quantengine.runtime.streaming``.

Delegates to :func:`quantengine.runtime.streaming.cli.main`. Subcommands:
``start``, ``replay``, ``status`` (see ``cli.py`` module docstring).

Per S35 D13, this is the only sanctioned invocation path; notebook-
driven use of the engine is not supported.
"""

from __future__ import annotations

import sys

from quantengine.runtime.streaming.cli import main

if __name__ == "__main__":
    sys.exit(main())
