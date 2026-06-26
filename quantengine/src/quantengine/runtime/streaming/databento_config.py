"""Databento Live API configuration: credentials via DATABENTO_API_KEY.

The Databento Live websocket authenticates via a single API key. The
key is the only secret carried by this config object; dataset, schema,
and symbol-subscription parameters are session metadata and live at the
``DatabentoTradeFeed`` construction site (PR2 of this sprint), not here.

Credentials load exclusively from the ``DATABENTO_API_KEY`` environment
variable. No default values, no fallback files, no command-line
overrides — operator-provided env var or refuse to construct.
``DatabentoConfig.__repr__`` redacts the key so structured logging
(``logger.info(config)``) and exception locals do not leak the secret
to journald / Sentry / external aggregators.

S22 precedent (``quantengine.execution.ibkr.config``) treats its env
vars as connection coordinates rather than secrets and omits redaction
on that basis. Databento's API key IS a secret — the redaction here is
the load-bearing distinction from the S22 pattern.

See sprint plan ``s36-real-feeds-and-async-broker`` D1 for the
broader credentials-and-redaction rationale; D4 for the SDK pin policy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


class MissingCredentialError(RuntimeError):
    """Raised by ``DatabentoConfig.from_env`` when the credential env var is unset.

    Distinct from ``KeyError`` so callers can catch the credentials gap
    specifically without swallowing unrelated dict lookups. Distinct
    from ``ValueError`` so a configuration omission is not conflated
    with a malformed-value error (an empty string is treated as a
    missing credential, not a value error — see ``from_env`` docstring).
    """


_REPR_PREFIX_CHARS = 6


@dataclass(frozen=True, slots=True)
class DatabentoConfig:
    """Configuration for a Databento Live API session.

    Only the API key is captured here. Dataset, schema, and symbol-
    subscription parameters belong at ``DatabentoTradeFeed`` construction
    (PR2); they are session metadata, not credentials.
    """

    api_key: str

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("api_key must be non-empty")

    def __repr__(self) -> str:
        prefix = self.api_key[:_REPR_PREFIX_CHARS]
        return f"DatabentoConfig(api_key='{prefix}***')"

    @classmethod
    def from_env(cls) -> DatabentoConfig:
        """Build a ``DatabentoConfig`` from ``DATABENTO_API_KEY``.

        Raises ``MissingCredentialError`` if the variable is unset OR
        empty. The empty-string case is treated identically to "unset"
        because dotenv-style loaders sometimes inject empty values when
        a key is declared but not provided — surfacing that as a
        ``ValueError`` from ``__post_init__`` would mask the real
        operator action (set the variable).
        """
        api_key = os.environ.get("DATABENTO_API_KEY", "")
        if not api_key:
            raise MissingCredentialError(
                "DATABENTO_API_KEY environment variable is not set "
                "(or is empty). DatabentoConfig refuses default values "
                "by design; set the variable before constructing the "
                "Databento feed."
            )
        return cls(api_key=api_key)


__all__ = ["DatabentoConfig", "MissingCredentialError"]
