"""Hermetic tests for ``quantengine.runtime.streaming.databento_config``.

Invariants exercised:
  - ``DATABENTO_API_KEY`` is the sole credential surface; no fallback.
  - Empty env-var is treated as missing (dotenv-injection UX).
  - ``__repr__`` never contains the full key (secret redaction).
  - ``MissingCredentialError`` is a distinct exception type — not a
    ``KeyError`` (so callers don't catch it via generic dict-lookup
    handling) and not a ``ValueError`` (so an omitted credential is
    not conflated with a malformed value).
  - Direct construction with an empty key raises ``ValueError``.
  - ``DatabentoConfig`` is frozen (mutation refused at runtime).
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from quantengine.runtime.streaming.databento_config import (
    DatabentoConfig,
    MissingCredentialError,
)

_FAKE_KEY = "db-fakekey0123456789abcdef0123456789"


class TestFromEnv:
    def test_happy_path_returns_config_with_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATABENTO_API_KEY", _FAKE_KEY)
        cfg = DatabentoConfig.from_env()
        assert cfg.api_key == _FAKE_KEY

    def test_missing_var_raises_missing_credential(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
        with pytest.raises(MissingCredentialError, match="DATABENTO_API_KEY"):
            DatabentoConfig.from_env()

    def test_empty_var_treated_as_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATABENTO_API_KEY", "")
        with pytest.raises(MissingCredentialError, match="DATABENTO_API_KEY"):
            DatabentoConfig.from_env()


class TestReprRedaction:
    def test_repr_does_not_contain_full_key(self) -> None:
        cfg = DatabentoConfig(api_key=_FAKE_KEY)
        assert _FAKE_KEY not in repr(cfg)

    def test_repr_shows_prefix_and_redaction_marker(self) -> None:
        cfg = DatabentoConfig(api_key=_FAKE_KEY)
        r = repr(cfg)
        assert _FAKE_KEY[:6] in r
        assert "***" in r

    def test_repr_does_not_crash_on_short_keys(self) -> None:
        # Pathological short key (not a real Databento key shape). The
        # redaction marker must still be present so a future reader of
        # the repr cannot mistake the full key as un-redacted.
        cfg = DatabentoConfig(api_key="abc")
        assert "***" in repr(cfg)


class TestDirectConstruction:
    def test_empty_api_key_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="api_key"):
            DatabentoConfig(api_key="")

    def test_frozen_refuses_mutation(self) -> None:
        cfg = DatabentoConfig(api_key=_FAKE_KEY)
        with pytest.raises(FrozenInstanceError):
            setattr(cfg, "api_key", "other")


class TestExceptionHierarchy:
    def test_missing_credential_error_is_not_keyerror(self) -> None:
        assert not issubclass(MissingCredentialError, KeyError)

    def test_missing_credential_error_is_not_valueerror(self) -> None:
        assert not issubclass(MissingCredentialError, ValueError)

    def test_missing_credential_error_is_runtimeerror(self) -> None:
        assert issubclass(MissingCredentialError, RuntimeError)
