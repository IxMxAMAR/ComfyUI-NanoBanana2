"""Security regression tests for v2.1 audit fixes."""

import pytest

from gemini_client import (
    redact_secret, sanitize_model_id, get_api_key,
    is_transient_error, retry_with_backoff,
)


# ---------------------------------------------------------------------------
# Secret redaction — these would have leaked the user's key in v2.0 logs.
# ---------------------------------------------------------------------------

class TestRedactSecret:
    def test_redacts_google_api_key(self):
        msg = "Auth failed with key AIzaSyDxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        out = redact_secret(msg)
        assert "AIzaSy" not in out
        assert "REDACTED" in out

    def test_redacts_x_goog_api_key_header(self):
        msg = "x-goog-api-key: AIzaSecretXXXXXXXXXXXXXXXXXXXXXXXXXXX abc"
        out = redact_secret(msg)
        assert "REDACTED" in out

    def test_redacts_authorization_bearer(self):
        msg = "Authorization: Bearer ya29.abcdefSecret"
        out = redact_secret(msg)
        # "ya29..." doesn't match AIza regex, but the header pattern catches it
        assert "REDACTED" in out

    def test_empty_passthrough(self):
        assert redact_secret("") == ""
        assert redact_secret(None) is None

    def test_no_key_passthrough(self):
        msg = "Some boring error message"
        assert redact_secret(msg) == msg


# ---------------------------------------------------------------------------
# Model-ID sanitization — closes Lyria/Veo URL-path injection.
# ---------------------------------------------------------------------------

class TestSanitizeModelId:
    def test_accepts_normal_model(self):
        assert sanitize_model_id("gemini-3.1-pro-preview") == "gemini-3.1-pro-preview"
        assert sanitize_model_id("imagen-4.0-generate-001") == "imagen-4.0-generate-001"
        assert sanitize_model_id("lyria-3-pro-preview") == "lyria-3-pro-preview"

    def test_rejects_path_traversal(self):
        with pytest.raises(ValueError):
            sanitize_model_id("../other_endpoint")

    def test_rejects_slash(self):
        with pytest.raises(ValueError):
            sanitize_model_id("models/gemini/foo")

    def test_rejects_colon(self):
        with pytest.raises(ValueError):
            sanitize_model_id("gemini:predict")

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            sanitize_model_id("")

    def test_rejects_query_string(self):
        with pytest.raises(ValueError):
            sanitize_model_id("gemini?evil=1")


# ---------------------------------------------------------------------------
# API key resolution — quote stripping (.env mistake).
# ---------------------------------------------------------------------------

class TestGetApiKey:
    def test_direct_input(self):
        assert get_api_key("AIzaSomeKey").startswith("AIza")

    def test_strips_whitespace(self):
        assert get_api_key("  AIza  ") == "AIza"

    def test_strips_double_quotes(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        # User mistakenly typed `GEMINI_API_KEY="AIza..."` in .env
        assert get_api_key('"AIzaSecret"') == "AIzaSecret"

    def test_strips_single_quotes(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        assert get_api_key("'AIzaSecret'") == "AIzaSecret"

    def test_raises_when_missing(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(ValueError):
            get_api_key("")

    def test_env_fallback(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaFromEnv")
        assert get_api_key("") == "AIzaFromEnv"


# ---------------------------------------------------------------------------
# is_transient_error — v2.0 used substring soup which false-positived on
# "500x500" in prompts.
# ---------------------------------------------------------------------------

class TestIsTransientError:
    def test_word_boundary_avoids_500x500_false_positive(self):
        # "500x500" — "500" is followed by an alphanumeric, so the \b word
        # boundary in our regex does NOT match. (v2.0's `"500" in s` did.)
        e = ValueError("Image dimensions 500x500 are invalid")
        assert is_transient_error(e) is False

    def test_word_boundary_avoids_5000_false_positive(self):
        # "5000" — leading word boundary matches at "5", but trailing
        # boundary requires non-word after "0", so \b500\b does not match.
        e = ValueError("max_output_tokens=5000 exceeds limit")
        assert is_transient_error(e) is False

    def test_passthrough_on_unrelated_error(self):
        e = ValueError("something else entirely")
        assert is_transient_error(e) is False

    def test_detects_standalone_500(self):
        e = RuntimeError("HTTP 500 Internal Server Error")
        assert is_transient_error(e) is True

    def test_quota_429_is_permanent(self):
        # Quota exhaustion is permanent — retrying just wastes credits.
        e = RuntimeError("HTTP 429 RESOURCE_EXHAUSTED: quota exceeded")
        assert is_transient_error(e) is False


# ---------------------------------------------------------------------------
# retry_with_backoff — must always raise/return, never silently return None.
# ---------------------------------------------------------------------------

class TestRetryWithBackoff:
    def test_returns_value_on_success(self):
        assert retry_with_backoff(lambda: 42) == 42

    def test_raises_on_non_transient(self):
        calls = []

        def fn():
            calls.append(1)
            raise ValueError("nope")

        with pytest.raises(ValueError):
            retry_with_backoff(fn, retries=3, base_delay=0.01)
        assert len(calls) == 1  # No retries for non-transient

    def test_retries_then_raises_on_transient(self):
        calls = []

        def fn():
            calls.append(1)
            raise RuntimeError("HTTP 503 service unavailable")

        with pytest.raises(RuntimeError):
            retry_with_backoff(fn, retries=3, base_delay=0.01)
        assert len(calls) == 3  # tried all 3 attempts

    def test_eventually_succeeds(self):
        calls = []

        def fn():
            calls.append(1)
            if len(calls) < 2:
                raise RuntimeError("HTTP 503")
            return "ok"

        assert retry_with_backoff(fn, retries=3, base_delay=0.01) == "ok"
        assert len(calls) == 2
