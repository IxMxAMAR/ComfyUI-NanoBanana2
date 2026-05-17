"""Tests for shared/retry.py — max_bytes cap, response close, Retry-After parsing."""

import pytest
import requests
from unittest.mock import patch, MagicMock

from shared.retry import (
    api_request_with_retry, download_file, stream_to_file,
    _get_retry_delay, DEFAULT_MAX_BYTES,
)
from shared.errors import APIPermanentError, APITransientError


class TestApiRequestRetry:
    def test_rejects_negative_max_retries(self):
        with pytest.raises(ValueError):
            api_request_with_retry("GET", "http://example.com", max_retries=-1)

    def test_returns_2xx(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("shared.retry._DEFAULT_SESSION.request", return_value=mock_resp):
            r = api_request_with_retry("GET", "http://example.com")
            assert r.status_code == 200


class TestDownloadFile:
    def test_enforces_max_bytes(self):
        """A server sending more than max_bytes must trigger APIPermanentError."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = lambda: None
        # Yield 5x4kb chunks = 20kb, then cap at 10kb
        mock_resp.iter_content = lambda chunk_size: (b"x" * 4096 for _ in range(5))
        mock_resp.__enter__ = lambda self: self
        mock_resp.__exit__ = lambda self, *a: None

        with patch("shared.retry._DEFAULT_SESSION.get", return_value=mock_resp):
            with pytest.raises(APIPermanentError):
                download_file("http://example.com/big", max_bytes=10_240, retries=0)

    def test_does_not_follow_redirects_by_default(self):
        """SSRF guard: redirects must be opt-in."""
        capture = {}

        def fake_get(url, **kw):
            capture.update(kw)
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.raise_for_status = lambda: None
            mock_resp.iter_content = lambda chunk_size: iter([b"ok"])
            mock_resp.__enter__ = lambda self: self
            mock_resp.__exit__ = lambda self, *a: None
            return mock_resp

        with patch("shared.retry._DEFAULT_SESSION.get", side_effect=fake_get):
            download_file("http://example.com/x", retries=0)
            assert capture.get("allow_redirects") is False


class TestStreamToFile:
    def test_enforces_max_bytes_and_cleans_up(self, tmp_path):
        out = tmp_path / "video.bin"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = lambda: None
        mock_resp.iter_content = lambda chunk_size: (b"x" * 4096 for _ in range(5))
        mock_resp.__enter__ = lambda self: self
        mock_resp.__exit__ = lambda self, *a: None

        with patch("shared.retry._DEFAULT_SESSION.get", return_value=mock_resp):
            with pytest.raises(APIPermanentError):
                stream_to_file(
                    "http://example.com/big", str(out),
                    max_bytes=10_240, retries=0,
                )
        # Partial file should have been removed
        assert not out.exists()

    def test_successful_download(self, tmp_path):
        out = tmp_path / "small.bin"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = lambda: None
        mock_resp.iter_content = lambda chunk_size: iter([b"hello", b" world"])
        mock_resp.__enter__ = lambda self: self
        mock_resp.__exit__ = lambda self, *a: None

        with patch("shared.retry._DEFAULT_SESSION.get", return_value=mock_resp):
            n = stream_to_file("http://example.com/x", str(out), retries=0)
        assert n == len(b"hello world")
        assert out.read_bytes() == b"hello world"


class TestGetRetryDelay:
    def test_integer_seconds(self):
        resp = MagicMock()
        resp.headers = {"Retry-After": "5"}
        assert _get_retry_delay(resp, base_delay=1.0, attempt=0) == 5.0

    def test_http_date(self):
        from email.utils import format_datetime
        from datetime import datetime, timezone, timedelta

        target = datetime.now(timezone.utc) + timedelta(seconds=30)
        resp = MagicMock()
        resp.headers = {"Retry-After": format_datetime(target, usegmt=True)}
        d = _get_retry_delay(resp, base_delay=1.0, attempt=0)
        # Allow a few seconds of slop for test timing
        assert 25 <= d <= 35

    def test_falls_back_when_no_header(self):
        resp = MagicMock()
        resp.headers = {}
        d = _get_retry_delay(resp, base_delay=1.0, attempt=2)
        # jittered exp: 1 * 4 * [0.8, 1.2] = [3.2, 4.8]
        assert 3.0 <= d <= 5.0
