"""HTTP request helpers with exponential-backoff retry, size limits, and SSRF guards.

v2.1 changes (security pass):
  - Response objects are always closed (context manager / explicit close)
    so connections don't leak when the body isn't fully consumed.
  - `download_file` enforces a hard max_bytes cap (default 256 MiB) to
    prevent a malicious or misconfigured server from OOM'ing the worker.
  - `download_file` disables auto-redirect-follow by default — callers
    that need redirects can opt in via allow_redirects=True. This blocks
    one common SSRF vector (HTTP → file:// or internal IP redirects).
  - `_get_retry_delay` honors RFC 7231 HTTP-date Retry-After headers, not
    only integer seconds.
  - Default request timeout reduced from 60s to 30s; download timeout to
    60s. Long-running ops should pass explicit timeouts.
  - Module-level Session enables connection pooling / keep-alive across
    repeated calls.
"""

import io
import random
import time
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from typing import Optional

import requests
from requests.exceptions import ConnectionError as ReqConnectionError
from requests.exceptions import Timeout as ReqTimeout
from requests.exceptions import ChunkedEncodingError as ReqChunkedEncodingError
from requests.exceptions import RequestException as ReqRequestException

from .errors import APITransientError, APIPermanentError, parse_error_response


# Module-level session for connection pooling. requests creates a new
# TCP/TLS connection per call when no session is supplied — bad for latency
# when a workflow does many API calls back-to-back.
_DEFAULT_SESSION = requests.Session()

# Hard ceilings to prevent OOM / hang. Override per-call if needed.
DEFAULT_MAX_BYTES = 256 * 1024 * 1024   # 256 MiB
DEFAULT_TIMEOUT = 30.0                  # per-request
DEFAULT_DOWNLOAD_TIMEOUT = 60.0


def api_request_with_retry(
    method: str,
    url: str,
    *,
    session: Optional[requests.Session] = None,
    max_retries: int = 3,
    base_delay: float = 2.0,
    timeout: float = DEFAULT_TIMEOUT,
    transient_codes: tuple = (429, 500, 502, 503, 504),
    service_name: str = "API",
    **kwargs,
) -> requests.Response:
    """Make an HTTP request with exponential backoff retry on transient failures.

    Returns a `requests.Response`. Caller is responsible for closing it (or
    use a `with` block). Non-2xx-but-allowed responses are returned to the
    caller; transient errors are retried with jittered backoff.

    Raises ValueError if `max_retries < 0`.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")

    requester = session or _DEFAULT_SESSION
    last_error: Optional[APITransientError] = None

    for attempt in range(max_retries + 1):
        try:
            response = requester.request(method, url, timeout=timeout, **kwargs)

            if response.status_code < 400:
                return response

            body = response.text
            error = parse_error_response(service_name, response.status_code, body)

            if response.status_code in transient_codes:
                # Quota-style 429s come back as APIQuotaError (subclass of
                # APIPermanentError) from parse_error_response — don't retry.
                from .errors import APIQuotaError
                if isinstance(error, APIQuotaError):
                    response.close()
                    raise error
                last_error = error  # type: ignore[assignment]
                response.close()
                if attempt < max_retries:
                    delay = _get_retry_delay(response, base_delay, attempt)
                    time.sleep(delay)
                    continue
                else:
                    raise last_error
            else:
                response.close()
                raise error

        except (
            ReqConnectionError,
            ReqTimeout,
            ReqChunkedEncodingError,
        ) as exc:
            last_error = APITransientError(
                service_name, 0, f"Connection error: {exc}"
            )
            if attempt < max_retries:
                delay = (base_delay * (2 ** attempt)) * random.uniform(0.8, 1.2)
                time.sleep(delay)
                continue
            else:
                raise last_error from exc

    # Defensive — every iteration above either returns/continues/raises.
    if last_error is not None:
        raise last_error
    raise RuntimeError("api_request_with_retry exhausted attempts without an error object")


def _get_retry_delay(
    response: requests.Response, base_delay: float, attempt: int
) -> float:
    """Compute backoff delay, honoring Retry-After (seconds OR HTTP-date)."""
    retry_after = response.headers.get("Retry-After")
    if retry_after is not None:
        # Try integer/float seconds first
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass
        # Try RFC 7231 HTTP-date
        try:
            target = parsedate_to_datetime(retry_after)
            if target is not None:
                # Naive dates are treated as UTC per RFC 7231
                if target.tzinfo is None:
                    target = target.replace(tzinfo=timezone.utc)
                now = datetime.now(timezone.utc)
                delta = (target - now).total_seconds()
                if delta > 0:
                    return delta
        except (TypeError, ValueError):
            pass
    # Fallback: jittered exponential
    return (base_delay * (2 ** attempt)) * random.uniform(0.8, 1.2)


def download_file(
    url: str,
    *,
    retries: int = 3,
    timeout: float = DEFAULT_DOWNLOAD_TIMEOUT,
    chunk_size: int = 8192,
    max_bytes: int = DEFAULT_MAX_BYTES,
    allow_redirects: bool = False,
    session: Optional[requests.Session] = None,
) -> bytes:
    """Download a file into memory with retry, size cap, and SSRF guard.

    Defaults to allow_redirects=False (set True only when you trust the URL).
    Enforces max_bytes to prevent a hostile / misconfigured server from
    OOM'ing the worker by streaming an infinite body.
    """
    last_error: Optional[Exception] = None
    requester = session or _DEFAULT_SESSION

    for attempt in range(retries + 1):
        try:
            with requester.get(
                url, stream=True, timeout=timeout, allow_redirects=allow_redirects
            ) as response:
                response.raise_for_status()

                buffer = io.BytesIO()
                total = 0
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > max_bytes:
                        raise APIPermanentError(
                            "download", 413,
                            f"Response exceeded max_bytes={max_bytes}",
                        )
                    buffer.write(chunk)
                return buffer.getvalue()

        except APIPermanentError:
            # Don't retry size-cap failures — they're deterministic.
            raise
        except ReqRequestException as exc:
            last_error = exc
            if attempt < retries:
                delay = (2.0 * (2 ** attempt)) * random.uniform(0.8, 1.2)
                time.sleep(delay)
                continue

    raise APITransientError(
        "download", 0,
        f"Download failed after {retries + 1} attempts: {last_error}",
    )


def stream_to_file(
    url: str,
    file_path: str,
    *,
    retries: int = 3,
    timeout: float = DEFAULT_DOWNLOAD_TIMEOUT,
    chunk_size: int = 8192,
    max_bytes: int = DEFAULT_MAX_BYTES,
    allow_redirects: bool = False,
    session: Optional[requests.Session] = None,
) -> int:
    """Stream a URL to disk with retry, size cap, and explicit response close.

    Returns the number of bytes written. Removes any partial file on
    failure so callers don't leave half-downloaded videos behind.
    """
    import os
    last_error: Optional[Exception] = None
    requester = session or _DEFAULT_SESSION

    for attempt in range(retries + 1):
        try:
            with requester.get(
                url, stream=True, timeout=timeout, allow_redirects=allow_redirects
            ) as response:
                response.raise_for_status()

                total = 0
                with open(file_path, "wb") as fh:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > max_bytes:
                            raise APIPermanentError(
                                "download", 413,
                                f"Response exceeded max_bytes={max_bytes}",
                            )
                        fh.write(chunk)
                return total

        except APIPermanentError:
            # Clean up partial file then re-raise (deterministic — no retry).
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except OSError:
                pass
            raise
        except ReqRequestException as exc:
            last_error = exc
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except OSError:
                pass
            if attempt < retries:
                delay = (2.0 * (2 ** attempt)) * random.uniform(0.8, 1.2)
                time.sleep(delay)
                continue

    raise APITransientError(
        "download", 0,
        f"stream_to_file failed after {retries + 1} attempts: {last_error}",
    )
