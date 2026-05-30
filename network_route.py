"""NanoBanana network routing.

Provides a NanoBanana_NetworkRoute node that produces an NB_NETWORK config
(a typed dict) which every other NanoBanana node accepts as an optional
input. When wired, the recipient routes its HTTP traffic through the
configured proxy — letting you make Gemini API calls appear to originate
from a specific region (e.g. the US) without affecting any other apps on
the machine.

Tested by actually issuing a request through the configured proxy to
https://ipinfo.io/json and reporting the egress IP + country — so you can
verify, in the workflow, that your traffic is going where you expect.
"""
from __future__ import annotations

import json
import os
import urllib.parse


def _validate_proxy_url(url: str) -> str:
    """Light validation of the proxy URL — return the normalized url or raise."""
    url = (url or "").strip()
    if not url:
        return ""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https", "socks5", "socks5h", "socks4"):
        raise ValueError(
            f"proxy_url must start with http://, https://, socks5://, socks5h:// or socks4://, "
            f"got: {parsed.scheme!r}"
        )
    if not parsed.hostname:
        raise ValueError(f"proxy_url is missing a host: {url!r}")
    return url


def _probe_egress(proxy_url: str, timeout: float = 8.0) -> dict:
    """Issue a one-shot GET to https://ipinfo.io/json through the configured
    proxy (or direct if proxy_url=="") and return the parsed JSON. Used by the
    test_now widget so the user can verify routing on the spot."""
    import requests
    proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None
    r = requests.get("https://ipinfo.io/json", proxies=proxies, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _format_probe(info: dict, proxy_url: str) -> str:
    ip = info.get("ip", "?")
    country = info.get("country", "?")
    region = info.get("region", "")
    city = info.get("city", "")
    org = info.get("org", "")
    bits = [f"egress IP {ip}", f"country {country}"]
    if region or city:
        bits.append(f"{region}/{city}".strip("/"))
    if org:
        bits.append(org)
    route = "DIRECT" if not proxy_url else "via proxy"
    return f"{route} | " + " | ".join(bits)


class NanoBanana_NetworkRoute:
    """Produce an NB_NETWORK config for every NanoBanana action node.

    Wire the `network` output into ANY NanoBanana node's `network` input —
    that node's request to Google will tunnel through the configured proxy.
    Leave a node's network input unconnected and it calls directly as before.

    Typical use: rent a US-egress proxy (Mullvad SOCKS5, your own VPS running
    dante/squid, or a paid residential service) and plug its URL in to make
    every NB API call appear to come from the US.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Master switch. When OFF, recipients call directly even if wired.",
                }),
                "proxy_url": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Full proxy URL. Examples:\n"
                        "  http://us-proxy.example.com:3128\n"
                        "  http://user:pass@us-proxy.example.com:3128\n"
                        "  socks5://127.0.0.1:1080\n"
                        "  socks5h://user:pass@us-vps.example.com:1080\n"
                        "Use socks5h:// (not socks5://) when you want DNS resolved "
                        "on the proxy side too — that prevents your local resolver "
                        "from leaking the original lookup."
                    ),
                }),
                "label": ("STRING", {
                    "default": "US route",
                    "tooltip": "Free-form tag shown in the status text. Cosmetic only.",
                }),
                "test_now": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Toggle to TRUE to probe the proxy: makes a GET to ipinfo.io "
                        "through the configured proxy and reports the egress IP, "
                        "country, and ISP. Use this to verify your route works "
                        "before plugging into production nodes."
                    ),
                }),
            },
            "optional": {
                "http_username": ("STRING", {
                    "default": "",
                    "tooltip": "Optional. If set, overrides any user:pass in proxy_url.",
                }),
                "http_password": ("STRING", {
                    "default": "",
                    "password": True,
                    "tooltip": "Optional. Paired with http_username.",
                }),
            },
        }

    RETURN_TYPES = ("NB_NETWORK", "STRING")
    RETURN_NAMES = ("network", "status")
    FUNCTION = "build"
    CATEGORY = "NanoBanana2/Config"

    def build(self, enabled: bool, proxy_url: str, label: str, test_now: bool,
              http_username: str = "", http_password: str = ""):
        try:
            url = _validate_proxy_url(proxy_url)
        except Exception as e:
            return ({"enabled": False, "proxy_url": "", "label": label,
                     "error": str(e)},
                    f"INVALID proxy_url: {e}")

        # If creds were supplied separately, splice them into the URL.
        if url and http_username:
            parsed = urllib.parse.urlparse(url)
            netloc = f"{urllib.parse.quote(http_username, safe='')}"
            if http_password:
                netloc += f":{urllib.parse.quote(http_password, safe='')}"
            # Strip any existing creds from parsed.netloc; keep host:port
            host = parsed.hostname or ""
            port = f":{parsed.port}" if parsed.port else ""
            netloc = f"{netloc}@{host}{port}"
            url = urllib.parse.urlunparse(parsed._replace(netloc=netloc))

        cfg = {"enabled": bool(enabled), "proxy_url": url if enabled else "",
               "label": label}

        # Optionally verify by actually hitting ipinfo.io through the route.
        status = f"{label} | enabled={enabled} | proxy={url or '(none)'}"
        if test_now:
            try:
                info = _probe_egress(url if enabled else "")
                status = f"{label} | " + _format_probe(info, url if enabled else "")
                # Stash probe result in the config so downstream debug nodes can see it.
                cfg["last_probe"] = info
            except Exception as e:
                status = f"{label} | PROBE FAILED: {e}"
                cfg["last_probe_error"] = str(e)

        # Always print so the user sees it in the console too.
        print(f"[NanoBanana NetworkRoute] {status}")
        return (cfg, status)


NODE_CLASS_MAPPINGS = {
    "NanoBanana_NetworkRoute": NanoBanana_NetworkRoute,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBanana_NetworkRoute": "NanoBanana - Network Route",
}
