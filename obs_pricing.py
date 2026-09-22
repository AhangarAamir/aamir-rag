"""Resolve USD / 1M token rates for the selected model.

OpenAI's Models API does not return prices. At startup we fetch the public
LiteLLM catalog (same table most OSS cost dashboards use), cache it, then look
up OPENAI_MODEL. Env overrides win. If the provider reports metrics.cost on a
run, that value is used instead of the table.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.request
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.error import URLError

from agno.utils.log import log_info, log_warning

from config import DATA_DIR, OPENAI_MODEL

PRICES_URL = os.getenv(
    "OBS_PRICES_URL",
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json",
).strip()
CACHE_FILE = DATA_DIR / "model_prices.json"
FETCH_TIMEOUT_S = float(os.getenv("OBS_PRICES_TIMEOUT", "8"))

# Official short-context USD / 1M tokens: input, cached-input, output.
# Used when the catalog is unreachable. gpt-5.6-luna is $0.20 / $0.02 / $1.20.
_FALLBACK_RATES: dict[str, tuple[float, float, float]] = {
    "gpt-4o": (2.5, 1.25, 10.0),
    "gpt-4o-mini": (0.15, 0.075, 0.6),
    "gpt-4.1": (2.0, 0.5, 8.0),
    "gpt-4.1-mini": (0.4, 0.1, 1.6),
    "gpt-5.4": (2.5, 0.25, 15.0),
    "gpt-5": (2.5, 0.25, 15.0),
    "gpt-5.6-luna": (0.20, 0.02, 1.20),
    "gpt-5.6-terra": (2.0, 0.20, 12.0),
    "gpt-5.6-sol": (4.0, 0.40, 20.0),
    "gpt-6-astra": (10.0, 1.00, 50.0),
    "text-embedding-3-large": (0.13, 0.0, 0.0),
}

_LOCK = threading.Lock()
_CATALOG: dict[str, dict[str, Any]] = {}
_FETCHED_AT: str | None = None
_LOADED = False


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_override() -> tuple[float, float, float] | None:
    raw_in = os.getenv("OBS_INPUT_USD_PER_M", "").strip()
    raw_out = os.getenv("OBS_OUTPUT_USD_PER_M", "").strip()
    if not raw_in or not raw_out:
        return None
    raw_cached = os.getenv("OBS_CACHED_USD_PER_M", "").strip()
    cached = float(raw_cached) if raw_cached else float(raw_in) * 0.1
    return (float(raw_in), cached, float(raw_out))


def _per_million(entry: dict[str, Any]) -> tuple[float, float, float] | None:
    inp = entry.get("input_cost_per_token")
    out = entry.get("output_cost_per_token")
    if inp is None or out is None:
        return None
    cached = entry.get("cache_read_input_token_cost")
    if cached is None:
        cached = float(inp) * 0.1
    return (
        round(float(inp) * 1_000_000.0, 6),
        round(float(cached) * 1_000_000.0, 6),
        round(float(out) * 1_000_000.0, 6),
    )


def _usable_catalog(raw: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, entry in raw.items():
        if key == "sample_spec" or not isinstance(entry, dict):
            continue
        if entry.get("input_cost_per_token") is None:
            continue
        out[str(key)] = entry
    return out


def _save_cache(catalog: dict[str, dict[str, Any]], fetched_at: str) -> None:
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(
            json.dumps({"fetched_at": fetched_at, "catalog": catalog}),
            encoding="utf-8",
        )
    except OSError as exc:
        log_warning(f"Could not cache model prices: {exc}")


def _load_cache() -> tuple[dict[str, dict[str, Any]], str | None]:
    if not CACHE_FILE.is_file():
        return {}, None
    try:
        payload = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, None
    if isinstance(payload, dict) and "catalog" in payload:
        return _usable_catalog(payload.get("catalog")), payload.get("fetched_at")
    return _usable_catalog(payload), None


def _fetch_catalog() -> dict[str, dict[str, Any]]:
    request = urllib.request.Request(
        PRICES_URL,
        headers={"User-Agent": "legal-rag-agno-observability"},
    )
    with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_S) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return _usable_catalog(payload)


def load_catalog(raw: dict[str, Any], fetched_at: str | None = None) -> None:
    """Install a catalog dict (tests)."""
    global _CATALOG, _FETCHED_AT, _LOADED
    with _LOCK:
        _CATALOG = _usable_catalog(raw)
        _FETCHED_AT = fetched_at or _iso_now()
        _LOADED = True


def reset_for_tests() -> None:
    global _CATALOG, _FETCHED_AT, _LOADED
    with _LOCK:
        _CATALOG = {}
        _FETCHED_AT = None
        _LOADED = True


def refresh_prices(force: bool = False) -> None:
    """Fetch the LiteLLM price catalog once per process (or when force=True)."""
    global _CATALOG, _FETCHED_AT, _LOADED
    with _LOCK:
        if _LOADED and not force:
            return
        catalog: dict[str, dict[str, Any]] = {}
        fetched_at: str | None = None
        try:
            catalog = _fetch_catalog()
            fetched_at = _iso_now()
            if catalog:
                _save_cache(catalog, fetched_at)
        except (URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            log_warning(f"Model price catalog fetch failed, using cache/fallback: {exc}")
            catalog, fetched_at = _load_cache()
        _CATALOG = catalog
        _FETCHED_AT = fetched_at
        _LOADED = True
    inp, cached, out = model_rates(OPENAI_MODEL)
    source = pricing_source(OPENAI_MODEL)
    log_info(
        f"Observability pricing for {OPENAI_MODEL}: "
        f"${inp}/${cached}/${out} per 1M ({source})"
    )


def _lookup_catalog(model: str) -> tuple[float, float, float] | None:
    key = (model or "").strip()
    if not key:
        return None
    short = key.split("/")[-1]
    for candidate in (key, short, f"openai/{short}", f"azure/{short}"):
        entry = _CATALOG.get(candidate)
        if entry:
            rates = _per_million(entry)
            if rates:
                return rates
    openai_hit = None
    other_hit = None
    suffix = "/" + short
    for catalog_key, entry in _CATALOG.items():
        if catalog_key != short and not catalog_key.endswith(suffix):
            continue
        rates = _per_million(entry)
        if not rates:
            continue
        provider = str(entry.get("litellm_provider") or "")
        if provider == "openai":
            openai_hit = rates
            break
        if other_hit is None:
            other_hit = rates
    return openai_hit or other_hit


def _fallback_rates(model: str) -> tuple[float, float, float] | None:
    key = (model or "").strip()
    short = key.split("/")[-1]
    return _FALLBACK_RATES.get(key) or _FALLBACK_RATES.get(short)


def model_rates(model: str | None) -> tuple[float, float, float]:
    key = (model or "").strip() or OPENAI_MODEL
    if key == OPENAI_MODEL:
        override = _env_override()
        if override:
            return override
    found = _lookup_catalog(key)
    if found:
        return found
    fallback = _fallback_rates(key)
    if fallback:
        return fallback
    return (0.0, 0.0, 0.0)


def pricing_source(model: str | None) -> str:
    key = (model or "").strip() or OPENAI_MODEL
    if key == OPENAI_MODEL and _env_override():
        return "env"
    if _lookup_catalog(key):
        return "catalog"
    if _fallback_rates(key):
        return "fallback"
    return "unknown"


def catalog_status() -> dict[str, Any]:
    inp, cached, out = model_rates(OPENAI_MODEL)
    return {
        "selected_model": OPENAI_MODEL,
        "input_usd_per_m": inp,
        "cached_usd_per_m": cached,
        "output_usd_per_m": out,
        "source": pricing_source(OPENAI_MODEL),
        "fetched_at": _FETCHED_AT,
        "catalog_models": len(_CATALOG),
        "url": PRICES_URL,
    }


def estimate_cost(
    model: str | None,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    reported: float | None = None,
) -> float:
    if reported is not None and reported > 0:
        return float(reported)
    inp, cached, out = model_rates(model)
    billed_input = max(input_tokens - cached_tokens, 0)
    return (
        billed_input * inp + cached_tokens * cached + output_tokens * out
    ) / 1_000_000.0
