"""Explicit provider diagnostics. Not part of quiet inventory."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from .providers.http import (
    FailureClass,
    FetchFailure,
    FetchResult,
    classify_http_status,
    get_json_result,
    quota_from_headers,
)

CEREBRAS_PUBLIC_MODELS = "https://api.cerebras.ai/public/v1/models"
CEREBRAS_AUTH_MODELS = "https://api.cerebras.ai/v1/models"
CEREBRAS_CHAT = "https://api.cerebras.ai/v1/chat/completions"

GROQ_MODELS = "https://api.groq.com/openai/v1/models"
OPENROUTER_MODELS = "https://openrouter.ai/api/v1/models"

_CEREBRAS_TEAM_GUIDANCE = (
    "Cerebras key appears attached to a Team organization without "
    "active subscription/credits.\n\n"
    "Cerebras documents that free API usage is available from the "
    "Personal account.\n\n"
    "Create/use a Personal-account API key and retry."
)


def classify_cerebras_status(failure: FetchFailure | None) -> str:
    if failure is None:
        return "WORKING"
    if failure.failure_class == FailureClass.TEAM_BALANCE_REQUIRED or failure.status == 402:
        return "TEAM_BALANCE_REQUIRED"
    if failure.failure_class == FailureClass.AUTH_INVALID or failure.status == 401:
        return "AUTH_INVALID"
    if failure.failure_class == FailureClass.AUTH_FORBIDDEN or failure.status == 403:
        return "AUTH_FORBIDDEN"
    if failure.failure_class == FailureClass.RATE_LIMITED or failure.status == 429:
        return "RATE_LIMITED"
    if failure.failure_class == FailureClass.MODEL_UNAVAILABLE or failure.status == 404:
        return "MODEL_UNAVAILABLE"
    if failure.failure_class == FailureClass.WORKING:
        return "WORKING"
    return "UNKNOWN"


def _model_ids(payload: dict[str, Any] | None) -> list[str]:
    if not payload:
        return []
    rows = payload.get("data") or []
    out: list[str] = []
    for item in rows:
        if isinstance(item, dict) and item.get("id"):
            out.append(str(item["id"]))
    return out


def _key_present(env_name: str) -> bool:
    return bool(os.environ.get(env_name, "").strip())


def _catalog_section(result: FetchResult) -> dict[str, Any]:
    if result.ok is not None:
        return {
            "status": "ok",
            "models": _model_ids(result.ok.body),
            "quota_headers_parsed": list(
                (quota_from_headers(result.ok.headers)[0] is not None,)
            ),
        }
    failure = result.failure
    return {
        "status": failure.status if failure else None,
        "failure_class": failure.failure_class.value if failure else FailureClass.UNKNOWN.value,
        "message": failure.message if failure else "",
        "sanitized_code": failure.sanitized_code if failure else None,
        "models": [],
    }


def doctor_cerebras(*, probe: bool = False) -> dict[str, Any]:
    configured = _key_present("CEREBRAS_API_KEY")
    key = os.environ.get("CEREBRAS_API_KEY") if configured else None

    public = get_json_result(
        CEREBRAS_PUBLIC_MODELS,
        api_key=None,
        provider="cerebras",
        endpoint_class="public_catalog",
    )
    auth = (
        get_json_result(
            CEREBRAS_AUTH_MODELS,
            api_key=key,
            provider="cerebras",
            endpoint_class="auth_catalog",
        )
        if configured
        else FetchResult(
            failure=FetchFailure(
                provider="cerebras",
                endpoint_class="auth_catalog",
                status=None,
                failure_class=FailureClass.AUTH_INVALID,
                message="CEREBRAS_API_KEY not set",
            )
        )
    )

    classification = "WORKING" if auth.ok is not None else classify_cerebras_status(auth.failure)
    if not configured:
        classification = "AUTH_INVALID"

    guidance = ""
    if classification == "TEAM_BALANCE_REQUIRED":
        guidance = _CEREBRAS_TEAM_GUIDANCE

    probe_status: dict[str, Any] = {"ran": False}
    if probe and configured and auth.ok is not None:
        models = _model_ids(auth.ok.body)
        model = models[0] if models else None
        probe_status = _probe_chat(
            url=CEREBRAS_CHAT,
            api_key=key or "",
            model=model,
            provider="cerebras",
        )
        if probe_status.get("classification") and probe_status["classification"] != "WORKING":
            classification = str(probe_status["classification"])

    report = {
        "provider": "cerebras",
        "configured": configured,
        "public_catalog": _catalog_section(public),
        "authenticated_catalog": _catalog_section(auth),
        "accessible_model_ids": _model_ids(auth.ok.body) if auth.ok else [],
        "classification": classification,
        "guidance": guidance,
        "probe": probe_status,
    }
    return report


def _probe_chat(*, url: str, api_key: str, model: str | None, provider: str) -> dict[str, Any]:
    """ONE minimal chat completion. Opt-in only. Never log the key."""
    if not model:
        return {
            "ran": True,
            "classification": "MODEL_UNAVAILABLE",
            "status": None,
            "message": "no accessible model id for probe",
        }
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
        }
    ).encode()
    headers = {
        "User-Agent": "kerdoios/0.1",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=20.0) as resp:
            status = getattr(resp, "status", 200)
            raw_headers = {str(k).lower(): str(v) for k, v in resp.headers.items()}
            resp.read()
        return {
            "ran": True,
            "classification": "WORKING",
            "status": status,
            "model": model,
            "rate_limit_headers": {
                k: v
                for k, v in raw_headers.items()
                if k == "retry-after" or k.startswith("x-ratelimit") or "quota" in k
            },
        }
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read()
        except Exception:  # noqa: BLE001 — fail-open inventory/doctor
            body = b""
        message = body.decode("utf-8", errors="replace")[:240]
        failure_class = classify_http_status(int(exc.code), provider=provider)
        classification = failure_class.value
        if provider == "cerebras" and int(exc.code) == 402:
            classification = "TEAM_BALANCE_REQUIRED"
        return {
            "ran": True,
            "classification": classification,
            "status": int(exc.code),
            "model": model,
            "message": message.replace(api_key, "[redacted]") if api_key and api_key in message else message,
        }
    except Exception as exc:  # noqa: BLE001 — doctor must not crash
        return {
            "ran": True,
            "classification": "UNKNOWN",
            "status": None,
            "model": model,
            "message": str(exc)[:240],
        }


def doctor_groq(*, probe: bool = False) -> dict[str, Any]:
    configured = _key_present("GROQ_API_KEY")
    key = os.environ.get("GROQ_API_KEY") if configured else None
    auth = (
        get_json_result(
            GROQ_MODELS,
            api_key=key,
            provider="groq",
            endpoint_class="auth_catalog",
        )
        if configured
        else FetchResult(
            failure=FetchFailure(
                provider="groq",
                endpoint_class="auth_catalog",
                status=None,
                failure_class=FailureClass.AUTH_INVALID,
                message="GROQ_API_KEY not set",
            )
        )
    )
    classification = "WORKING" if auth.ok else classify_cerebras_status(auth.failure)
    if not configured:
        classification = "AUTH_INVALID"
    probe_status: dict[str, Any] = {"ran": False}
    if probe and configured and auth.ok is not None:
        models = _model_ids(auth.ok.body)
        # Prefer a known free-plan chat id when present.
        preferred = next(
            (
                m
                for m in models
                if m
                in {
                    "openai/gpt-oss-20b",
                    "openai/gpt-oss-120b",
                    "qwen/qwen3.8-27b",
                    "groq/compound",
                    "groq/compound-mini",
                }
            ),
            models[0] if models else None,
        )
        probe_status = _probe_chat(
            url="https://api.groq.com/openai/v1/chat/completions",
            api_key=key or "",
            model=preferred,
            provider="groq",
        )
    return {
        "provider": "groq",
        "configured": configured,
        "authenticated_catalog": _catalog_section(auth),
        "accessible_model_ids": _model_ids(auth.ok.body) if auth.ok else [],
        "classification": classification,
        "probe": probe_status,
    }


def doctor_openrouter(*, probe: bool = False) -> dict[str, Any]:
    configured = _key_present("OPENROUTER_API_KEY")
    key = os.environ.get("OPENROUTER_API_KEY")
    # Public catalog works without a key.
    catalog = get_json_result(
        OPENROUTER_MODELS,
        api_key=key,
        provider="openrouter",
        endpoint_class="catalog",
    )
    classification = "WORKING" if catalog.ok else classify_cerebras_status(catalog.failure)
    return {
        "provider": "openrouter",
        "configured": configured,
        "catalog": _catalog_section(catalog),
        "accessible_model_ids": _model_ids(catalog.ok.body) if catalog.ok else [],
        "classification": classification,
        "probe": {"ran": False, "note": "openrouter probe not enabled in doctor yet"},
    }




def _env_doctor(provider: str, env_names: tuple[str, ...], catalog_url: str | None = None) -> dict[str, Any]:
    configured = any(_key_present(name) for name in env_names)
    key = None
    for name in env_names:
        if _key_present(name):
            key = os.environ.get(name)
            break
    catalog = {"status": None, "models": []}
    classification = "AUTH_INVALID"
    if configured and catalog_url:
        result = get_json_result(
            catalog_url,
            api_key=key,
            provider=provider,
            endpoint_class="auth_catalog",
        )
        catalog = _catalog_section(result)
        classification = "WORKING" if result.ok else classify_cerebras_status(result.failure)
    elif configured:
        classification = "UNKNOWN"
    return {
        "provider": provider,
        "configured": configured,
        "authenticated_catalog": catalog,
        "accessible_model_ids": catalog.get("models") or [],
        "classification": classification,
        "probe": {"ran": False},
    }


def doctor_vercel(*, probe: bool = False) -> dict[str, Any]:
    _ = probe
    return _env_doctor("vercel", ("AI_GATEWAY_API_KEY", "VERCEL_AI_GATEWAY_API_KEY"), "https://ai-gateway.vercel.sh/v1/models")


def doctor_gemini(*, probe: bool = False) -> dict[str, Any]:
    _ = probe
    return _env_doctor("gemini", ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_AI_API_KEY"), "https://generativelanguage.googleapis.com/v1beta/openai/models")


def doctor_mistral(*, probe: bool = False) -> dict[str, Any]:
    _ = probe
    return _env_doctor("mistral", ("MISTRAL_API_KEY",), "https://api.mistral.ai/v1/models")


def doctor_cohere(*, probe: bool = False) -> dict[str, Any]:
    _ = probe
    return _env_doctor("cohere", ("COHERE_API_KEY",), "https://api.cohere.com/compatibility/v1/models")


def doctor_nvidia_nim(*, probe: bool = False) -> dict[str, Any]:
    _ = probe
    return _env_doctor("nvidia-nim", ("NVIDIA_API_KEY", "NGC_API_KEY", "NIM_API_KEY"), "https://integrate.api.nvidia.com/v1/models")


def doctor_huggingface(*, probe: bool = False) -> dict[str, Any]:
    _ = probe
    return _env_doctor("huggingface", ("HF_TOKEN", "HUGGINGFACE_API_KEY", "HUGGINGFACEHUB_API_TOKEN"), "https://router.huggingface.co/v1/models")


def doctor_nous(*, probe: bool = False) -> dict[str, Any]:
    _ = probe
    try:
        from .providers.nous import _hermes_runtime, discover
    except Exception as exc:  # noqa: BLE001 — optional Hermes
        return {
            "provider": "nous",
            "configured": False,
            "classification": "UNKNOWN",
            "message": str(exc)[:240],
            "probe": {"ran": False},
        }
    runtime = _hermes_runtime()
    configured = bool(runtime and (runtime.get("api_key") or runtime.get("access_token")))
    offers = discover(free_only=True) if configured else []
    return {
        "provider": "nous",
        "configured": configured,
        "accessible_model_ids": [o.model for o in offers if o.model],
        "classification": "WORKING" if offers else ("AUTH_INVALID" if not configured else "UNKNOWN"),
        "probe": {"ran": False},
        "note": "uses Hermes resolve_nous_runtime_credentials; no OAuth copy in Kerdoios",
    }


_DOCTORS = {
    "cerebras": doctor_cerebras,
    "groq": doctor_groq,
    "openrouter": doctor_openrouter,
    "nous": doctor_nous,
    "vercel": doctor_vercel,
    "gemini": doctor_gemini,
    "mistral": doctor_mistral,
    "cohere": doctor_cohere,
    "nvidia-nim": doctor_nvidia_nim,
    "huggingface": doctor_huggingface,
}


def doctor_all(*, probe: bool = False) -> dict[str, Any]:
    reports = {name: fn(probe=False) for name, fn in _DOCTORS.items()}
    # Global doctor never probes by default even if probe=True was passed by mistake
    # for all; per-provider doctor <name> --probe remains explicit.
    _ = probe
    return {
        "providers": reports,
        "summary": {
            name: {
                "configured": report.get("configured"),
                "classification": report.get("classification"),
            }
            for name, report in reports.items()
        },
    }


def run_doctor(provider: str | None = None, *, probe: bool = False) -> dict[str, Any]:
    if not provider:
        return doctor_all(probe=False)
    key = provider.strip().lower()
    fn = _DOCTORS.get(key)
    if fn is None:
        return {
            "provider": key,
            "classification": "UNKNOWN",
            "message": f"no doctor registered for {key}",
            "known": sorted(_DOCTORS),
        }
    return fn(probe=probe)
