import asyncio
import json
import logging
import shutil
import subprocess
import time
from typing import AsyncGenerator, Dict, List, Optional, Tuple

import httpx

from app.config import Settings
from app.schemas import ChatRequest, HealthCheckResult

logger = logging.getLogger(__name__)


class OllamaProxyError(Exception):
    def __init__(self, message: str, attempts: Optional[List[Dict[str, str]]] = None):
        super().__init__(message)
        self.attempts = attempts or []


def _as_messages(payload: ChatRequest) -> List[Dict[str, str]]:
    if payload.messages:
        return [{"role": message.role, "content": message.content} for message in payload.messages]
    return [{"role": "user", "content": payload.prompt or ""}]


def _models_for_mode(settings: Settings, payload: ChatRequest) -> List[str]:
    if payload.model:
        return [payload.model]

    remote_models: List[str]
    if payload.mode == "code":
        remote_models = [settings.code_model, settings.remote_primary_model]
    elif payload.mode == "vision":
        remote_models = [settings.vision_model, settings.remote_primary_model]
    else:
        remote_models = [settings.remote_primary_model]

    seen = set()
    return [model for model in remote_models if model and not (model in seen or seen.add(model))]


def _keep_alive_value(raw: str) -> object:
    try:
        return int(raw)
    except ValueError:
        return raw


async def _call_ollama_chat(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    retries: int,
    keep_alive: Optional[str] = None,
    options: Optional[Dict[str, object]] = None,
) -> Tuple[str, str]:
    endpoint = f"{base_url.rstrip('/')}/api/chat"
    payload: Dict[str, object] = {"model": model, "messages": messages, "stream": False}
    if keep_alive is not None:
        payload["keep_alive"] = _keep_alive_value(keep_alive)
    if options:
        payload["options"] = options
    last_error: Optional[str] = None

    for attempt in range(retries + 1):
        try:
            response = await client.post(endpoint, json=payload)
            response.raise_for_status()
            data = response.json()
            content = data.get("message", {}).get("content")
            if not content:
                raise OllamaProxyError("Empty response content from Ollama.")
            return content, model
        except (httpx.TimeoutException, httpx.HTTPError, OllamaProxyError) as exc:
            if isinstance(exc, httpx.TimeoutException):
                last_error = "timeout"
            elif isinstance(exc, httpx.HTTPStatusError):
                last_error = f"http_{exc.response.status_code}"
            elif isinstance(exc, httpx.HTTPError):
                last_error = "http_error"
            else:
                last_error = "empty_response"
            logger.warning(
                "Ollama call failed", extra={"endpoint": endpoint, "model": model, "attempt": attempt + 1}
            )
            if attempt < retries:
                await asyncio.sleep(0.2 * (attempt + 1))

    raise OllamaProxyError(last_error or "Unknown Ollama failure")


async def route_chat_request(settings: Settings, payload: ChatRequest) -> Tuple[str, str, List[Dict[str, str]]]:
    messages = _as_messages(payload)
    errors: List[Dict[str, str]] = []

    timeout = httpx.Timeout(settings.request_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for model in _models_for_mode(settings, payload):
            for remote_url in settings.remote_urls():
                try:
                    output, used_model = await _call_ollama_chat(
                        client=client,
                        base_url=remote_url,
                        model=model,
                        messages=messages,
                        retries=settings.max_retries,
                        keep_alive=settings.ollama_keep_alive,
                        options=payload.options,
                    )
                    return output, used_model, [{"source": "remote", "url": remote_url}]
                except OllamaProxyError as exc:
                    errors.append({"source": "remote", "url": remote_url, "model": model, "error": str(exc)})

        local_model = payload.model or settings.local_fallback_model
        try:
            output, used_model = await _call_ollama_chat(
                client=client,
                base_url=settings.ollama_local_base_url,
                model=local_model,
                messages=messages,
                retries=settings.max_retries,
                keep_alive=settings.ollama_keep_alive,
                options=payload.options,
            )
            return output, used_model, [{"source": "local", "url": settings.ollama_local_base_url}]
        except OllamaProxyError as exc:
            errors.append(
                {
                    "source": "local",
                    "url": settings.ollama_local_base_url,
                    "model": local_model,
                    "error": str(exc),
                }
            )

    raise OllamaProxyError("All configured Ollama routes failed.", attempts=errors)


async def _stream_ollama_chat(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    keep_alive: Optional[str],
) -> AsyncGenerator[str, None]:
    endpoint = f"{base_url.rstrip('/')}/api/chat"
    payload: Dict[str, object] = {"model": model, "messages": messages, "stream": True}
    if keep_alive is not None:
        payload["keep_alive"] = _keep_alive_value(keep_alive)

    async with client.stream("POST", endpoint, json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            token = chunk.get("message", {}).get("content", "")
            if token:
                yield token
            if chunk.get("done"):
                break


async def stream_chat_request(settings: Settings, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
    """Streaming counterpart to route_chat_request: same primary -> local-fallback
    order, but yields token deltas as they arrive instead of waiting for the
    full response. Fallback only applies before any token has been yielded for
    an attempt -- once streaming has started, a mid-stream failure is surfaced
    rather than silently retried on a different model (which would garble or
    duplicate output a caller may have already started using)."""
    candidates: List[Tuple[str, str]] = [
        (settings.remote_primary_model, remote_url) for remote_url in settings.remote_urls()
    ]
    candidates.append((settings.local_fallback_model, settings.ollama_local_base_url))

    timeout = httpx.Timeout(settings.request_timeout_seconds)
    last_error: Optional[Exception] = None

    async with httpx.AsyncClient(timeout=timeout) as client:
        for model, base_url in candidates:
            started = False
            try:
                async for token in _stream_ollama_chat(client, base_url, model, messages, settings.ollama_keep_alive):
                    started = True
                    yield token
                return
            except (httpx.TimeoutException, httpx.HTTPError) as exc:
                last_error = exc
                if started:
                    raise OllamaProxyError(f"Streaming interrupted mid-response: {exc}") from exc
                continue

    raise OllamaProxyError(f"All configured Ollama routes failed for streaming chat: {last_error}")


async def check_ollama_reachability(url: str, timeout_seconds: float) -> HealthCheckResult:
    start = time.perf_counter()
    endpoint = f"{url.rstrip('/')}/api/tags"
    timeout = httpx.Timeout(timeout_seconds)

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.get(endpoint)
            response.raise_for_status()
            latency_ms = int((time.perf_counter() - start) * 1000)
            return HealthCheckResult(url=url, reachable=True, latency_ms=latency_ms)
        except (httpx.TimeoutException, httpx.HTTPError) as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            if isinstance(exc, httpx.TimeoutException):
                detail = "timeout"
            elif isinstance(exc, httpx.HTTPStatusError):
                detail = f"http_{exc.response.status_code}"
            else:
                detail = "unreachable"
            return HealthCheckResult(url=url, reachable=False, latency_ms=latency_ms, detail=detail)


def detect_cuda() -> Dict[str, object]:
    has_nvidia_smi = bool(shutil.which("nvidia-smi"))
    nvidia_smi_output = None
    cuda_available = False

    if has_nvidia_smi:
        try:
            result = subprocess.run(
                ["nvidia-smi", "-L"],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
            nvidia_smi_output = result.stdout.strip() or result.stderr.strip() or "nvidia-smi detected"
            cuda_available = True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            nvidia_smi_output = "unavailable"

    jetson_hint = shutil.which("tegrastats") is not None

    return {
        "cuda_available": cuda_available,
        "has_nvidia_smi": has_nvidia_smi,
        "jetson_hint": jetson_hint,
        "nvidia_smi": nvidia_smi_output,
    }
