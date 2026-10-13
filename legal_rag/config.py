"""Shared paths and model settings for ingest + query agents."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

DATA_DIR = ROOT_DIR / "data"
INCOMING_DIR = DATA_DIR / "incoming"
VECTOR_DB_DIR = DATA_DIR / "vectordb"
COLLECTION_NAME = "legal_contracts"

INCOMING_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

from legal_rag.knowledge.catalog import ensure_incoming_layout

ensure_incoming_layout(INCOMING_DIR)

_PROVIDER_ALIASES = {
    "openai": "openai",
    "openailike": "openailike",
    "openai-like": "openailike",
    "openai_like": "openailike",
    "ollama": "ollama",
    "litellm": "litellm",
    "vllm": "vllm",
}
_EMBEDDING_PROVIDER_ALIASES = {
    **_PROVIDER_ALIASES,
    "huggingface": "huggingface",
    "hugging-face": "huggingface",
    "hugging_face": "huggingface",
    "hf": "huggingface",
    "sentence-transformers": "huggingface",
    "sentence_transformers": "huggingface",
    "sentencetransformer": "huggingface",
}
_DEFAULT_OPENAI_LIKE_OPTIONS = {"num_ctx": 32768, "num_batch": 2048}


def normalize_llm_provider(raw: str | None) -> str:
    """openai keeps the Responses API. ollama, litellm, and vllm use OpenAILike."""
    value = (raw or "openai").strip().lower()
    provider = _PROVIDER_ALIASES.get(value)
    if provider is None:
        raise RuntimeError(
            "LLM_PROVIDER must be openai, openailike, ollama, litellm, or vllm."
        )
    return provider


def parse_openai_like_options(raw: str | None) -> dict[str, Any]:
    if raw is None or not raw.strip():
        return dict(_DEFAULT_OPENAI_LIKE_OPTIONS)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("OPENAI_LIKE_OPTIONS must be a JSON object.") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("OPENAI_LIKE_OPTIONS must be a JSON object.")
    return parsed


def is_ollama_backend(provider: str, backend: str | None, base_url: str) -> bool:
    kind = (backend or "").strip().lower()
    if kind == "ollama" or provider == "ollama":
        return True
    if kind in {"litellm", "vllm"} or provider in {"litellm", "vllm"}:
        return False
    url = (base_url or "").lower()
    return "ollama" in url or ":11434" in url


def openai_like_request_body(options: dict[str, Any], *, ollama: bool) -> dict[str, Any]:
    body: dict[str, Any] = {}
    if options:
        body["options"] = options
    if ollama:
        body["think"] = False
    return body


def resolve_embedding_provider(raw: str | None, llm_provider: str) -> str:
    """Empty embedding provider follows the chat provider."""
    if raw is None or not raw.strip():
        return llm_provider
    provider = _EMBEDDING_PROVIDER_ALIASES.get(raw.strip().lower())
    if provider is None:
        raise RuntimeError(
            "EMBEDDING_PROVIDER must be openai, openailike, ollama, litellm, "
            "vllm, or huggingface."
        )
    return provider


def strip_openai_v1(url: str) -> str:
    value = url.strip().rstrip("/")
    if value.endswith("/v1"):
        return value[:-3]
    return value


_CACHE_MISS_MARKERS = (
    "local_files_only",
    "cached snapshot",
    "cached files",
    "outgoing traffic has been disabled",
    "couldn't connect",
    "cannot find the requested files",
    "not the path to a directory",
)


def is_missing_cached_model(exc: BaseException) -> bool:
    if type(exc).__name__ == "LocalEntryNotFoundError":
        return True
    text = str(exc).lower()
    return any(marker in text for marker in _CACHE_MISS_MARKERS)


def huggingface_model_cache_dir() -> Path:
    return DATA_DIR / "models"


def _position_id_values(buffer: Any) -> list[int] | None:
    try:
        flat = buffer.detach().reshape(-1)
        values = flat.tolist()
    except Exception:
        return None
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        return None
    return values


def _position_ids_are_corrupt(buffer: Any) -> bool:
    """GTE stores position_ids as a non-persistent buffer, so it is not in the checkpoint.

    Transformers 5 leaves that buffer uninitialized. encode then does rope[position_ids]
    and the garbage index is reported as out of bounds for the chunk length.
    """
    numel = getattr(buffer, "numel", None)
    if not callable(numel) or numel() == 0:
        return True
    values = _position_id_values(buffer)
    if values is None:
        return True
    width = len(values)
    return any(value < 0 or value >= width for value in values)


def repair_gte_position_ids(model: Any, torch_module: Any | None = None) -> int:
    """Rewrite corrupt position_ids buffers. Returns how many were repaired."""
    torch = torch_module
    if torch is None:
        try:
            import torch as imported
        except ImportError:
            return 0
        torch = imported
    named_buffers = getattr(model, "named_buffers", None)
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_buffers) or not callable(named_modules):
        return 0
    modules = dict(named_modules())
    max_positions = 0
    for module in modules.values():
        config = getattr(module, "config", None)
        width = getattr(config, "max_position_embeddings", None)
        if isinstance(width, int) and width > max_positions:
            max_positions = width
    if max_positions <= 0:
        max_positions = 8192
    repaired = 0
    for qualified, buffer in list(named_buffers()):
        if not qualified.endswith("position_ids") or not _position_ids_are_corrupt(buffer):
            continue
        parent_name, _, attr = qualified.rpartition(".")
        parent = modules.get(parent_name)
        register = getattr(parent, "register_buffer", None)
        if parent is None or not callable(register):
            continue
        width = buffer.numel() or max_positions
        shape = tuple(buffer.shape) if buffer.numel() else (1, width)
        device = getattr(buffer, "device", None)
        fresh = torch.arange(width, device=device, dtype=torch.long).reshape(shape)
        register(attr, fresh, persistent=False)
        repaired += 1
    return repaired


class CpuSentenceTransformer:
    """Run a Hugging Face embedder on CPU in float32, without a prompt offset.

    Alibaba GTE keeps position ids in a buffer that is not saved with the weights.
    Transformers 5 does not restore it, so encode indexes the sequence with garbage
    and raises "index ... is out of bounds for dimension 0".
    """

    def __init__(self, model: Any) -> None:
        move = getattr(model, "to", None)
        if callable(move):
            move("cpu")
        as_float = getattr(model, "float", None)
        if callable(as_float):
            as_float()
        mark_eval = getattr(model, "eval", None)
        if callable(mark_eval):
            mark_eval()
        if hasattr(model, "default_prompt_name"):
            model.default_prompt_name = None
        prompts = getattr(model, "prompts", None)
        if isinstance(prompts, dict):
            model.prompts = {}
        repair_gte_position_ids(model)
        self._model = model

    def encode(self, sentences: Any, prompt: str | None = None, normalize_embeddings: bool = False, **kwargs: Any):
        del prompt
        kwargs.pop("prompt_name", None)
        try:
            return self._model.encode(
                sentences,
                normalize_embeddings=normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False,
                **kwargs,
            )
        except TypeError:
            return self._model.encode(sentences, normalize_embeddings=normalize_embeddings)

    def to(self, device: str) -> "CpuSentenceTransformer":
        del device
        move = getattr(self._model, "to", None)
        if callable(move):
            move("cpu")
        return self


def _cpu_model_kwargs() -> dict[str, Any]:
    """GTE v1.5 packs tokens by default. On CPU that index is corrupt, so every chunk fails."""
    packed: dict[str, Any] = {
        "config_kwargs": {
            "unpad_inputs": False,
            "use_memory_efficient_attention": False,
        },
    }
    try:
        import torch
    except ImportError:
        return packed
    packed["model_kwargs"] = {"torch_dtype": torch.float32}
    return packed


def load_cached_sentence_transformer(model_id: str, cache_dir: Path, loader: Any):
    """Use the files already in cache_dir. Download only when they are absent."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    def _construct(*, local_files_only: bool) -> Any:
        folder = str(cache_dir)
        cpu = {"cache_folder": folder, "device": "cpu", **_cpu_model_kwargs()}
        attempts = [
            {**cpu, "local_files_only": local_files_only, "trust_remote_code": True},
            {**cpu, "local_files_only": local_files_only},
            {"cache_folder": folder, "device": "cpu"},
        ]
        last_type_error: TypeError | None = None
        for kwargs in attempts:
            try:
                return loader(model_id, **kwargs)
            except TypeError as exc:
                last_type_error = exc
        if last_type_error is None:
            raise RuntimeError(f"Could not load Hugging Face model {model_id}.")
        raise last_type_error

    try:
        return _construct(local_files_only=True)
    except TypeError:
        return loader(model_id, cache_folder=str(cache_dir), device="cpu")
    except Exception as exc:
        if not is_missing_cached_model(exc):
            raise
        return _construct(local_files_only=False)


def ollama_embed_host(
    embedding_base_url: str,
    chat_base_url: str,
    *,
    chat_is_ollama: bool,
) -> str:
    """Native Ollama host. A chat /v1 URL is reused only when chat is Ollama too."""
    if embedding_base_url:
        return strip_openai_v1(embedding_base_url)
    if chat_is_ollama and chat_base_url:
        return strip_openai_v1(chat_base_url)
    return "http://127.0.0.1:11434"


# Chat and embedding HTTP calls. Covers connect and read, including a busy vLLM server.
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "300"))

LLM_PROVIDER = normalize_llm_provider(os.getenv("LLM_PROVIDER"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip()
OPENAI_LIKE_BACKEND = os.getenv("OPENAI_LIKE_BACKEND", "").strip().lower()
OPENAI_LIKE_OPTIONS = parse_openai_like_options(os.getenv("OPENAI_LIKE_OPTIONS"))
EMBEDDING_PROVIDER = resolve_embedding_provider(
    os.getenv("EMBEDDING_PROVIDER"),
    LLM_PROVIDER,
)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "").strip()
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "").strip()
_embedding_dimensions = os.getenv("EMBEDDING_DIMENSIONS", "").strip()
EMBEDDING_DIMENSIONS = int(_embedding_dimensions) if _embedding_dimensions else None
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "agentic").strip().lower()
CHUNKING_MAX_SIZE = int(os.getenv("CHUNKING_MAX_SIZE", "4000"))
CHUNKER_ASYNC = os.getenv("CHUNKER_ASYNC", "true").strip().lower() in {"1", "true", "yes", "on"}
CHUNKER_CONCURRENCY = max(1, int(os.getenv("CHUNKER_CONCURRENCY", "8")))
SEARCH_MAX_RESULTS = int(os.getenv("SEARCH_MAX_RESULTS", "12"))
QUERY_SESSION_ID = os.getenv("QUERY_SESSION_ID", "legal-query-default")
NUM_HISTORY_RUNS = int(os.getenv("NUM_HISTORY_RUNS", "5"))
SESSIONS_DB_FILE = DATA_DIR / "sessions.db"

HOST = os.getenv("HOST") or os.getenv("AGENT_OS_HOST") or "127.0.0.1"
PORT = int(os.getenv("PORT") or os.getenv("AGENT_OS_PORT") or "8000")

_DEFAULT_CORS_ORIGINS = (
    "https://uracs.enp.ril.com",
    "https://uracs.ucg.ril.com",
)


def _parse_cors_origins(raw: str | None) -> list[str]:
    """Comma-separated origins. Bare hostnames get an https:// prefix."""
    values = _DEFAULT_CORS_ORIGINS if raw is None else raw.split(",")
    origins: list[str] = []
    seen: set[str] = set()
    for value in values:
        origin = value.strip().rstrip("/")
        if not origin:
            continue
        if "://" not in origin:
            origin = f"https://{origin}"
        if origin not in seen:
            seen.add(origin)
            origins.append(origin)
    return origins


CORS_ALLOWED_ORIGINS = _parse_cors_origins(os.getenv("CORS_ALLOWED_ORIGINS"))


def _quiet_openai_httpx_finalizers() -> None:
    """OpenAI 3.x wraps httpx2. Client GC calls __del__ -> is_closed -> _state.

    If the wrapper is already half-destroyed, _state is gone and Python prints
    'Exception ignored'. Searches still succeed. Swallow that destructor noise.
    """
    try:
        from openai._base_client import AsyncHttpxClientWrapper, SyncHttpxClientWrapper
    except ImportError:
        return

    def _safe_sync_del(self) -> None:
        try:
            if getattr(self, "_state", None) is None or self.is_closed:
                return
            self.close()
        except Exception:
            pass

    def _safe_async_del(self) -> None:
        try:
            if getattr(self, "_state", None) is None or self.is_closed:
                return
            import asyncio

            loop = asyncio.get_running_loop()
            loop.create_task(self.aclose())
        except Exception:
            pass

    SyncHttpxClientWrapper.__del__ = _safe_sync_del  # type: ignore[method-assign]
    AsyncHttpxClientWrapper.__del__ = _safe_async_del  # type: ignore[method-assign]


_quiet_openai_httpx_finalizers()


def enable_provider_error_tracebacks() -> None:
    """Print the full stack when a provider or agent run fails.

    Agno otherwise logs only str(exception). vLLM errors often have an empty
    message, which shows up as "Error from OpenAI API:" with nothing after it.
    """
    from agno.utils.log import agent_logger, set_log_tracebacks, team_logger, workflow_logger

    set_log_tracebacks(True)
    for current in (agent_logger, team_logger, workflow_logger):
        for handler in getattr(current, "handlers", ()):
            if hasattr(handler, "rich_tracebacks"):
                handler.rich_tracebacks = True


enable_provider_error_tracebacks()


def require_api_key() -> None:
    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is missing. Copy .env.example to .env and set the key."
            )
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    else:
        if not OPENAI_BASE_URL:
            raise RuntimeError(
                "OPENAI_BASE_URL is required when LLM_PROVIDER is "
                "openailike, ollama, litellm, or vllm."
            )
        if not OPENAI_MODEL:
            raise RuntimeError("OPENAI_MODEL is required.")
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY or "not-provided"

    if EMBEDDING_PROVIDER == "openai":
        if not (EMBEDDING_API_KEY or OPENAI_API_KEY):
            raise RuntimeError(
                "OPENAI_API_KEY or EMBEDDING_API_KEY is required when "
                "EMBEDDING_PROVIDER is openai."
            )
        return
    if not EMBEDDING_MODEL:
        raise RuntimeError("EMBEDDING_MODEL is required.")
    if EMBEDDING_PROVIDER in {"ollama", "huggingface"}:
        if EMBEDDING_DIMENSIONS is None:
            if EMBEDDING_PROVIDER == "ollama":
                hint = "nomic-embed-text is 768, all-minilm is 384."
            else:
                hint = (
                    "sentence-transformers/all-MiniLM-L6-v2 is 384, "
                    "Alibaba-NLP/gte-base-en-v1.5 is 768, "
                    "Alibaba-NLP/gte-large-en-v1.5 is 1024."
                )
            raise RuntimeError(
                "EMBEDDING_DIMENSIONS is required when EMBEDDING_PROVIDER is "
                f"{EMBEDDING_PROVIDER}. It must match that model ({hint})"
            )
        return
    if not EMBEDDING_BASE_URL and not (LLM_PROVIDER != "openai" and OPENAI_BASE_URL):
        raise RuntimeError(
            "EMBEDDING_BASE_URL is required when EMBEDDING_PROVIDER is "
            "openailike, litellm, or vllm."
        )


def build_chat_model():
    """OpenAI uses the Responses API. Compatible servers use OpenAILike."""
    if LLM_PROVIDER == "openai":
        from agno.models.openai import OpenAIResponses

        return OpenAIResponses(id=OPENAI_MODEL, timeout=REQUEST_TIMEOUT_SECONDS)

    from agno.models.openai.like import OpenAILike

    ollama = is_ollama_backend(LLM_PROVIDER, OPENAI_LIKE_BACKEND, OPENAI_BASE_URL)
    extra_body = openai_like_request_body(OPENAI_LIKE_OPTIONS, ollama=ollama)
    return OpenAILike(
        id=OPENAI_MODEL,
        api_key=OPENAI_API_KEY or "not-provided",
        base_url=OPENAI_BASE_URL,
        provider="OpenAILike",
        timeout=REQUEST_TIMEOUT_SECONDS,
        extra_body=extra_body or None,
    )


def build_embedder():
    """Embedding provider is independent. Ollama can serve its own embedding model."""
    if EMBEDDING_PROVIDER == "openai":
        from agno.knowledge.embedder.openai import OpenAIEmbedder

        kwargs: dict[str, Any] = {
            "id": EMBEDDING_MODEL,
            "client_params": {"timeout": REQUEST_TIMEOUT_SECONDS},
        }
        key = EMBEDDING_API_KEY or OPENAI_API_KEY
        if key:
            kwargs["api_key"] = key
        if EMBEDDING_BASE_URL:
            kwargs["base_url"] = EMBEDDING_BASE_URL
        if EMBEDDING_DIMENSIONS is not None:
            kwargs["dimensions"] = EMBEDDING_DIMENSIONS
        return OpenAIEmbedder(**kwargs)

    if EMBEDDING_PROVIDER == "ollama":
        try:
            from agno.knowledge.embedder.ollama import OllamaEmbedder
        except ImportError as exc:
            raise RuntimeError(
                "Ollama embeddings need the ollama package. Install it with `uv add ollama`."
            ) from exc
        if EMBEDDING_DIMENSIONS is None:
            raise RuntimeError(
                "EMBEDDING_DIMENSIONS is required when EMBEDDING_PROVIDER is ollama. "
                "It must match that model (nomic-embed-text is 768)."
            )
        chat_is_ollama = is_ollama_backend(
            LLM_PROVIDER,
            OPENAI_LIKE_BACKEND,
            OPENAI_BASE_URL,
        )
        return OllamaEmbedder(
            id=EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIMENSIONS,
            timeout=REQUEST_TIMEOUT_SECONDS,
            host=ollama_embed_host(
                EMBEDDING_BASE_URL,
                OPENAI_BASE_URL,
                chat_is_ollama=chat_is_ollama,
            ),
        )

    if EMBEDDING_PROVIDER == "huggingface":
        if EMBEDDING_DIMENSIONS is None:
            raise RuntimeError(
                "EMBEDDING_DIMENSIONS is required when EMBEDDING_PROVIDER is huggingface."
            )
        try:
            from sentence_transformers import SentenceTransformer

            from agno.knowledge.embedder.sentence_transformer import SentenceTransformerEmbedder
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face embeddings need sentence-transformers. "
                "Install it with `uv add sentence-transformers`."
            ) from exc
        client = CpuSentenceTransformer(
            load_cached_sentence_transformer(
                EMBEDDING_MODEL,
                huggingface_model_cache_dir(),
                SentenceTransformer,
            )
        )
        return SentenceTransformerEmbedder(
            id=EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIMENSIONS,
            sentence_transformer_client=client,
        )

    from agno.knowledge.embedder.openai_like import OpenAILikeEmbedder

    base_url = EMBEDDING_BASE_URL or OPENAI_BASE_URL
    kwargs = {
        "id": EMBEDDING_MODEL,
        "api_key": EMBEDDING_API_KEY or OPENAI_API_KEY or "not-provided",
        "base_url": base_url,
        "client_params": {"timeout": REQUEST_TIMEOUT_SECONDS},
    }
    if EMBEDDING_DIMENSIONS is not None:
        kwargs["dimensions"] = EMBEDDING_DIMENSIONS
    return OpenAILikeEmbedder(**kwargs)
