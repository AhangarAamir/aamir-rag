"""Chat model selection for OpenAI and OpenAI-compatible servers."""

from __future__ import annotations

import os
import unittest

import config
from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.knowledge.embedder.openai_like import OpenAILikeEmbedder
from agno.models.openai import OpenAIResponses
from agno.models.openai.like import OpenAILike


class LlmProviderTests(unittest.TestCase):
    def test_provider_errors_include_tracebacks(self) -> None:
        from agno.utils.log import agent_logger, log_tracebacks

        self.assertTrue(log_tracebacks)
        self.assertTrue(any(getattr(handler, "rich_tracebacks", False) for handler in agent_logger.handlers))
    def test_provider_aliases(self) -> None:
        self.assertEqual(config.normalize_llm_provider("openai"), "openai")
        self.assertEqual(config.normalize_llm_provider("OpenAI-Like"), "openailike")
        self.assertEqual(config.normalize_llm_provider("ollama"), "ollama")
        self.assertEqual(config.normalize_llm_provider("litellm"), "litellm")
        self.assertEqual(config.normalize_llm_provider("vllm"), "vllm")
        with self.assertRaises(RuntimeError):
            config.normalize_llm_provider("anthropic")

    def test_default_options_and_ollama_thinking(self) -> None:
        options = config.parse_openai_like_options(None)
        self.assertEqual(options, {"num_ctx": 32768, "num_batch": 2048})
        body = config.openai_like_request_body(options, ollama=True)
        self.assertEqual(body["options"], options)
        self.assertIs(body["think"], False)

    def test_non_ollama_keeps_options_without_think(self) -> None:
        self.assertFalse(
            config.is_ollama_backend("vllm", "", "http://127.0.0.1:8000/v1")
        )
        self.assertFalse(
            config.is_ollama_backend("litellm", "", "http://127.0.0.1:4000/v1")
        )
        self.assertFalse(
            config.is_ollama_backend("openailike", "vllm", "http://127.0.0.1:11434/v1")
        )
        body = config.openai_like_request_body(
            {"num_ctx": 32768, "num_batch": 2048},
            ollama=False,
        )
        self.assertNotIn("think", body)
        self.assertEqual(body["options"]["num_batch"], 2048)

    def test_embedding_provider_follows_chat_when_unset(self) -> None:
        self.assertEqual(config.resolve_embedding_provider(None, "ollama"), "ollama")
        self.assertEqual(config.resolve_embedding_provider("  ", "openai"), "openai")
        self.assertEqual(config.resolve_embedding_provider("ollama", "openai"), "ollama")
        self.assertEqual(config.resolve_embedding_provider("HuggingFace", "openai"), "huggingface")
        self.assertEqual(config.resolve_embedding_provider("hf", "ollama"), "huggingface")
        with self.assertRaises(RuntimeError):
            config.normalize_llm_provider("huggingface")

    def test_huggingface_model_loads_from_cache_without_downloading_again(self) -> None:
        calls: list[dict] = []
        stored = {"ready": False}

        def loader(model_id: str, **kwargs):
            calls.append(kwargs)
            if kwargs.get("local_files_only") and not stored["ready"]:
                raise OSError(
                    "Cannot find the requested files in the disk cache and "
                    "outgoing traffic has been disabled."
                )
            if kwargs.get("local_files_only") is False:
                stored["ready"] = True
            return model_id

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            cache = config.Path(tmp)
            config.load_cached_sentence_transformer("Alibaba-NLP/gte-base-en-v1.5", cache, loader)
            config.load_cached_sentence_transformer("Alibaba-NLP/gte-base-en-v1.5", cache, loader)

        self.assertEqual(
            [call["local_files_only"] for call in calls],
            [True, False, True],
        )
        self.assertTrue(all(call["cache_folder"] == tmp for call in calls))
        self.assertTrue(all(call["device"] == "cpu" for call in calls))

    def test_huggingface_cpu_wrapper_skips_prompt_offset(self) -> None:
        calls: list[dict] = []

        class _Model:
            default_prompt_name = "query"
            prompts = {"query": "query: "}

            def to(self, device: str) -> None:
                calls.append({"to": device})

            def float(self) -> None:
                calls.append({"float": True})

            def eval(self) -> None:
                calls.append({"eval": True})

            def encode(self, sentences, **kwargs):
                calls.append({"sentences": sentences, **kwargs})
                return [0.1, 0.2]

        wrapped = config.CpuSentenceTransformer(_Model())
        wrapped.encode("clause 10", prompt="query: ", normalize_embeddings=True)
        self.assertIsNone(wrapped._model.default_prompt_name)
        self.assertEqual(wrapped._model.prompts, {})
        encoded = calls[-1]
        self.assertEqual(encoded["sentences"], "clause 10")
        self.assertNotIn("prompt", encoded)
        self.assertNotIn("prompt_name", encoded)
        self.assertTrue(encoded["normalize_embeddings"])

    def test_ollama_embed_host_strips_v1(self) -> None:
        self.assertEqual(
            config.ollama_embed_host(
                "",
                "http://127.0.0.1:11434/v1",
                chat_is_ollama=True,
            ),
            "http://127.0.0.1:11434",
        )
        self.assertEqual(
            config.ollama_embed_host(
                "",
                "http://127.0.0.1:8000/v1",
                chat_is_ollama=False,
            ),
            "http://127.0.0.1:11434",
        )

    def test_ollama_detected_from_provider_or_url(self) -> None:
        self.assertTrue(config.is_ollama_backend("ollama", "", "http://gpu:9000/v1"))
        self.assertTrue(
            config.is_ollama_backend("openailike", "", "http://127.0.0.1:11434/v1")
        )
        self.assertTrue(
            config.is_ollama_backend("openailike", "ollama", "http://gpu:8000/v1")
        )

    def test_openai_model_stays_on_responses_api(self) -> None:
        with _patched_provider(provider="openai", model_id="gpt-4o"):
            model = config.build_chat_model()
        self.assertIsInstance(model, OpenAIResponses)
        self.assertEqual(model.id, "gpt-4o")
        self.assertEqual(model.timeout, 300)

    def test_ollama_model_sends_options_and_disables_thinking(self) -> None:
        with _patched_provider(
            provider="ollama",
            model_id="qwen3:32b",
            api_key="",
            base_url="http://127.0.0.1:11434/v1",
            embedding_model="nomic-embed-text",
            embedding_dimensions=768,
        ):
            model = config.build_chat_model()
            embedder = config.build_embedder()
        self.assertIsInstance(model, OpenAILike)
        self.assertEqual(model.provider, "OpenAILike")
        self.assertEqual(model.id, "qwen3:32b")
        self.assertEqual(model.api_key, "not-provided")
        self.assertEqual(model.base_url, "http://127.0.0.1:11434/v1")
        self.assertEqual(model.timeout, 300)
        self.assertEqual(embedder.timeout, 300)
        extra = model.get_request_params()["extra_body"]
        self.assertEqual(extra["options"], {"num_ctx": 32768, "num_batch": 2048})
        self.assertIs(extra["think"], False)
        self.assertIsInstance(embedder, OllamaEmbedder)
        self.assertEqual(embedder.id, "nomic-embed-text")
        self.assertEqual(embedder.dimensions, 768)
        self.assertEqual(embedder.host, "http://127.0.0.1:11434")

    def test_openai_chat_can_use_an_ollama_embedding_model(self) -> None:
        with _patched_provider(
            provider="openai",
            model_id="gpt-4o",
            embedding_provider="ollama",
            embedding_model="mxbai-embed-large",
            embedding_base_url="http://gpu:11434/v1",
            embedding_dimensions=1024,
        ):
            model = config.build_chat_model()
            embedder = config.build_embedder()
        self.assertIsInstance(model, OpenAIResponses)
        self.assertIsInstance(embedder, OllamaEmbedder)
        self.assertEqual(embedder.id, "mxbai-embed-large")
        self.assertEqual(embedder.host, "http://gpu:11434")
        self.assertEqual(embedder.dimensions, 1024)

    def test_openai_embeddings_stay_on_openai(self) -> None:
        with _patched_provider(provider="openai", model_id="gpt-4o"):
            embedder = config.build_embedder()
        self.assertIsInstance(embedder, OpenAIEmbedder)
        self.assertEqual(embedder.id, "text-embedding-3-large")
        self.assertEqual(embedder.client_params["timeout"], 300)

    def test_litellm_model_omits_think(self) -> None:
        with _patched_provider(
            provider="litellm",
            model_id="openai/gpt-4o",
            api_key="sk-local",
            base_url="http://127.0.0.1:4000/v1",
            embedding_base_url="http://127.0.0.1:4000/v1",
            embedding_dimensions=768,
        ):
            model = config.build_chat_model()
            embedder = config.build_embedder()
        extra = model.get_request_params()["extra_body"]
        self.assertNotIn("think", extra)
        self.assertEqual(model.api_key, "sk-local")
        self.assertEqual(model.timeout, 300)
        self.assertEqual(embedder.client_params["timeout"], 300)
        self.assertIsInstance(embedder, OpenAILikeEmbedder)
        self.assertEqual(embedder.dimensions, 768)
        self.assertEqual(embedder.base_url, "http://127.0.0.1:4000/v1")

    def test_openailike_requires_base_url_not_a_real_key(self) -> None:
        previous = os.environ.get("OPENAI_API_KEY")
        try:
            with _patched_provider(
                provider="openailike",
                model_id="my-model",
                api_key="",
                base_url="",
            ):
                with self.assertRaises(RuntimeError):
                    config.require_api_key()
            with _patched_provider(
                provider="vllm",
                model_id="my-model",
                api_key="",
                base_url="http://127.0.0.1:8000/v1",
            ):
                config.require_api_key()
                self.assertEqual(os.environ["OPENAI_API_KEY"], "not-provided")
            with _patched_provider(
                provider="openai",
                model_id="gpt-4o",
                api_key="sk-chat",
                embedding_provider="ollama",
                embedding_model="nomic-embed-text",
            ):
                with self.assertRaises(RuntimeError):
                    config.require_api_key()
            with _patched_provider(
                provider="ollama",
                model_id="qwen3.5:2b",
                api_key="",
                base_url="http://127.0.0.1:11434/v1",
                embedding_provider="huggingface",
                embedding_model="Alibaba-NLP/gte-base-en-v1.5",
            ):
                with self.assertRaises(RuntimeError) as missing_dims:
                    config.require_api_key()
                self.assertIn("EMBEDDING_DIMENSIONS", str(missing_dims.exception))
        finally:
            if previous is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = previous


class _patched_provider:
    def __init__(
        self,
        *,
        provider: str,
        model_id: str,
        api_key: str = "sk-test",
        base_url: str = "",
        backend: str = "",
        embedding_provider: str | None = None,
        embedding_model: str = "text-embedding-3-large",
        embedding_base_url: str = "",
        embedding_dimensions: int | None = None,
    ) -> None:
        self.values = {
            "LLM_PROVIDER": provider,
            "OPENAI_MODEL": model_id,
            "OPENAI_API_KEY": api_key,
            "OPENAI_BASE_URL": base_url,
            "OPENAI_LIKE_BACKEND": backend,
            "OPENAI_LIKE_OPTIONS": {"num_ctx": 32768, "num_batch": 2048},
            "EMBEDDING_PROVIDER": embedding_provider or provider,
            "EMBEDDING_MODEL": embedding_model,
            "EMBEDDING_API_KEY": "",
            "EMBEDDING_BASE_URL": embedding_base_url,
            "EMBEDDING_DIMENSIONS": embedding_dimensions,
        }
        self.saved: dict[str, object] = {}

    def __enter__(self) -> None:
        for name, value in self.values.items():
            self.saved[name] = getattr(config, name)
            setattr(config, name, value)

    def __exit__(self, *exc: object) -> None:
        for name, value in self.saved.items():
            setattr(config, name, value)


if __name__ == "__main__":
    unittest.main()
