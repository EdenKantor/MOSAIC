"""Provider selection belongs to composition; core code has no provider model names."""

from mosaic.core.config import ModelConfig
from mosaic.providers.base import ModelProvider
from mosaic.providers.fake import FakeModelProvider
from mosaic.providers.groq import GroqModelProvider
from mosaic.providers.ollama import OllamaModelProvider


def provider_factory(config: ModelConfig) -> ModelProvider:
    if config.provider == "fake":
        return FakeModelProvider(config.policy, config.failed_calls)
    if config.provider == "ollama":
        return OllamaModelProvider(config.timeout_seconds)
    if config.provider == "groq":
        return GroqModelProvider(config.timeout_seconds)
    raise ValueError("Unsupported provider configuration")
