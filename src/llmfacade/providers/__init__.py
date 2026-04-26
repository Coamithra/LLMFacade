from __future__ import annotations

PROVIDER_REGISTRY: dict[str, tuple[str, str]] = {
    "anthropic": ("llmfacade.providers.anthropic", "AnthropicProvider"),
    "openai": ("llmfacade.providers.openai", "OpenAIProvider"),
    "google": ("llmfacade.providers.google", "GoogleProvider"),
    "gemini": ("llmfacade.providers.google", "GoogleProvider"),
    "ollama": ("llmfacade.providers.ollama", "OllamaProvider"),
}
