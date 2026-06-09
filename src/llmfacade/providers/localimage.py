from __future__ import annotations

import asyncio
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llmfacade.exceptions import (
    AuthenticationError,
    ProviderError,
    ProviderNotInstalledError,
    RateLimitError,
    UnsupportedFeature,
)
from llmfacade.models import ImageResult, ReferenceImage, _apply_save_dir
from llmfacade.provider import Provider
from llmfacade.providers._openai_images import (
    build_edit_kwargs,
    build_generate_kwargs,
    media_type_for,
    parse_images_response,
)
from llmfacade.providers._sd_launch import (
    _DIR_PATH_KNOBS,
    _FILE_PATH_KNOBS,
    _SdLaunchEntry,
    default_provider_sd_defaults,
    derive_image_model_id,
)
from llmfacade.providers._sd_lifecycle import _SdServerSupervisor

if TYPE_CHECKING:
    from llmfacade.facade import LLM
    from llmfacade.image import ImageModel


class LocalImageProvider(Provider):
    """Local image generation against an OpenAI-compatible image server.

    Targets any server that speaks the OpenAI Images API — stable-diffusion.cpp's
    ``sd-server`` (the image analog of ``llama-server``; Flux / SD / SDXL /
    Qwen-Image), LocalAI, etc. — reusing the ``openai`` SDK as transport, exactly
    as the llamacpp provider reuses it for chat.

    **Two modes**, decided by the presence of ``base_url`` at construction:

    * **External** (``base_url=...``): talk to a server the user is already
      running, e.g. ``LLM().new_provider("localimage",
      base_url="http://127.0.0.1:1234/v1")``. No process management. Any
      ``SD_LAUNCH_KNOBS`` value passed in this mode raises ``UnsupportedFeature``.
    * **Managed** (``base_url=None``): the provider owns an ``sd-server``
      subprocess. ``new_image_model(model=.../diffusion_model=..., ...)`` registers
      a launch entry; the first ``generate_image`` for a model lazily spawns
      sd-server. Because sd-server is strictly single-model-per-process, the
      supervisor keeps **one** process alive and *swaps on demand* — a
      ``generate_image`` for a different registered model tears the running process
      down and spawns the new one. Use ``provider.shutdown()`` for explicit
      teardown (atexit + signal handlers also call it).

    No auth is required in either mode (``api_key`` is optional; a placeholder is
    sent to satisfy the SDK).

    Server-specific knobs that have no OpenAI equivalent (steps, cfg, sampler,
    seed, ...) go through ``extra=`` — forwarded as the SDK's ``extra_body`` for
    servers that read JSON fields (LocalAI). For ``sd-server`` specifically, those
    are instead embedded in the prompt via its
    ``<sd_cpp_extra_args>{...}</sd_cpp_extra_args>`` syntax.
    """

    NAME = "localimage"
    API_KEY_ENV = None
    SUPPORTS: frozenset[str] = frozenset({"image_generation"})

    def __init__(
        self,
        *,
        manager: LLM | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        log_dir: Any | None = None,
        cache_dir: Any | None = None,
        cache_mode: str | None = None,
        # Managed-mode-only knobs
        llmfacade_dir: str | Path | None = None,
        binary: str = "sd-server",
        startup_timeout: float | None = None,
        # SD_LAUNCH_KNOBS as provider-level defaults (managed mode only)
        model: str | None = None,
        diffusion_model: str | None = None,
        vae: str | None = None,
        clip_l: str | None = None,
        clip_g: str | None = None,
        t5xxl: str | None = None,
        llm: str | None = None,
        taesd: str | None = None,
        lora_model_dir: str | None = None,
        threads: int | None = None,
        max_vram: float | None = None,
        offload_to_cpu: bool | None = None,
        fa: bool | None = None,
        diffusion_fa: bool | None = None,
        extra_args: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        # Mode is decided here and never changes.
        self._managed = base_url is None

        sess_dir = (Path(llmfacade_dir) if llmfacade_dir else Path.cwd() / ".llmfacade").resolve()
        self._llmfacade_dir = sess_dir
        self._binary = binary
        self._startup_timeout = startup_timeout

        explicit_launch: dict[str, Any] = {
            "model": model,
            "diffusion_model": diffusion_model,
            "vae": vae,
            "clip_l": clip_l,
            "clip_g": clip_g,
            "t5xxl": t5xxl,
            "llm": llm,
            "taesd": taesd,
            "lora_model_dir": lora_model_dir,
            "threads": threads,
            "max_vram": max_vram,
            "offload_to_cpu": offload_to_cpu,
            "fa": fa,
            "diffusion_fa": diffusion_fa,
            "extra_args": tuple(extra_args) if extra_args is not None else None,
        }
        if not self._managed:
            offending = sorted(k for k, v in explicit_launch.items() if v is not None)
            if offending:
                raise UnsupportedFeature(
                    f"launch knobs {offending!r} require managed mode (omit base_url= to enable)",
                    self.NAME,
                    None,
                )
            self._launch_defaults: dict[str, Any] = {}
            self._supervisor: _SdServerSupervisor | None = None
        else:
            merged = default_provider_sd_defaults()
            for k, v in explicit_launch.items():
                if v is not None:
                    merged[k] = v
            self._launch_defaults = merged
            self._supervisor = _SdServerSupervisor(
                llmfacade_dir=sess_dir, binary=binary, startup_timeout=startup_timeout
            )

        super().__init__(
            manager=manager,
            api_key=api_key,
            base_url=base_url,
            log_dir=log_dir,
            cache_dir=cache_dir,
            cache_mode=cache_mode,
        )

    # ---- client construction ---------------------------------------------

    def _init_client(self) -> None:
        try:
            import openai as _openai
        except ImportError as e:
            raise ProviderNotInstalledError(
                "OpenAI SDK not installed. Run: pip install llmfacade[localimage]"
            ) from e
        self._module = _openai
        # Local servers need no key; the SDK requires a non-empty value.
        self._api_key = self._api_key_override or "-"
        self._client: Any = None
        self._aclient: Any = None
        self._client_base: str | None = None
        # Serialises client (re)builds against concurrent first-calls.
        self._client_lock = threading.Lock()
        # Serialises managed-mode generate so a model swap can't happen mid-request
        # (sync and async locks do not synchronise; a process should pick one).
        self._gen_lock = threading.Lock()
        self._gen_alock = asyncio.Lock()
        if not self._managed:
            self._build_image_clients(self._base_url or "")

    def _build_image_clients(self, openai_base: str) -> None:
        client_kwargs: dict[str, Any] = {"base_url": openai_base, "api_key": self._api_key}
        if self._managed:
            # We own the sd-server and may stop it out from under an in-flight
            # request during a swap. Retrying against the now-dead local port
            # adds no value (backoff + connect timeout each); fail fast instead.
            # External mode talks to a real server, so it keeps the SDK default.
            client_kwargs["max_retries"] = 0
        self._client = self._module.OpenAI(**client_kwargs)
        self._aclient = self._module.AsyncOpenAI(**client_kwargs)
        self._client_base = openai_base

    def _ensure_image_client(self, openai_base: str) -> None:
        """Build (or rebuild) the openai clients when the target base URL changed —
        a managed-mode swap allocates a fresh port, so the client must follow."""
        with self._client_lock:
            if self._client is None or self._client_base != openai_base:
                self._build_image_clients(openai_base)

    # ---- managed-mode model factory --------------------------------------

    def new_image_model(
        self,
        model_id: str | None = None,
        *,
        name: str | None = None,
        capability_override: frozenset[str] | None = None,
        # SD_LAUNCH_KNOBS (managed mode only)
        model: str | None = None,
        diffusion_model: str | None = None,
        vae: str | None = None,
        clip_l: str | None = None,
        clip_g: str | None = None,
        t5xxl: str | None = None,
        llm: str | None = None,
        taesd: str | None = None,
        lora_model_dir: str | None = None,
        threads: int | None = None,
        max_vram: float | None = None,
        offload_to_cpu: bool | None = None,
        fa: bool | None = None,
        diffusion_fa: bool | None = None,
        extra_args: list[str] | tuple[str, ...] | None = None,
        # Per-model image-generation defaults (both modes)
        n: int | None = None,
        size: str | None = None,
        aspect_ratio: str | None = None,
        quality: str | None = None,
        background: str | None = None,
        output_format: str | None = None,
    ) -> ImageModel:
        from llmfacade.image import ImageModel

        explicit_launch: dict[str, Any] = {
            "model": model,
            "diffusion_model": diffusion_model,
            "vae": vae,
            "clip_l": clip_l,
            "clip_g": clip_g,
            "t5xxl": t5xxl,
            "llm": llm,
            "taesd": taesd,
            "lora_model_dir": lora_model_dir,
            "threads": threads,
            "max_vram": max_vram,
            "offload_to_cpu": offload_to_cpu,
            "fa": fa,
            "diffusion_fa": diffusion_fa,
            "extra_args": tuple(extra_args) if extra_args is not None else None,
        }
        nonnull_launch_keys = sorted(k for k, v in explicit_launch.items() if v is not None)

        if not self._managed:
            if nonnull_launch_keys:
                raise UnsupportedFeature(
                    f"launch knobs {nonnull_launch_keys!r} require managed mode "
                    "(omit base_url= on the provider to enable)",
                    self.NAME,
                    model_id,
                )
            if name is not None:
                raise UnsupportedFeature(
                    "name= is a managed-mode kwarg (omit base_url= on the provider "
                    "to enable). In external mode pass the model id positionally.",
                    self.NAME,
                    model_id,
                )
            if model_id is None:
                raise ValueError(
                    "external-mode new_image_model() requires a positional model_id "
                    "(the model name your image server is configured to expose)."
                )
            return ImageModel(
                provider=self,
                model_id=model_id,
                capability_override=capability_override,
                n=n,
                size=size,
                aspect_ratio=aspect_ratio,
                quality=quality,
                background=background,
                output_format=output_format,
            )

        # Managed mode: cascade provider-level launch defaults < model overrides.
        merged = dict(self._launch_defaults)
        for k, v in explicit_launch.items():
            if v is not None:
                merged[k] = v
        if model_id is not None:
            if name is not None and name != model_id:
                raise ValueError(
                    f"new_image_model() got conflicting names: positional={model_id!r} "
                    f"vs name={name!r}. Pass one or the other."
                )
            name = model_id
        if not (merged.get("model") or merged.get("diffusion_model")):
            raise ValueError(
                "managed-mode new_image_model() requires model= (single-file checkpoint) "
                "or diffusion_model= (set at provider or model scope)"
            )

        # Existence-check every provided file/dir path and normalise to absolute,
        # so argv passes resolvable paths and the model-id hash is stable.
        for knob in _FILE_PATH_KNOBS:
            value = merged.get(knob)
            if value is not None:
                p = Path(value)
                if not p.exists():
                    raise FileNotFoundError(f"{knob} not found: {p}")
                merged[knob] = str(p.resolve())
        for knob in _DIR_PATH_KNOBS:
            value = merged.get(knob)
            if value is not None:
                p = Path(value)
                if not p.is_dir():
                    raise FileNotFoundError(f"{knob} is not a directory: {p}")
                merged[knob] = str(p.resolve())

        derived = derive_image_model_id(merged, name)
        entry = _SdLaunchEntry(
            model_id=derived,
            model=merged.get("model"),
            diffusion_model=merged.get("diffusion_model"),
            vae=merged.get("vae"),
            clip_l=merged.get("clip_l"),
            clip_g=merged.get("clip_g"),
            t5xxl=merged.get("t5xxl"),
            llm=merged.get("llm"),
            taesd=merged.get("taesd"),
            lora_model_dir=merged.get("lora_model_dir"),
            threads=merged.get("threads"),
            max_vram=merged.get("max_vram"),
            offload_to_cpu=bool(merged.get("offload_to_cpu", False)),
            fa=bool(merged.get("fa", False)),
            diffusion_fa=bool(merged.get("diffusion_fa", False)),
            extra_args=tuple(merged.get("extra_args") or ()),
        )
        assert self._supervisor is not None
        self._supervisor.register(entry)

        return ImageModel(
            provider=self,
            model_id=derived,
            capability_override=capability_override,
            n=n,
            size=size,
            aspect_ratio=aspect_ratio,
            quality=quality,
            background=background,
            output_format=output_format,
        )

    def _resolve_managed_model(self, model: str | None) -> str:
        """Resolve the target model id for a managed-mode generate. An explicit
        ``model`` wins; with exactly one registered model it's inferred; otherwise
        raise listing the registered ids."""
        assert self._supervisor is not None
        if model is not None:
            return model
        entries = self._supervisor.entries
        if len(entries) == 1:
            return entries[0].model_id
        if not entries:
            raise ValueError(
                "managed-mode generate_image requires model=<id>; no image models "
                "are registered. Call provider.new_image_model(...) first."
            )
        names = [e.model_id for e in entries]
        raise ValueError(
            "managed-mode generate_image on a multi-model provider requires "
            f"model=<id>; registered: {names!r}"
        )

    # ---- generation hooks ------------------------------------------------
    # The base Provider.generate_image / agenerate_image are the audit-logging
    # chokepoints; we implement the raw hooks. In managed mode the hook first
    # ensures the right sd-server is running (swapping if needed) under a
    # generation lock so a concurrent call can't swap the model out mid-request.

    def _image_kwargs(
        self,
        prompt: str,
        model: str | None,
        n: int,
        size: str | None,
        quality: str | None,
        background: str | None,
        output_format: str | None,
        reference_images: Sequence[ReferenceImage] | None,
        extra: dict[str, Any] | None,
    ) -> tuple[str, dict[str, Any]]:
        # request_b64=True: OpenAI-compatible local servers default to returning
        # URLs; we want the bytes inline.
        if reference_images:
            return "edit", build_edit_kwargs(
                model=model,
                prompt=prompt,
                reference_images=reference_images,
                n=n,
                size=size,
                extra=extra,
                request_b64=True,
            )
        return "generate", build_generate_kwargs(
            model=model,
            prompt=prompt,
            n=n,
            size=size,
            quality=quality,
            background=background,
            output_format=output_format,
            extra=extra,
            request_b64=True,
        )

    def _generate_image_raw(
        self,
        prompt: str,
        *,
        model: str | None = None,
        n: int = 1,
        size: str | None = None,
        aspect_ratio: str | None = None,
        quality: str | None = None,
        background: str | None = None,
        output_format: str | None = None,
        reference_images: Sequence[ReferenceImage] | None = None,
        save_dir: str | Path | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ImageResult:
        if not self._managed:
            return self._call_images_sync(
                prompt,
                model,
                n,
                size,
                quality,
                background,
                output_format,
                reference_images,
                save_dir,
                extra,
            )
        with self._gen_lock:
            target = self._resolve_managed_model(model)
            assert self._supervisor is not None
            base = self._supervisor.ensure_model(target)
            self._ensure_image_client(base.rstrip("/") + "/v1")
            return self._call_images_sync(
                prompt,
                target,
                n,
                size,
                quality,
                background,
                output_format,
                reference_images,
                save_dir,
                extra,
            )

    async def _agenerate_image_raw(
        self,
        prompt: str,
        *,
        model: str | None = None,
        n: int = 1,
        size: str | None = None,
        aspect_ratio: str | None = None,
        quality: str | None = None,
        background: str | None = None,
        output_format: str | None = None,
        reference_images: Sequence[ReferenceImage] | None = None,
        save_dir: str | Path | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ImageResult:
        if not self._managed:
            return await self._call_images_async(
                prompt,
                model,
                n,
                size,
                quality,
                background,
                output_format,
                reference_images,
                save_dir,
                extra,
            )
        async with self._gen_alock:
            target = self._resolve_managed_model(model)
            assert self._supervisor is not None
            # ensure_model spawns/swaps a subprocess synchronously; this mirrors
            # the llamacpp provider, which also drives its sync supervisor from
            # the async path.
            base = self._supervisor.ensure_model(target)
            self._ensure_image_client(base.rstrip("/") + "/v1")
            return await self._call_images_async(
                prompt,
                target,
                n,
                size,
                quality,
                background,
                output_format,
                reference_images,
                save_dir,
                extra,
            )

    def _call_images_sync(
        self,
        prompt: str,
        model: str | None,
        n: int,
        size: str | None,
        quality: str | None,
        background: str | None,
        output_format: str | None,
        reference_images: Sequence[ReferenceImage] | None,
        save_dir: str | Path | None,
        extra: dict[str, Any] | None,
    ) -> ImageResult:
        endpoint, kwargs = self._image_kwargs(
            prompt, model, n, size, quality, background, output_format, reference_images, extra
        )
        try:
            if endpoint == "edit":
                raw = self._client.images.edit(**kwargs)
            else:
                raw = self._client.images.generate(**kwargs)
        except self._module.AuthenticationError as e:
            raise AuthenticationError(str(e)) from e
        except self._module.RateLimitError as e:
            raise RateLimitError(str(e)) from e
        except self._module.APIError as e:
            raise ProviderError(str(e), original=e) from e
        result = parse_images_response(
            raw,
            model=model or "localimage",
            provider=self.NAME,
            fallback_media_type=media_type_for(output_format),
        )
        return _apply_save_dir(result, save_dir)

    async def _call_images_async(
        self,
        prompt: str,
        model: str | None,
        n: int,
        size: str | None,
        quality: str | None,
        background: str | None,
        output_format: str | None,
        reference_images: Sequence[ReferenceImage] | None,
        save_dir: str | Path | None,
        extra: dict[str, Any] | None,
    ) -> ImageResult:
        endpoint, kwargs = self._image_kwargs(
            prompt, model, n, size, quality, background, output_format, reference_images, extra
        )
        try:
            if endpoint == "edit":
                raw = await self._aclient.images.edit(**kwargs)
            else:
                raw = await self._aclient.images.generate(**kwargs)
        except self._module.AuthenticationError as e:
            raise AuthenticationError(str(e)) from e
        except self._module.RateLimitError as e:
            raise RateLimitError(str(e)) from e
        except self._module.APIError as e:
            raise ProviderError(str(e), original=e) from e
        result = parse_images_response(
            raw,
            model=model or "localimage",
            provider=self.NAME,
            fallback_media_type=media_type_for(output_format),
        )
        return _apply_save_dir(result, save_dir)

    # ---- explicit lifecycle ----------------------------------------------

    def shutdown(self) -> None:
        """Tear down any managed-mode ``sd-server`` subprocess. Idempotent; no-op
        in external mode. atexit and signal handlers also call this."""
        if self._supervisor is not None:
            self._supervisor.shutdown()
