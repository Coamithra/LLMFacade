# Image generation support (`generate_image`: OpenAI + Google + local)

Trello: [Image generation support](https://trello.com/c/yN1FyNzu) (card `6a20ada6`)

## Context

LLMFacade is text-only today: every provider implements the chat/completion hooks
(`_complete_raw` / `_stream_raw` / async). There is no image-*generation* entry point
(`ImageBlock` exists only as *input* for vision). The MTGAI project's Art Generation
stage wants to pick a hosted (OpenAI / Google) **or local (Flux et al.)** image generator for
card art, including **reference-image conditioning** (feed a character/location reference so art
stays on-model).

## Research findings (current wire shapes, verified June 2026)

**OpenAI** (`openai>=2.32`, already a dependency):
- `client.images.generate(prompt, model, n, size, quality, background, output_format)` — text→image.
  `gpt-image-*` returns base64 (`resp.data[i].b64_json`; `url` is None).
- `client.images.edit(image=[...], prompt, mask=, n, size)` — reference/edit; `image=` takes a
  **list** (≤16) of `(filename, bytes, mimetype)` tuples. Reference-image path.
- `resp.usage`: `input_tokens` / `output_tokens` / `total_tokens` (+ details) on `gpt-image-*`;
  `None` on dall-e. **No dollar figure.** Async: `AsyncOpenAI().images.generate / .edit`.
  Errors: `openai.AuthenticationError / RateLimitError / APIError`.

**Google** (`google-genai>=1.73`, already a dependency) — **Gemini-native only**.
(Imagen is being shut down per Google's Gemini API deprecations page — discarded.)
- `gemini-2.5-flash-image` ("Nano Banana") via `client.models.generate_content(model,
  contents=[prompt, <reference Parts>], config=GenerateContentConfig(response_modalities=["IMAGE"],
  image_config=ImageConfig(aspect_ratio, image_size)))`.
- Output: `resp.candidates[0].content.parts[i].inline_data.data` (bytes) + `.mime_type`.
- **Reference images**: pass `types.Part.from_bytes(blk.data, blk.media_type)` in `contents`.
- Usage: `resp.usage_metadata`. Async: `client.aio.models.generate_content`.
  Errors: existing `_reraise`.

**Local** — target an **OpenAI-compatible local image server** and reuse the OpenAI Images
transport (the same trick llamacpp uses for chat). Confirmed: stable-diffusion.cpp's **`sd-server`**
(same author as llama.cpp; pure C/C++; Flux/SD/SDXL/Qwen-Image) exposes OpenAI-compatible
`POST /v1/images/generations` + `/v1/images/edits`, returns `data[].b64_json` when
`response_format:"b64_json"`, accepts `model/prompt/n/size`, embeds SD-specific args via
`<sd_cpp_extra_args>{...}</sd_cpp_extra_args>` in the prompt, launches `sd-server -m model.safetensors`
(default `127.0.0.1:1234`). **LocalAI** is an equivalent OpenAI-compatible option. ComfyUI has no
native OpenAI-compatible API → out of scope.
- **This card: EXTERNAL mode only** (point `base_url` at a running OpenAI-compatible image server).
- **Managed mode** (spawn/supervise `sd-server`, mirroring the llama-swap supervisor) → **follow-up card.**

## Design

### New wire-format types — `models.py`

```python
@dataclass(frozen=True, slots=True)
class ImageUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    image_count: int = 0

@dataclass(frozen=True, slots=True)
class ImageResult:
    images: list[ImageBlock]      # reuse the existing ImageBlock (bytes + media_type)
    usage: ImageUsage | None      # None when the provider reports nothing
    model: str
    provider: str
    paths: list[Path] = field(default_factory=list)   # filled iff save_dir was given
    raw: object = field(default=None, repr=False, compare=False)

    def save(self, dest, *, prefix="image") -> list[Path]:
        """Write each image into directory `dest` as `<prefix>_<i><ext>` (ext from media_type)."""
```

Exported from `llmfacade/__init__.py`.

### Provider base — `provider.py`

```python
def generate_image(self, prompt, *, model=None, n=1, size=None, aspect_ratio=None,
    quality=None, background=None, output_format=None,
    reference_images: list[ImageBlock] | None = None,
    save_dir: str | Path | None = None, extra: dict[str, Any] | None = None,
) -> ImageResult:
    raise UnsupportedFeature("image_generation", self.NAME, model)
async def agenerate_image(...) -> ImageResult: raise UnsupportedFeature(...)
def new_image_model(self, model_id, *, n=None, size=None, aspect_ratio=None,
    quality=None, background=None, output_format=None) -> ImageModel: ...
```

Base raises `UnsupportedFeature("image_generation", ...)` so Anthropic / llamacpp fail fast and
`is_available("image_generation")` is False for them. `save_dir` given → write files + fill `paths`.

### Shared OpenAI-Images helpers — new `providers/_openai_images.py`

Used by BOTH the OpenAI provider and the local provider (both speak the OpenAI Images API):
- `build_generate_kwargs(model, prompt, n, size, quality, background, output_format, extra, request_b64)`
- `build_edit_kwargs(model, prompt, reference_images, n, size, extra, request_b64)` — builds
  `image=[(f"ref{i}.png", blk.data, blk.media_type), ...]`, optional `mask` from `extra`.
- `parse_images_response(raw, *, model, provider, fallback_media_type) -> ImageResult` — reads
  `data[i].b64_json` → `ImageBlock`; maps `usage` → `ImageUsage`. Raises a clear `ProviderError`
  if a datum has only `url` (point at gpt-image-1 / `response_format=b64_json`).

`request_b64=True` sends `response_format="b64_json"` (needed for local sd-server; for `gpt-image-*`
it's omitted since the model always returns b64 and rejects the param).

### `ImageModel` binder — new `src/llmfacade/image.py`

Provider-agnostic. Holds `provider` + `model_id` + optional per-model defaults
(`n/size/aspect_ratio/quality/background/output_format`). `generate(prompt, **overrides)` /
`agenerate(...)` forward to `provider.generate_image(model=self._model_id, ...)`, per-model defaults
applied for any arg left unset. `capability_override` like `Model`. Kept out of `model.py`.

### `LLM.generate_image` convenience — `facade.py`

```python
def generate_image(self, prompt, *, provider, model, base_url=None, api_key=None,
                   **gen_kwargs) -> ImageResult:
    key = (provider, base_url)
    p = self._image_providers.get(key) or self.new_provider(provider, base_url=base_url,
                                                            api_key=api_key)
    self._image_providers[key] = p
    return p.generate_image(prompt, model=model, **gen_kwargs)
# + agenerate_image
```

`base_url`/`api_key` surface for local (`provider="localimage", base_url="http://127.0.0.1:1234/v1"`);
hosted providers use manager `api_keys`/env and omit both. Provider cached per `(provider, base_url)`.

### Capability gating

- Add the pure flag `"image_generation"` to `OpenAIProvider.SUPPORTS`, `GoogleProvider.SUPPORTS`,
  and `LocalImageProvider.SUPPORTS`.
- `ImageModel` narrows via `capability_override`, same as `Model`.

### OpenAI provider — `providers/openai.py`

`generate_image`/`agenerate_image`: `reference_images` → `images.edit` (via `build_edit_kwargs`,
`request_b64=False`); else `images.generate` (via `build_generate_kwargs`). Parse with
`parse_images_response(..., provider="openai", fallback_media_type from output_format)`. Same error
mapping as `_complete_raw`.

### Google provider — `providers/google.py`

`generate_image`/`agenerate_image`: build `contents=[prompt] + [Part.from_bytes(blk.data,
blk.media_type) for blk in reference_images]`; `generate_content(model, contents,
config=GenerateContentConfig(response_modalities=["IMAGE"], image_config=ImageConfig(
aspect_ratio=aspect_ratio)))`; pull `inline_data` image parts from `candidates[0].content.parts`;
map `usage_metadata` → `ImageUsage`. Existing `_reraise`.

### Local image provider — new `providers/localimage.py`

`class LocalImageProvider(Provider)`, `NAME="localimage"`, `API_KEY_ENV=None`,
`SUPPORTS={"image_generation"}`. Registered in `PROVIDER_REGISTRY` as `"localimage"`.
- External mode only: `base_url` **required** at construction (raise a clear error pointing at the
  managed-mode follow-up if missing). `api_key` optional → placeholder (`"-"`); local servers need none.
- `_init_client` builds `openai.OpenAI(base_url=..., api_key=...)` + async (lazy import; reuses the
  `openai` extra). `generate_image`/`agenerate_image` use the shared `_openai_images` helpers with
  `request_b64=True`; `reference_images` → `images.edit`. `extra` forwarded as `extra_body=` (LocalAI
  server params) — doc the sd-server `<sd_cpp_extra_args>` prompt syntax as the alternative.
- pyproject: add an optional `localimage = ["openai>=2.32"]` extra (mirrors `llamacpp`).

## Tests (mock the SDK client — no real API calls)

`tests/test_image_generation.py`:
- OpenAI: `generate` builds correct kwargs + returns `ImageResult` (bytes from `b64_json`,
  `ImageUsage` from `resp.usage`); `reference_images` → `images.edit` with list `image=`.
- Google: `generate_content` called with `response_modalities=["IMAGE"]` + reference `Part`s;
  parses `inline_data`; maps `usage_metadata`.
- Local: `localimage` requires `base_url`; sends `response_format="b64_json"`; parses `b64_json`;
  `reference_images` → `images.edit`; url-only datum → `ProviderError`.
- `ImageResult.save()` writes files with correct extensions; `save_dir=` populates `.paths`.
- Capability: `"image_generation"` in openai/google/localimage SUPPORTS, absent on anthropic/llamacpp;
  base `generate_image` raises `UnsupportedFeature`.
- `ImageModel.generate` binds model_id + applies per-model defaults.
- `LLM.generate_image` resolves + caches the provider (incl. `base_url` for local) and delegates.

`tests/integration/test_image_generation_live.py` — gated `@pytest.mark.integration`, **not run**:
one real OpenAI gen, one Gemini-native reference edit, one local sd-server gen (skipped unless a
`LOCALIMAGE_BASE_URL` env is set).

## Docs

- `CLAUDE.md`: new **Image generation** section (surface, `image_generation` flag, the three
  providers, Gemini-native-not-Imagen note, reference-image matrix, local external-vs-managed,
  `ImageResult`/`ImageUsage`/`ImageModel`); update `models.py` / openai / google key-file bullets,
  the `localimage` provider entry, `__init__` exports, and **Future work** (managed local mode).

## Out of scope (→ follow-up cards / documented limitations)

- **Managed local mode** (spawn/supervise `sd-server`, like llama-swap) → follow-up card.
- Imagen (deprecated) and Vertex-only Imagen reference editing.
- Anthropic / llamacpp image generation (no hosted image API) → base raises `UnsupportedFeature`.
- ComfyUI native graph API (no OpenAI-compatible endpoint).
- Built-in price table / dollar-cost estimate (return raw token usage; caller derives cost).
- JSONL/HTML logging of image generations + response-cache integration → possible follow-up.
- MTGAI consumer wiring (separate MTGAI card).
```
