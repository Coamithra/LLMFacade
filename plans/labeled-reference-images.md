# Labeled / interleaved reference images in `generate_image`

Trello card [4FFTlmAZ](https://trello.com/c/4FFTlmAZ) — "Support labeled / interleaved reference images in generate_image (entity→image binding)".

## Context

`generate_image(prompt, reference_images=[ImageBlock, ...])` currently appends reference images as an **unlabeled bag** after a single text prompt. A caller cannot tell the model *which image is which subject*, so the manual-chat binding pattern — "this is Adam:" [img] "this is Bert:" [img] "draw Adam waving at Bert" — is impossible. The motivating consumer is MTGAI character-ref binding, where cards with multiple recurring entities render with no reliable entity→face binding.

## Design

Chosen API shape: **`LabeledImage` frozen dataclass + tuple shorthand** (keeps the plain `list[ImageBlock]` fully back-compat).

### `models.py`
- New `@dataclass(frozen=True, slots=True) LabeledImage` with fields `label: str`, `image: ImageBlock`.
- Type alias `ReferenceImage = ImageBlock | LabeledImage | tuple[str, ImageBlock]`.
- `normalize_reference_images(refs) -> list[tuple[str | None, ImageBlock]]`: coerces each item to a `(label_or_None, ImageBlock)` pair. A bare `ImageBlock` → `(None, block)`; a `LabeledImage` or `(label, block)` tuple → `(label, block)`. Raises `TypeError` on anything else. This is the single chokepoint providers consume so "is any item labeled?" is `any(label for label, _ in pairs)`.

### Google (`google.py` `_image_contents`) — the primary, true-binding target
- Normalize. If **no** item is labeled → keep the exact current shape (`[{text: prompt}]` then `inline_data` parts) byte-for-byte (back-compat).
- If **any** labeled → interleave: for each pair emit the label as a `{text: label}` part (verbatim — caller controls phrasing) immediately before its `{inline_data}`, then append `{text: prompt}` **last**. This establishes each identity before the instruction, matching the card's target wire shape.

### OpenAI / localimage (`_openai_images.py` `build_edit_kwargs`) — best-effort order binding
- The `images.edit` endpoint takes a flat `image=[]` list + one prompt; it **cannot** interleave per-image labels. True interleaving needs the Responses API image path — out of scope (documented limitation).
- Normalize. Images still upload in list order. If any labeled, synthesize a textual preamble (`"Reference image 1 is Adam. Reference image 2 is Bert."`) prepended to the prompt so the model has an order→identity map. Unlabeled path is unchanged.

### Logging (`_image_log.py` `build_image_record`)
- Add a `reference_labels` field (list of the non-empty labels; `[]` when none). Keeps `reference_images` as the count. Render labels in the HTML section when present. (Labels are caller-chosen short strings, not sensitive; image bytes are still never logged.)

### Signature plumbing (type hints only, pass-through)
Widen `reference_images: list[ImageBlock] | None` → `Sequence[ReferenceImage] | None` on: `provider.generate_image/agenerate_image/_generate_image_raw/_agenerate_image_raw/_log_image_generation`, `facade.generate_image/agenerate_image`, `image.ImageModel.generate/agenerate`, the three providers' raw methods + helpers, and `build_edit_kwargs` / `build_image_record`.

### Export
`LabeledImage` exported from `llmfacade` (`__init__.py` import + `__all__`).

## Tests (`tests/test_image_generation.py`, `tests/test_image_logging.py`)
- `test_normalize_reference_images_*`: ImageBlock→(None,·), LabeledImage, tuple shorthand, mixed list, bad item raises `TypeError`.
- `test_google_labeled_references_interleaved`: label text parts precede each image, prompt is the last part.
- `test_google_unlabeled_references_unchanged`: back-compat (prompt first).
- `test_openai_labeled_references_preamble`: prompt gains the order preamble; `image` list order preserved.
- `test_openai_tuple_shorthand_accepted`.
- `test_image_log_records_reference_labels`.

## Out of scope
- True interleaving on the OpenAI **edit** endpoint (needs Responses API image-generation path) — best-effort textual preamble only.
- Any change to the chat/`Conversation` vision path (this card is image *generation* only).

## Verification
- `ruff check src/`, `ruff format src/`, `python -c "import llmfacade"`, `pytest` (unit only; no integration — image integration tests cost money / are gated).
