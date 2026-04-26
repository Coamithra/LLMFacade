from __future__ import annotations

import inspect
import typing
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Union, get_args, get_origin


@dataclass(frozen=True, slots=True)
class Tool:
    """A registered tool: original function + JSON schema."""

    name: str
    description: str
    schema: dict[str, Any]
    fn: Callable[..., Any] = field(repr=False)
    is_async: bool = False


_PRIMITIVES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _annotation_to_schema(annotation: Any) -> dict[str, Any]:
    """Translate a Python type annotation into a JSON-schema fragment.

    Supports str/int/float/bool, list[T], dict, Literal[...], and Union (Optional)."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Literal:
        values = list(args)
        if all(isinstance(v, str) for v in values):
            return {"type": "string", "enum": values}
        if all(isinstance(v, (int, float)) for v in values):
            return {"type": "number", "enum": values}
        return {"enum": values}

    if origin is Union or (origin is not None and str(origin) == "types.UnionType"):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _annotation_to_schema(non_none[0])
        return {"anyOf": [_annotation_to_schema(a) for a in non_none]}

    if origin in (list, tuple):
        item_schema = _annotation_to_schema(args[0]) if args else {}
        return {"type": "array", "items": item_schema or {}}

    if origin is dict:
        return {"type": "object"}

    if isinstance(annotation, type):
        if annotation in _PRIMITIVES:
            return {"type": _PRIMITIVES[annotation]}
        if issubclass(annotation, list):
            return {"type": "array"}
        if issubclass(annotation, dict):
            return {"type": "object"}

    return {}


def _build_schema(fn: Callable[..., Any]) -> tuple[str, dict[str, Any]]:
    """Build (description, json_schema) for a function from its signature + docstring."""
    sig = inspect.signature(fn)
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        hints = {}

    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        annotation = hints.get(param_name, param.annotation)
        prop_schema = _annotation_to_schema(annotation)
        if not prop_schema:
            prop_schema = {"type": "string"}
        properties[param_name] = prop_schema
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required

    description = inspect.getdoc(fn) or ""
    return description, schema


def tool(fn: Callable[..., Any]) -> Tool:
    """Decorator: turn a Python function into a Tool with auto-generated JSON schema.

    The function's name becomes the tool name, its docstring becomes the description,
    and its signature + type hints determine the input schema."""
    description, schema = _build_schema(fn)
    return Tool(
        name=fn.__name__,
        description=description,
        schema=schema,
        fn=fn,
        is_async=inspect.iscoroutinefunction(fn),
    )
