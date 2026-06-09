"""@tool decorator: schema introspection, sync/async dispatch."""

from __future__ import annotations

import asyncio
from typing import Literal

import pytest

from llmfacade import tool
from llmfacade.models import ToolCall


def test_basic_schema():
    @tool
    def get_weather(city: str, units: str = "C") -> str:
        """Get current weather for a city."""
        return f"{city}:{units}"

    assert get_weather.name == "get_weather"
    assert "Get current weather" in get_weather.description
    assert get_weather.schema["type"] == "object"
    props = get_weather.schema["properties"]
    assert props["city"]["type"] == "string"
    assert props["units"]["type"] == "string"
    assert get_weather.schema["required"] == ["city"]


def test_literal_becomes_enum():
    @tool
    def pick(side: Literal["heads", "tails"]) -> str:
        """Pick a coin side."""
        return side

    assert pick.schema["properties"]["side"]["enum"] == ["heads", "tails"]


def test_literal_single_bool_is_boolean():
    @tool
    def confirm(ack: Literal[True]) -> str:
        """Require an explicit acknowledgement."""
        return "ok"

    prop = confirm.schema["properties"]["ack"]
    assert prop == {"type": "boolean", "enum": [True]}


def test_literal_both_bools_is_boolean():
    @tool
    def toggle(state: Literal[True, False]) -> str:
        """Flip a switch."""
        return str(state)

    prop = toggle.schema["properties"]["state"]
    assert prop == {"type": "boolean", "enum": [True, False]}


def test_literal_mixed_bool_int_falls_through_to_bare_enum():
    @tool
    def mixed(v: Literal[True, 2]) -> str:
        """Mixed bool/int literal."""
        return str(v)

    prop = mixed.schema["properties"]["v"]
    assert prop == {"enum": [True, 2]}


def test_literal_ints_still_number():
    @tool
    def level(n: Literal[1, 2, 3]) -> str:
        """Pick a level."""
        return str(n)

    prop = level.schema["properties"]["n"]
    assert prop == {"type": "number", "enum": [1, 2, 3]}


def test_list_and_int():
    @tool
    def make(items: list[str], count: int = 1) -> str:
        """Make stuff."""
        return f"{count} of {items}"

    props = make.schema["properties"]
    assert props["items"]["type"] == "array"
    assert props["items"]["items"]["type"] == "string"
    assert props["count"]["type"] == "integer"


def test_invoke_dispatch():
    @tool
    def add(a: int, b: int) -> int:
        """Add two ints."""
        return a + b

    call = ToolCall(id="x", name="add", input={"a": 2, "b": 3}, _fn=add.fn)
    assert call.invoke() == 5


def test_invoke_without_fn_raises():
    call = ToolCall(id="x", name="missing", input={})
    with pytest.raises(RuntimeError):
        call.invoke()


def test_async_tool():
    @tool
    async def fetch(url: str) -> str:
        """Fetch a URL."""
        return f"got:{url}"

    assert fetch.is_async
    call = ToolCall(id="x", name="fetch", input={"url": "y"}, _fn=fetch.fn)
    result = asyncio.run(call.ainvoke())
    assert result == "got:y"
