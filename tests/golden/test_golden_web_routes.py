"""Characterization tests: pin the web dashboard's route table and its OpenAPI schema.

web_routes.txt holds one line per registered route in registration order (kind, methods, path, endpoint
name), with the static mount last; web_openapi.json holds `app.openapi()` with sorted keys, so any drift in
a handler name, parameter, default, docstring or request model shows up. Re-generate deliberately with
`WAVETRACE_UPDATE_GOLDEN=1 pytest tests/golden`.
"""

import json
import os
import warnings
from pathlib import Path

from fastapi.routing import APIRoute, APIWebSocketRoute
from starlette.routing import Mount, Route

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)  # foxglove_websocket imports the legacy websockets API
    from web.app import app

EXPECTED = Path(__file__).parent / "expected"


def _describe_route(route) -> str:
    if isinstance(route, APIWebSocketRoute):
        return f"websocket - {route.path} {route.name}"
    if isinstance(route, Mount):
        return f"mount - {route.path or '/'} {route.name}"
    assert isinstance(route, (APIRoute, Route)), type(route)
    return f"{type(route).__name__.lower()} {','.join(sorted(route.methods))} {route.path} {route.name}"


def _assert_matches_golden(snapshot: str, file_name: str) -> None:
    expected_path = EXPECTED / file_name
    if os.environ.get("WAVETRACE_UPDATE_GOLDEN"):
        expected_path.write_bytes(snapshot.encode())
    assert snapshot == expected_path.read_bytes().decode()


def test_route_table_pins_every_route_in_registration_order():
    _assert_matches_golden("".join(f"{_describe_route(route)}\n" for route in app.routes), "web_routes.txt")


def test_openapi_schema_is_pinned():
    _assert_matches_golden(json.dumps(app.openapi(), sort_keys=True, indent=1) + "\n", "web_openapi.json")
