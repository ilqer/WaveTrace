"""No package, module, class or function under the Python source roots is named for a dumping ground
(`utils`, `helpers`, `common`, `misc`, `manager`): such a name admits its contents share no responsibility,
so a hit is a design finding to resolve, not a rename. O(files): one `ast` parse per module."""
import ast
import os
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOTS = ("wavetrace", "web", "scripts", "firmware/pi")
DUMPING_GROUND_WORDS = frozenset({"util", "utils", "helper", "helpers", "common", "misc", "manager", "managers"})
# web/ui/node_modules ships third-party Python (node-gyp's `common.py`)
_SKIPPED_DIRECTORY_NAMES = frozenset({"node_modules", "__pycache__"})
_IDENTIFIER_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")


def _words(name: str) -> set[str]:
    return {word.lower() for word in _IDENTIFIER_WORD.findall(name)}


def _is_dumping_ground(name: str) -> bool:
    return not _words(name).isdisjoint(DUMPING_GROUND_WORDS)


def _python_files() -> list[Path]:
    files = []
    for source_root in SOURCE_ROOTS:
        for directory, subdirectory_names, file_names in os.walk(REPO_ROOT / source_root):
            subdirectory_names[:] = [name for name in subdirectory_names
                                     if name not in _SKIPPED_DIRECTORY_NAMES and not name.startswith(".")]
            files += [Path(directory) / name for name in file_names if name.endswith(".py")]
    return files


def _violations(path: Path) -> list[str]:
    relative_path = path.relative_to(REPO_ROOT)
    package_parts = relative_path.parts[:-1]
    found = [f"{Path(*package_parts[:depth + 1])}: package {part!r}"
             for depth, part in enumerate(package_parts) if _is_dumping_ground(part)]
    if _is_dumping_ground(path.stem):
        found.append(f"{relative_path}: module {path.stem!r}")
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and _is_dumping_ground(node.name):
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            found.append(f"{relative_path}:{node.lineno}: {kind} {node.name!r}")
    return found


@pytest.mark.parametrize("name", ["utils", "common", "FooManager", "Helpers", "HTTPHelper", "misc_tools"])
def test_dumping_ground_word_is_found_in_any_casing(name):
    assert _is_dumping_ground(name)


def test_source_tree_has_no_dumping_ground_names():
    files = _python_files()
    assert REPO_ROOT / "wavetrace" / "application" / "ports.py" in files
    violations = sorted({violation for path in files for violation in _violations(path)})
    assert not violations, "dumping-ground names:\n" + "\n".join(violations)
