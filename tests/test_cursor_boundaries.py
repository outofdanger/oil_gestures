from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURSOR_ROOTS = (
    PROJECT_ROOT / "oil_gestures" / "cursor",
    PROJECT_ROOT / "oil_gestures" / "gestures" / "cursor",
)
GENERAL_RECOGNITION_ROOTS = (
    PROJECT_ROOT / "oil_gestures" / "gestures" / "static",
    PROJECT_ROOT / "oil_gestures" / "gestures" / "dynamic",
)


def _imports(root: Path) -> set[str]:
    modules: set[str] = set()
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
            elif isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
    return modules


def test_cursor_does_not_import_static_or_dynamic_recognition() -> None:
    imports = set().union(*(_imports(root) for root in CURSOR_ROOTS))

    assert not any(module.startswith("oil_gestures.gestures.static") for module in imports)
    assert not any(module.startswith("oil_gestures.gestures.dynamic") for module in imports)


def test_static_and_dynamic_recognition_do_not_import_cursor() -> None:
    imports = set().union(*(_imports(root) for root in GENERAL_RECOGNITION_ROOTS))

    assert not any(module.startswith("oil_gestures.cursor") for module in imports)
    assert not any(module.startswith("oil_gestures.gestures.cursor") for module in imports)
