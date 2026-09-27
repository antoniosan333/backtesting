from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOMAIN_PATHS = (
    ROOT / "src" / "lambdaclass" / "strategies",
    ROOT / "src" / "lambdaclass" / "backtest",
    ROOT / "strategies",
)


def _reporting_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(
                alias.name
                for alias in node.names
                if alias.name == "lambdaclass.reporting" or alias.name.startswith("lambdaclass.reporting.")
            )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "lambdaclass.reporting" or module.startswith("lambdaclass.reporting."):
                imports.append(module)
    return imports


@pytest.mark.parametrize(
    "path",
    [path for domain_path in DOMAIN_PATHS for path in sorted(domain_path.rglob("*.py"))],
    ids=lambda path: str(path.relative_to(ROOT)),
)
def test_strategy_and_backtest_modules_do_not_import_reporting(path: Path) -> None:
    assert not _reporting_imports(path)
