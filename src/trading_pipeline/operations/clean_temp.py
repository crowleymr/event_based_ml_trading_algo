"""Safely inspect or purge disposable repository-local ``.tmp`` contents."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _temporary_root(repository: Path) -> Path:
    root = repository.resolve(strict=True)
    if not (root / "pyproject.toml").is_file() or not (root / ".git").exists():
        raise ValueError("Run cleanup only for a repository root containing .git and pyproject.toml.")
    temporary = root / ".tmp"
    if temporary.is_symlink():
        raise ValueError("Refusing cleanup because repository .tmp is a symbolic link.")
    if temporary.exists() and not temporary.is_dir():
        raise ValueError("Refusing cleanup because repository .tmp is not a directory.")
    return temporary


def clean(repository: Path, *, purge: bool = False) -> list[Path]:
    """List immediate .tmp children, optionally removing only those children."""
    temporary = _temporary_root(repository)
    if not temporary.exists():
        return []
    root = repository.resolve(strict=True)
    resolved_temp = temporary.resolve(strict=True)
    if resolved_temp.parent != root or resolved_temp.name != ".tmp":
        raise ValueError("Refusing cleanup because .tmp does not resolve to the repository's .tmp child.")

    children = sorted(temporary.iterdir(), key=lambda path: path.name.casefold())
    for child in children:
        # Immediate children are the only deletion targets. Resolve non-links to
        # detect junctions/reparse points that could redirect recursive removal.
        if child.is_symlink():
            continue  # unlinking the link itself does not traverse its target
        resolved = child.resolve(strict=True)
        if resolved.parent != resolved_temp:
            raise ValueError(f"Refusing cleanup because {child.name!r} escapes .tmp.")
    if purge:
        for child in children:
            if child.is_symlink() or not child.is_dir():
                child.unlink()
            else:
                shutil.rmtree(child)
    return children


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect disposable .tmp contents; pass --purge to remove its immediate children."
    )
    parser.add_argument("--purge", action="store_true", help="recursively remove children under repository .tmp")
    args = parser.parse_args(argv)
    repository = Path(__file__).resolve().parents[3]
    try:
        children = clean(repository, purge=args.purge)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if not children:
        print("No .tmp contents found.")
    elif args.purge:
        print(f"Removed {len(children)} immediate child item(s) under {repository / '.tmp'}.")
    else:
        print("Would remove the following .tmp children (rerun with --purge to delete):")
        for child in children:
            print(f"  {child.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
