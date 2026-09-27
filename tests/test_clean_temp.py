from pathlib import Path

import pytest

from trading_pipeline.operations.clean_temp import clean


def _repository(path: Path) -> Path:
    path.mkdir()
    (path / ".git").mkdir()
    (path / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    return path


def test_inspect_and_purge_only_tmp_children(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repo")
    scratch = repository / ".tmp" / "pytest" / "task"
    scratch.mkdir(parents=True)
    (scratch / "output.txt").write_text("scratch", encoding="utf-8")
    protected = repository / "runs" / "immutable.json"
    protected.parent.mkdir()
    protected.write_text("evidence", encoding="utf-8")

    listed = clean(repository)
    assert listed == [repository / ".tmp" / "pytest"]
    assert (scratch / "output.txt").exists()

    clean(repository, purge=True)
    assert (repository / ".tmp").is_dir()
    assert list((repository / ".tmp").iterdir()) == []
    assert protected.read_text(encoding="utf-8") == "evidence"
    assert (repository / "pyproject.toml").is_file()


def test_refuses_symlinked_tmp_root(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repo")
    external = tmp_path / "external"
    external.mkdir()
    (external / "keep.txt").write_text("keep", encoding="utf-8")
    try:
        (repository / ".tmp").symlink_to(external, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable")

    with pytest.raises(ValueError, match="symbolic link"):
        clean(repository, purge=True)
    assert (external / "keep.txt").exists()


def test_refuses_non_repository_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="repository root"):
        clean(tmp_path)
