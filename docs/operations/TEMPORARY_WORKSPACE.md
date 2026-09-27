# Temporary workspace and cleanup policy

## Purpose

All new disposable repository-local work belongs under the single hidden `.tmp/`
directory. This prevents pytest, agents and development tools from leaving many
unexplained folders at repository root and avoids repeatedly expanding `.gitignore`.

Use this layout:

```text
.tmp/
  pytest/<task>/
  tools/<tool>/<task>/
  downloads/<task>/
  renders/<task>/
```

Names should be short human-readable task labels. A timestamp or hash is unnecessary
because nothing under `.tmp/` is evidence or part of a reproducible result.

## Required practice

- Run pytest with `--basetemp .tmp/pytest/<task>` when an explicit base directory is
  needed. Reuse or purge that task directory before running the same command again.
- Configure tool caches, temporary renders and disposable downloads below `.tmp/`.
- Do not add a new root-level temporary pattern to `.gitignore`; route the producer to
  `.tmp/` instead.
- Purge a task directory after its command exits and before handoff. A periodic full
  purge is appropriate when no test, build, notebook or agent process is using it.
- A tool failure does not turn scratch output into evidence. Preserve a required
  diagnostic in `reports/operations/` or another declared artefact location, then
  remove the scratch copy.

## Never purge as temporary data

The following locations are not temporary workspace and must not be included in a
scratch cleanup:

- `data/`: raw caches, canonical datasets and pinned study inputs;
- `runs/`: immutable experiment artefacts and failure/checkpoint lineage;
- `reports/`: generated evidence and operational monitoring logs;
- `configs/`, `docs/`, `notebooks/`, `src/` and `tests/`;
- environment and authority files such as `pyproject.toml`, requirements files,
  approved protocols, manifests and approval records; and
- `.venv/` and `.git/`.

## Safe purge procedure

Inspect cleanup targets first, then purge only after confirming no process is using
`.tmp/`. Run from the repository root:

```powershell
 .venv\Scripts\python.exe -m trading_pipeline.operations.clean_temp
 .venv\Scripts\python.exe -m trading_pipeline.operations.clean_temp --purge
```

The first command lists immediate children without deleting anything. The second
removes those children recursively while retaining `.tmp/`. The command derives the
repository root from its source module, requires `.git` and `pyproject.toml`, rejects
a `.tmp` link or non-directory, and validates each child before deletion. It never
accepts a caller-supplied deletion path. Do not replace it with a wildcard or point
cleanup at `data/`, `runs/`, `reports/` or the repository root.

## Existing root-level legacy folders

The currently ignored root-level `.pytest-*`, `pytest-*`, `tmp*`,
`.streamlit-skills-*` and `rl-price-eligibility-pytest` directories are legacy test or
tool scratch. They are not canonical research inputs or published evidence. They may be
removed in a one-time cleanup after all active processes finish, but first inspect any
unexpectedly named folder and confirm that no required diagnostic exists only there.

The legacy ignore rules remain temporarily so a clean checkout is not flooded with
untracked files. They should not be treated as permission to create more root-level
scratch directories. Once the one-time cleanup is complete, those compatibility rules
can be removed from `.gitignore`; `/.tmp/` is the permanent rule.
