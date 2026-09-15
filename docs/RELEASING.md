# Releasing

Version is the `version` field in [`pyproject.toml`](../pyproject.toml)
(`0.5.0` as of this writing). There is no separate `__version__` module.

## Branches and tags

| Ref | Role |
|-----|------|
| `develop` | Integration branch. CI runs here; do **not** tag releases here. |
| `main` | Stable line. Every release is a merge (or fast-forward) of `develop` into `main`. |
| `vX.Y.Z` | Annotated tag on the **`main` commit** that has that version in `pyproject.toml`. |

`0.4.4` on `main` (`2648b88`, 2026-07-07) was never tagged. From the **next**
version bump onward, every published version gets a `v*` tag and a GitHub
Release whose notes are the matching section of [`CHANGELOG.md`](../CHANGELOG.md).

## Checklist (next version bump)

1. On `develop`, set `project.version` in `pyproject.toml` (semver:
   patch / minor / major).
2. In `CHANGELOG.md`, rename `## [Unreleased]` to
   `## [X.Y.Z] - YYYY-MM-DD`, leave a fresh empty `## [Unreleased]` above it,
   and point the `[Unreleased]` / `[X.Y.Z]` links at the bottom of the file.
   After `v0.5.0` exists, that is `compare/v0.5.0...HEAD` and
   `compare/v0.5.0...vX.Y.Z`. The 0.5.0 section itself used
   `compare/2648b88...v0.5.0` because `0.4.4` was untagged.
3. Open a PR `develop` → `main` (or merge locally if that is the usual flow).
   CI on `main` must be green.
4. On `main`, after the merge:

   ```bash
   git checkout main
   git pull
   git tag -a "vX.Y.Z" -m "ost_photometry X.Y.Z"
   git push origin main
   git push origin "vX.Y.Z"
   ```

5. Pushing the tag runs [`.github/workflows/release.yml`](../.github/workflows/release.yml),
   which creates the GitHub Release from `CHANGELOG.md`. To do it by hand:

   ```bash
   python scripts/changelog_section.py vX.Y.Z > /tmp/notes.md
   gh release create "vX.Y.Z" --title "vX.Y.Z" --notes-file /tmp/notes.md
   ```

Do not retag. If the notes need a fix, edit the GitHub Release (and
`CHANGELOG.md` on `develop`) rather than moving the tag.

## What not to do

- Do not bump `pyproject.toml` without a changelog section for that version.
- Do not tag `develop` or a commit whose `version` does not match the tag.
- Do not use `vX.Y.Z` on a commit that is not an ancestor of `main`.
