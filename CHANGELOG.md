# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.1] - 2026-07-27

### Added

- Tutorial and embedding-filter notebooks now render as native documentation
  pages under `docs/5.notebooks/`, instead of living outside the docs site.
- `[project.urls]` metadata (Repository, Documentation, Issues, Changelog)
  for the PyPI project page.
- A QA workflow that runs lint, type-checking, formatting, and the test
  suite on every push and pull request, rather than only at release time.

### Changed

- Re-locked dependencies, resolving all open Dependabot alerts (jupyter-server,
  jupyterlab, pillow, mistune, tornado, urllib3, setuptools, soupsieve, bleach,
  torch, and other transitive dev/notebook dependencies).
- Switched the static type checker from `ty` to `pyright`.
- Notebook files are now excluded from ruff's lint and format checks; their
  compact, pedagogical style is intentional rather than a rule violation.
