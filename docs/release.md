# Release process

## Pre-release checks

1. Update `CHANGELOG.md`, `CITATION.cff`, `pyproject.toml`, and
   `src/dfckit/__init__.py` to the same version.
2. Run the complete optional-dependency test environment:

   ```bash
   python -m unittest discover
   ruff check src tests
   mkdocs build --strict
   ```

3. Build and inspect both distributions:

   ```bash
   rm -rf build dist
   python -m build
   python -m twine check dist/*
   ```

4. Install the wheel into a clean environment and run XCP-D integration plus
   fitted-model save/load/prediction smoke tests.
5. Confirm that the source distribution contains documentation, tests, license,
   citation metadata, and no private data or local server paths.
6. Create an annotated version tag only after CI passes.

## 1.x compatibility

The 1.x series follows semantic versioning. Patch releases contain fixes and
documentation changes; minor releases may add public APIs without changing
existing field semantics; breaking API changes require a new major version.
