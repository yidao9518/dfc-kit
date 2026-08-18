# Contributing

Contributions should preserve the package boundary: production inputs begin at
XCP-D parcellated derivatives, while cohort rules, clinical semantics, figures,
and manuscript workflows remain in downstream applications.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[all,dev,docs]'
python -m unittest discover
ruff check src tests
mkdocs build --strict
```

## Change requirements

- Temporal methods must prove that no operation crosses a censor gap.
- Learned methods must expose fit-subject identity and reject held-out overlap.
- Randomized methods must require or record a reproducible seed.
- Inferential APIs must identify the statistical unit, tail, and correction scope.
- New numerical kernels need known-value tests and, when migrated from a research
  implementation, a small redistributable regression fixture.
- Public examples must not contain private participant data or project-specific
  clinical semantics.

Use focused commits and explain any change to a mathematical or statistical
contract in both tests and user documentation.
