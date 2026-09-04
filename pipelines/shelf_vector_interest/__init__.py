"""Shelf-face-vector customer interest pipeline package.

Self-contained: does not import from `pipelines/shelf_interest` or any other
sibling pipeline, per the project convention in `pipelines/README.md`. It
only shares the generic model infra (`src/analytics/pose.py`,
`src/reid/embedder.py`) and its own config file.
"""
