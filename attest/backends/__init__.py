"""Backends implement the Embedder/Generator interfaces.

- `mock`  — deterministic fakes, no model. Used to build & test on Linux.
- `mlx`   — the real Apple Silicon backend (added when we're on the Mac).
"""
