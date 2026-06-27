"""Shim so `pip install -e .` works on older setuptools (pre-PEP 660).

All real configuration lives in pyproject.toml; this just lets legacy editable
installs (setup.py develop) succeed on environments whose setuptools is too old
to support the modern build_editable hook.
"""

from setuptools import setup

setup()
