"""Attest — a local-model workbench where every operation has a trust signal.

This package is the *engine*. It contains all the real logic and is fully
testable without a GPU or a model (using the mock backend). Any UI — including
the future Mac desktop app — is just a thin client over this package.
"""

__version__ = "0.0.1"
