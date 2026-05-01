"""CLI sub-package.

Exposes argument parsing (``parse_args``) and the resolved path model
(``AppPaths``) as the public surface of the command-line interface layer.
All argv handling lives here so that the rest of the codebase can remain
ignorant of how paths are supplied.
"""
from .paths import AppPaths
from .args import parse_args

__all__ = [
    "AppPaths",
    "parse_args",
]
