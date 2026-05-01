"""Entry point for ``python -m src``.

Delegates immediately to :func:`src.app.main` and forwards its integer
exit code to the OS via :func:`SystemExit` so that callers (shells, CI
runners, etc.) can observe success/failure through the process exit status.
"""
from .app import main

if __name__ == "__main__":
    raise SystemExit(main())
