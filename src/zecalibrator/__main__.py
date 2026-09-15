"""``python -m zecalibrator`` entry point (routes to the CLI only)."""

from __future__ import annotations

import sys

from zecalibrator.cli import main

if __name__ == "__main__":
    sys.exit(main())
