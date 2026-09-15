"""ZeCalibrator command-line interface (bootstrap skeleton).

This bootstrap exposes only ``--help`` and ``--version``. No calibration,
matching or other scientific capability is implemented or advertised here.
"""

from __future__ import annotations

import argparse

from zecalibrator._version import __version__

PROG = "zecalibrator"

_DESCRIPTION = (
    "ZeCalibrator — raw astronomical sensor FITS calibration engine.\n\n"
    "Bootstrap skeleton (Phase 2): package layout, version, resources and "
    "storage plumbing only. No calibration, matching or other scientific "
    "capability is implemented or advertised in this release."
)


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser (help/version only)."""
    parser = argparse.ArgumentParser(prog=PROG, description=_DESCRIPTION)
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="show program's version number and exit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    parser.parse_args(argv)
    # No subcommands are implemented; invoking with no arguments shows help.
    parser.print_help()
    return 0


__all__ = ["PROG", "build_parser", "main"]
