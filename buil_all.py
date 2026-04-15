"""Backward-compatible wrapper for build_all.py."""

from build_all import main


if __name__ == "__main__":
    raise SystemExit(main())
