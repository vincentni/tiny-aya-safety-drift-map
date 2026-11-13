# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.14.1"]
# ///
"""Quant job for `hf jobs uv run`; the launcher adds the pinned driftmap install."""

import sys

from driftmap.scripts.quant import main

if __name__ == "__main__":
    sys.exit(main())
