#!/usr/bin/env python3
"""Launch the GUI from a clone, without installing the package.

    python gui/run_gui.py
"""

import sys
from pathlib import Path

# Put src/ and gui/ on the path so pvc_dlif and pvc_dlif_gui both import.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

from pvc_dlif_gui.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
