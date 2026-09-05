"""Main window.

    python gui/run_gui.py          # from a clone
    pvc-dlif-gui                   # once the package is installed

The window is three tabs and a status bar; each tab is its own module.  This
file only wires them together and works out where the repository is, so the
Pipeline tab can find ``scripts/``.
"""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk


def find_repo_root() -> Path:
    """The clone this GUI belongs to: the folder holding scripts/ and configs/.

    Walk up from this file first (the normal case, running from a clone), then
    fall back to the current directory for an installed package.
    """
    for candidate in [Path(__file__).resolve(), *Path(__file__).resolve().parents, Path.cwd()]:
        folder = candidate if candidate.is_dir() else candidate.parent
        if (folder / "scripts").is_dir() and (folder / "configs").is_dir():
            return folder
    return Path.cwd()


class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("pvc-dlif — partial volume correction for DLIF")
        self.geometry("1060x780")
        self.minsize(920, 640)

        self.repo_root = find_repo_root()
        _ensure_package_importable(self.repo_root)

        # ttk's default theme is ugly on Windows; 'vista'/'clam' are not.
        style = ttk.Style(self)
        for theme in ("vista", "clam", "default"):
            if theme in style.theme_names():
                style.theme_use(theme)
                break

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=6, pady=6)

        # Imported here so a missing optional dependency names the tab it broke.
        from .tab_analysis import AnalysisTab
        from .tab_pipeline import PipelineTab
        from .tab_pvc import PVCTab

        self.pvc_tab = PVCTab(notebook, self)
        self.pipeline_tab = PipelineTab(notebook, self)
        self.analysis_tab = AnalysisTab(notebook, self)
        notebook.add(self.pvc_tab, text="  PVC  ")
        notebook.add(self.pipeline_tab, text="  Pipeline  ")
        notebook.add(self.analysis_tab, text="  Analysis  ")

        self.status = ttk.Label(self, relief="sunken", anchor="w", padding=(6, 2),
                                text=f"repository: {self.repo_root}   |   {_backend_status()}")
        self.status.pack(fill="x", side="bottom")

        self.protocol("WM_DELETE_WINDOW", self._close)

    def _close(self) -> None:
        """Warn before quitting while something is still running."""
        busy = [name for name, tab in (("PVC", self.pvc_tab), ("Pipeline", self.pipeline_tab))
                if getattr(tab, "job", None) is not None and tab.job.running]   # type: ignore[union-attr]
        if busy and not messagebox.askyesno(
                "Still running", f"{', '.join(busy)} still running. Quit anyway?"):
            return
        self.destroy()


def _ensure_package_importable(repo_root: Path) -> None:
    """Allow running from a clone without `pip install -e .`."""
    src = repo_root / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _backend_status() -> str:
    """One-line note about what is and is not available, shown in the status bar."""
    import shutil
    parts = ["PETPVC: " + ("found" if shutil.which("petpvc") else "not on PATH (numpy fallback)")]
    try:
        import torch
        parts.append("torch: " + ("CUDA" if torch.cuda.is_available() else "CPU only"))
    except ImportError:
        parts.append("torch: not installed (stages 04/05 unavailable)")
    return "   |   ".join(parts)


def main() -> int:
    App().mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
