"""Small reusable widgets: tooltips, a log pane, path pickers, a table view.

Kept deliberately plain - plain ttk, no theming tricks - so the tab modules
read as layout rather than as widget plumbing.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk


class ToolTip:
    """Hover help for any widget. Same pattern as the phantom GUI."""

    def __init__(self, widget, text: str, delay: int = 500):
        self.widget, self.text, self.delay = widget, text, delay
        self.tip = None
        self._after_id = None
        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self._hide)

    def _schedule(self, _event=None):
        self._cancel()
        self._after_id = self.widget.after(self.delay, self._show)

    def _cancel(self):
        if self._after_id:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self):
        if self.tip:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)          # no title bar
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, justify="left", background="#ffffe0",
                 relief="solid", borderwidth=1, font=("tahoma", 8),
                 wraplength=420).pack()

    def _hide(self, _event=None):
        self._cancel()
        if self.tip:
            self.tip.destroy()
            self.tip = None


class LogPane(ttk.Frame):
    """Read-only scrolling text box that every long-running job writes into."""

    def __init__(self, parent, height: int = 12):
        super().__init__(parent)
        self.text = tk.Text(self, height=height, wrap="none", state="disabled",
                            font=("Consolas", 9), background="#1e1e1e",
                            foreground="#d4d4d4", insertbackground="#d4d4d4")
        bar = ttk.Scrollbar(self, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=bar.set)
        self.text.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")

        # Colour by severity; the tag is chosen in write().
        self.text.tag_configure("error", foreground="#f48771")
        self.text.tag_configure("warn", foreground="#dcdcaa")
        self.text.tag_configure("ok", foreground="#89d185")

    def write(self, message: str, tag: str | None = None) -> None:
        """Append a line. Safe to call from the Tk thread only."""
        if tag is None:
            lowered = message.lower()
            tag = ("error" if "error" in lowered or "failed" in lowered
                   else "warn" if "warn" in lowered
                   else None)
        self.text.configure(state="normal")
        self.text.insert("end", message.rstrip() + "\n", tag or ())
        self.text.see("end")
        self.text.configure(state="disabled")

    def clear(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")


class PathPicker(ttk.Frame):
    """Label + entry + Browse button. ``kind`` is 'file', 'files' or 'dir'."""

    def __init__(self, parent, label: str, kind: str = "file", value: str = "",
                 tooltip: str = "", filetypes=(("NIfTI", "*.nii *.nii.gz"), ("All", "*.*")),
                 width: int = 52):
        super().__init__(parent)
        self.kind, self.filetypes = kind, filetypes
        self.var = tk.StringVar(value=value)

        lbl = ttk.Label(self, text=label, width=16, anchor="w")
        lbl.pack(side="left")
        entry = ttk.Entry(self, textvariable=self.var, width=width)
        entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(self, text="Browse…", width=10, command=self._browse).pack(side="left")
        if tooltip:
            ToolTip(lbl, tooltip)
            ToolTip(entry, tooltip)

    def _browse(self):
        if self.kind == "dir":
            chosen = filedialog.askdirectory(title="Select folder")
        elif self.kind == "files":
            picked = filedialog.askopenfilenames(title="Select files", filetypes=self.filetypes)
            chosen = ";".join(picked) if picked else ""
        else:
            chosen = filedialog.askopenfilename(title="Select file", filetypes=self.filetypes)
        if chosen:
            self.var.set(chosen)

    def get(self) -> str:
        return self.var.get().strip()

    def path(self) -> Path | None:
        value = self.get()
        return Path(value) if value else None

    def paths(self) -> list[Path]:
        """For kind='files': split the semicolon-joined list."""
        return [Path(p) for p in self.get().split(";") if p.strip()]


class LabelledEntry(ttk.Frame):
    """Label + narrow entry, for a single parameter value."""

    def __init__(self, parent, label: str, value, width: int = 8, tooltip: str = ""):
        super().__init__(parent)
        self.var = tk.StringVar(value=str(value))
        lbl = ttk.Label(self, text=label)
        lbl.pack(side="left", padx=(0, 4))
        entry = ttk.Entry(self, textvariable=self.var, width=width)
        entry.pack(side="left")
        if tooltip:
            ToolTip(lbl, tooltip)
            ToolTip(entry, tooltip)

    def get(self) -> str:
        return self.var.get().strip()

    def as_int(self, default: int = 0) -> int:
        try:
            return int(float(self.get()))
        except ValueError:
            return default

    def as_float(self, default: float = 0.0) -> float:
        try:
            return float(self.get())
        except ValueError:
            return default


class TableView(ttk.Frame):
    """Treeview that displays a pandas DataFrame without needing pandas here."""

    def __init__(self, parent, height: int = 10):
        super().__init__(parent)
        self.tree = ttk.Treeview(self, show="headings", height=height)
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

    def show(self, frame, max_rows: int = 500, digits: int = 4) -> None:
        """Replace the contents with the first ``max_rows`` of a DataFrame."""
        self.tree.delete(*self.tree.get_children())
        columns = [str(c) for c in frame.columns]
        self.tree["columns"] = columns
        for name in columns:
            self.tree.heading(name, text=name)
            self.tree.column(name, width=max(70, min(190, 9 * len(name) + 40)),
                             anchor="center", stretch=False)
        for _, row in frame.head(max_rows).iterrows():
            self.tree.insert("", "end", values=[_fmt(v, digits) for v in row])


def _fmt(value, digits: int) -> str:
    """Numbers get rounded; everything else is str()'d."""
    if isinstance(value, float):
        if value != value:                       # NaN
            return ""
        return f"{value:.{digits}g}"
    return str(value)
