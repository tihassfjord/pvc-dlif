"""Small reusable widgets: tooltips, a log pane, path pickers, a table view.

Kept deliberately plain - plain ttk, no theming tricks - so the tab modules
read as layout rather than as widget plumbing.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


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
    """Treeview that displays a pandas DataFrame, sortable by clicking a header.

    Keeps the frame it was given so sorting and CSV export work on the real
    values, not on the rounded strings shown in the cells.
    """

    def __init__(self, parent, height: int = 10, with_export: bool = True):
        super().__init__(parent)
        self.frame = None
        self._sort_desc: dict[str, bool] = {}

        if with_export:
            bar = ttk.Frame(self)
            bar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 3))
            ttk.Button(bar, text="Export CSV…", command=self.export_csv).pack(side="right")
            self.info = ttk.Label(bar, text="", foreground="#666")
            self.info.pack(side="left")
        else:
            self.info = None

        self.tree = ttk.Treeview(self, show="headings", height=height)
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.grid(row=1, column=0, sticky="nsew")
        vsb.grid(row=1, column=1, sticky="ns")
        hsb.grid(row=2, column=0, sticky="ew")
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

    def show(self, frame, max_rows: int = 2000, digits: int = 4) -> None:
        """Replace the contents with a DataFrame (first ``max_rows`` rows)."""
        self.frame = frame
        self._digits = digits
        self._max_rows = max_rows
        self._render()

    def _render(self) -> None:
        frame = self.frame
        self.tree.delete(*self.tree.get_children())
        if frame is None:
            return
        columns = [str(c) for c in frame.columns]
        self.tree["columns"] = columns
        for name in columns:
            self.tree.heading(name, text=name, command=lambda c=name: self.sort_by(c))
            self.tree.column(name, width=max(70, min(190, 9 * len(name) + 40)),
                             anchor="center", stretch=False)
        for _, row in frame.head(self._max_rows).iterrows():
            self.tree.insert("", "end", values=[_fmt(v, self._digits) for v in row])
        if self.info is not None:
            shown = min(len(frame), self._max_rows)
            self.info.configure(text=f"{len(frame)} rows" + (f" (showing {shown})" if shown < len(frame) else ""))

    def sort_by(self, column: str) -> None:
        """Toggle ascending/descending on a column and redraw."""
        if self.frame is None:
            return
        descending = self._sort_desc.get(column, False)
        self.frame = self.frame.sort_values(column, ascending=not descending, kind="mergesort")
        self._sort_desc[column] = not descending
        self._render()

    def export_csv(self) -> None:
        if self.frame is None:
            messagebox.showinfo("Nothing to export", "Load a table first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=(("CSV", "*.csv"),))
        if path:
            self.frame.to_csv(path, index=False)


class Banner(ttk.Frame):
    """A coloured strip for results that must not be missed.

    ``kind`` is 'ok', 'warn', 'error' or 'info'.  Hidden until :meth:`show`
    is called; :meth:`hide` removes it again.
    """

    COLOURS = {
        "ok": ("#dff0d8", "#3c763d"),
        "warn": ("#fcf8e3", "#8a6d3b"),
        "error": ("#f2dede", "#a94442"),
        "info": ("#d9edf7", "#31708f"),
    }

    def __init__(self, parent, before=None):
        super().__init__(parent)
        self.label = tk.Label(self, anchor="w", justify="left", padx=10, pady=6,
                              font=("Segoe UI", 10, "bold"), wraplength=900)
        self.label.pack(fill="x")
        self._before = before          # widget to pack in front of, if any
        self._shown = False

    def show(self, text: str, kind: str = "info") -> None:
        background, foreground = self.COLOURS.get(kind, self.COLOURS["info"])
        self.label.configure(text=text, background=background, foreground=foreground)
        if not self._shown:
            if self._before is not None:
                self.pack(fill="x", pady=(6, 0), before=self._before)
            else:
                self.pack(fill="x", pady=(6, 0))
            self._shown = True

    def hide(self) -> None:
        if self._shown:
            self.pack_forget()
            self._shown = False


class CheckList(ttk.Frame):
    """Scrollable list of check boxes with select-all / none buttons."""

    def __init__(self, parent, height: int = 10, columns: int = 4):
        super().__init__(parent)
        self.columns = columns
        self.vars: dict[str, tk.BooleanVar] = {}

        bar = ttk.Frame(self)
        bar.pack(fill="x")
        ttk.Button(bar, text="All", width=6, command=lambda: self.set_all(True)).pack(side="left")
        ttk.Button(bar, text="None", width=6, command=lambda: self.set_all(False)).pack(side="left", padx=4)
        self.count = ttk.Label(bar, text="")
        self.count.pack(side="left", padx=8)

        canvas = tk.Canvas(self, height=height * 22, highlightthickness=0)
        bar_y = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.inner = ttk.Frame(canvas)
        self.inner.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.inner, anchor="nw")
        canvas.configure(yscrollcommand=bar_y.set)
        canvas.pack(side="left", fill="both", expand=True)
        bar_y.pack(side="right", fill="y")

    def set_items(self, items: list[str], checked: bool = True) -> None:
        for child in self.inner.winfo_children():
            child.destroy()
        self.vars.clear()
        for index, item in enumerate(items):
            var = tk.BooleanVar(value=checked)
            var.trace_add("write", lambda *_: self._update_count())
            self.vars[item] = var
            ttk.Checkbutton(self.inner, text=item, variable=var).grid(
                row=index // self.columns, column=index % self.columns, sticky="w", padx=6)
        self._update_count()

    def set_all(self, value: bool) -> None:
        for var in self.vars.values():
            var.set(value)

    def selected(self) -> list[str]:
        return [item for item, var in self.vars.items() if var.get()]

    def _update_count(self) -> None:
        self.count.configure(text=f"{len(self.selected())} of {len(self.vars)} selected")


def _fmt(value, digits: int) -> str:
    """Numbers get rounded; everything else is str()'d."""
    if isinstance(value, float):
        if value != value:                       # NaN
            return ""
        return f"{value:.{digits}g}"
    return str(value)
