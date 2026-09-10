"""Configuration loading for the thesis pipeline.

Every stage reads ``configs/thesis.yaml``.  Nothing else in the package holds
paths, PSF values, iteration counts or statistical settings, so a change to the
YAML changes the whole pipeline consistently.

Usage::

    from pvc_dlif.config import load_config
    cfg = load_config()                    # finds configs/thesis.yaml
    cfg = load_config("my_variant.yaml")   # or an explicit file
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

__all__ = ["Config", "Condition", "load_config", "find_config"]


def _normalise_path(value: Any) -> Path | None:
    """Turn a config path string into a :class:`Path`.

    Windows-style backslashes and forward slashes both work, on either OS.
    ``None`` and the empty string map to ``None`` so optional paths stay
    optional.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = os.path.expandvars(os.path.expanduser(text))
    return Path(text.replace("\\", "/"))


@dataclass(frozen=True)
class Condition:
    """One experimental condition: an input representation plus a model.

    A condition is the unit the paired statistics operate on -- every scan
    contributes exactly one predicted input function per condition.
    """

    name: str
    pvc_method: str | None
    pvc_iterations: int | None
    motion: bool
    model: str
    subset: str | None = None

    @property
    def is_corrected(self) -> bool:
        return self.pvc_method is not None

    @property
    def input_tag(self) -> str:
        """Directory tag identifying the *input series* this condition uses.

        Conditions that differ only in which model is applied share one input
        tag, so the corrected images are generated once and reused.
        """
        if self.pvc_method is None:
            tag = "orig"
        else:
            tag = f"{self.pvc_method.lower()}_i{self.pvc_iterations}"
        if self.motion:
            tag = f"mc_{tag}"
        return tag

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], default_iterations: int | None = None) -> "Condition":
        pvc = raw.get("pvc") or {}
        has_pvc = bool(pvc.get("method"))
        iterations = pvc.get("iterations", default_iterations if has_pvc else None)
        return cls(
            name=str(raw["name"]),
            pvc_method=(str(pvc["method"]).upper() if has_pvc else None),
            pvc_iterations=(int(iterations) if iterations is not None else None),
            motion=bool(raw.get("motion", False)),
            model=str(raw.get("model", "pretrained")),
            subset=(str(raw["subset"]) if raw.get("subset") else None),
        )


class Config:
    """Thin, attribute-friendly wrapper around the parsed YAML.

    Raw access stays available through :meth:`get` and ``cfg.raw`` so nothing
    is hidden, while the commonly used values are exposed as properties with
    the right types.
    """

    def __init__(self, raw: Mapping[str, Any], source: Path | None = None):
        self.raw: dict[str, Any] = copy.deepcopy(dict(raw))
        self.source = source
        self.regime: str = self._apply_regime()
        self._validate()

    # ------------------------------------------------------------------ #
    # training regime
    # ------------------------------------------------------------------ #
    def _apply_regime(self) -> str:
        """Merge the selected training regime into ``dlif.cv`` and ``dlif.train``.

        The two DLIF papers train the same family of model very differently:
        Kuttner et al. 2024 uses 17-fold cross-validation, one run per fold,
        200 epochs at 2e-4 with a plain MSE loss; the 2026 protocol uses 10
        folds, 10 runs, 1000 epochs at 1e-4 with a weighted MSE.  Which one
        applies is a study-design decision, not a hyperparameter to tune, so it
        is named once here and the merge happens at load: every existing reader
        of ``dlif.cv.*`` and ``dlif.train.*`` then sees the resolved values and
        the provenance record shows what was actually used.

        A regime patch only overrides the keys it names, so anything the
        published protocol does not speak to (device, amp, num_workers) keeps
        the base value.
        """
        dlif = self.raw.get("dlif")
        if not isinstance(dlif, Mapping):
            return "base"
        name = dlif.get("regime")
        if name is None:
            return "base"

        name = str(name)
        regimes = dlif.get("regimes") or {}
        if name not in regimes:
            known = ", ".join(sorted(str(k) for k in regimes)) or "none defined"
            raise ValueError(
                f"dlif.regime is '{name}' but no such regime exists. Known: {known}"
            )

        patch = regimes[name] or {}
        for section in ("cv", "train"):
            values = patch.get(section)
            if not isinstance(values, Mapping):
                continue
            target = self.raw["dlif"].setdefault(section, {})
            if not isinstance(target, dict):
                target = {}
                self.raw["dlif"][section] = target
            target.update(copy.deepcopy(dict(values)))
        return name

    # ------------------------------------------------------------------ #
    # generic access
    # ------------------------------------------------------------------ #
    def get(self, dotted: str, default: Any = None) -> Any:
        """Fetch a nested value, e.g. ``cfg.get("pvc.iterations_grid")``."""
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    # ------------------------------------------------------------------ #
    # paths
    # ------------------------------------------------------------------ #
    @property
    def dicom_root(self) -> Path:
        return self._require_path("paths.dicom_root")

    @property
    def dlif_data_root(self) -> Path:
        return self._require_path("paths.dlif_data_root")

    @property
    def dlif_repo(self) -> Path:
        return self._require_path("paths.dlif_repo")

    @property
    def work(self) -> Path:
        return self._require_path("paths.work")

    @property
    def petpvc_exe(self) -> Path | None:
        return _normalise_path(self.get("paths.petpvc_exe"))

    @property
    def falcon_exe(self) -> Path | None:
        return _normalise_path(self.get("paths.falcon_exe"))

    def _require_path(self, dotted: str) -> Path:
        path = _normalise_path(self.get(dotted))
        if path is None:
            raise KeyError(f"Required path '{dotted}' is not set in {self.source}")
        return path

    # Derived work-tree locations.  Created lazily by the stages that use them.
    @property
    def dir_native(self) -> Path:
        """4D NIfTI series at the reconstructed voxel grid, per scan."""
        return self.work / "native"

    @property
    def dir_pvc(self) -> Path:
        """PVC-corrected 4D series, one subdirectory per input tag."""
        return self.work / "pvc"

    @property
    def dir_motion(self) -> Path:
        return self.work / "motion"

    @property
    def dir_dlif_inputs(self) -> Path:
        """DLIF-ready ``IMG_<ID>.pkl`` trees, one per input tag."""
        return self.work / "dlif_inputs"

    @property
    def dir_models(self) -> Path:
        return self.work / "models"

    @property
    def dir_predictions(self) -> Path:
        return self.work / "predictions"

    @property
    def dir_results(self) -> Path:
        return self.work / "results"

    @property
    def dir_report(self) -> Path:
        return self.work / str(self.get("report.out_subdir", "report"))

    @property
    def dir_logs(self) -> Path:
        return self.work / "logs"

    # ------------------------------------------------------------------ #
    # typed shortcuts
    # ------------------------------------------------------------------ #
    @property
    def seed(self) -> int:
        return int(self.get("project.seed", 42))

    @property
    def fwhm_mm(self) -> tuple[float, float, float]:
        values = self.get("psf.fwhm_mm")
        if values is None or len(values) != 3:
            raise ValueError("psf.fwhm_mm must be a list of three values (x, y, z) in mm")
        x, y, z = (float(v) for v in values)
        return x, y, z

    @property
    def pvc_methods(self) -> list[str]:
        return [str(m).upper() for m in self.get("pvc.methods", ["RL", "RVC"])]

    @property
    def iterations_grid(self) -> list[int]:
        return [int(k) for k in self.get("pvc.iterations_grid", [10, 15, 20])]

    @property
    def iterations_primary(self) -> int:
        return int(self.get("pvc.iterations_primary", 15))

    @property
    def dlif_shapes(self) -> list[tuple[int, int, int]]:
        """Every input shape an input tree is built for."""
        shapes = self.get("dlif_grid.shapes")
        if not shapes:
            single = self.get("dlif_grid.shape", [96, 48, 48])
            shapes = [single]
        return [tuple(int(v) for v in shape) for shape in shapes]  # type: ignore[misc]

    @property
    def reference_shape(self) -> tuple[int, int, int]:
        """The shape the distributed inputs use, which the grid is calibrated on."""
        shape = self.get("dlif_grid.reference_shape") or self.dlif_shapes[-1]
        a, b, c = (int(v) for v in shape)
        return a, b, c

    # Kept for call sites that only need one shape; it is the reference tree.
    @property
    def dlif_shape(self) -> tuple[int, int, int]:
        return self.reference_shape

    @property
    def pretrained_shape(self) -> tuple[int, int, int]:
        shape = self.get("dlif.pretrained.input_shape", self.reference_shape)
        a, b, c = (int(v) for v in shape)
        return a, b, c

    @property
    def pretrained_add_average(self) -> bool:
        return bool(self.get("dlif.pretrained.add_average", False))

    @property
    def pretrained_weights(self) -> str:
        return str(self.get("dlif.pretrained.weights", "src/models/pretrained_weigths/DLIFNet.pt"))

    @property
    def retrain_shape(self) -> tuple[int, int, int]:
        shape = self.get("dlif.retrain_model.input_shape", self.reference_shape)
        a, b, c = (int(v) for v in shape)
        return a, b, c

    @property
    def retrain_add_average(self) -> bool:
        return bool(self.get("dlif.retrain_model.add_average", False))

    @property
    def retrain_model_name(self) -> str:
        return str(self.get("dlif.retrain_model.model_name", "DLIFNet_MAX"))

    @property
    def retrain_in_channels(self) -> int:
        return int(self.get("dlif.retrain_model.in_channels", 1))

    @property
    def n_frames(self) -> int:
        return int(self.get("dlif.n_frames", 42))

    @property
    def ignore_ids(self) -> set[str]:
        return {str(i) for i in self.get("dlif.ignore_ids", [])}

    @property
    def conditions(self) -> list[Condition]:
        """The experimental conditions, with the iteration count filled in.

        A condition may leave ``iterations`` unset, in which case it tracks
        ``pvc.iterations_primary``.  That keeps the primary setting in one place:
        changing it there cannot leave the conditions pointing at a correction
        the PVC stage was never asked to produce.
        """
        primary = self.iterations_primary
        return [Condition.from_dict(c, default_iterations=primary) for c in self.get("conditions", [])]

    def retrained_conditions(
        self,
        *,
        include_motion: bool = False,
        names: Iterable[str] | None = None,
    ) -> list[Condition]:
        """The conditions stage 05 trains, and that stage 06 then compares.

        Motion conditions are excluded by default.  They belong to stage 07:
        they are exploratory, they are scored on the motion-affected subset
        rather than the whole cohort, and stage 05 has no notion of ``subset``,
        so training them with the main grid would silently fit them to every
        scan.  Naming one explicitly still selects it.

        Stage 05 and the status report must agree on this set, or the progress
        fraction counts a different grid from the one being trained.
        """
        wanted = set(names) if names is not None else None
        return [
            c for c in self.conditions
            if c.model == "retrained"
            and (wanted is None or c.name in wanted)
            and (not c.motion or include_motion or wanted is not None)
        ]

    def condition(self, name: str) -> Condition:
        for cond in self.conditions:
            if cond.name == name:
                return cond
        raise KeyError(f"Unknown condition '{name}'. Known: {[c.name for c in self.conditions]}")

    @property
    def reference_condition(self) -> Condition:
        return self.condition(str(self.get("reference_condition", "baseline_pretrained")))

    @property
    def motion_affected_ids(self) -> list[str]:
        return [str(i) for i in self.get("motion.affected_ids", [])]

    # ------------------------------------------------------------------ #
    # validation
    # ------------------------------------------------------------------ #
    def _validate(self) -> None:
        problems: list[str] = []

        for required in ("paths", "psf", "pvc", "dlif", "conditions"):
            if required not in self.raw:
                problems.append(f"missing top-level section '{required}'")

        fwhm = self.get("psf.fwhm_mm")
        if fwhm is not None:
            if len(fwhm) != 3 or any(float(v) <= 0 for v in fwhm):
                problems.append("psf.fwhm_mm must be three positive numbers (x, y, z) in mm")

        domain = self.get("pvc.domain", "native")
        if domain not in {"native", "dlif_grid"}:
            problems.append(f"pvc.domain must be 'native' or 'dlif_grid', got {domain!r}")

        names = [c.get("name") for c in self.get("conditions", [])]
        if len(names) != len(set(names)):
            problems.append("condition names must be unique")

        ref = self.get("reference_condition")
        if ref is not None and ref not in names:
            problems.append(f"reference_condition {ref!r} is not one of the defined conditions")

        primary = int(self.get("pvc.iterations_primary", 15))
        produced = set(self.get("pvc.iterations_grid", [])) | {primary}
        configured_methods = {str(m).upper() for m in self.get("pvc.methods", [])}

        for cond in self.get("conditions", []):
            pvc = cond.get("pvc")
            if not pvc:
                continue
            method = str(pvc.get("method", "")).upper()
            if method not in {"RL", "RVC", "VC"}:
                problems.append(
                    f"condition {cond.get('name')!r}: pvc.method {method!r} is not an "
                    "iterative deconvolution method (expected RL or RVC)"
                )
            elif configured_methods and method not in configured_methods:
                problems.append(
                    f"condition {cond.get('name')!r} uses {method}, which is not in "
                    f"pvc.methods ({sorted(configured_methods)}); the PVC stage would never "
                    "produce its input"
                )
            iterations = pvc.get("iterations", primary)
            if iterations is not None and int(iterations) not in produced:
                problems.append(
                    f"condition {cond.get('name')!r} asks for {iterations} iterations, but the "
                    f"PVC stage only produces {sorted(produced)}; set pvc.iterations_primary or "
                    "add it to pvc.iterations_grid"
                )

        if problems:
            joined = "\n  - ".join(problems)
            raise ValueError(f"Invalid configuration in {self.source}:\n  - {joined}")

    # ------------------------------------------------------------------ #
    def ensure_work_tree(self, subdirs: Iterable[Path] | None = None) -> None:
        """Create the standard work-tree directories."""
        targets: Sequence[Path] = list(subdirs) if subdirs is not None else [
            self.dir_native,
            self.dir_pvc,
            self.dir_motion,
            self.dir_dlif_inputs,
            self.dir_models,
            self.dir_predictions,
            self.dir_results,
            self.dir_report,
            self.dir_logs,
        ]
        for target in targets:
            target.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"Config(source={self.source}, work={self.get('paths.work')})"


def find_config(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Locate ``thesis.yaml``.

    Search order: the explicit argument, ``$ML4PET_THESIS_CONFIG``, then
    ``configs/thesis.yaml`` walking up from this file and from the working
    directory.
    """
    if explicit is not None:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        return path

    env = os.environ.get("ML4PET_THESIS_CONFIG")
    if env:
        path = Path(env)
        if not path.exists():
            raise FileNotFoundError(f"$ML4PET_THESIS_CONFIG points at a missing file: {path}")
        return path

    starts = [Path(__file__).resolve().parent, Path.cwd().resolve()]
    seen: set[Path] = set()
    for start in starts:
        for parent in [start, *start.parents]:
            if parent in seen:
                continue
            seen.add(parent)
            candidate = parent / "configs" / "thesis.yaml"
            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        "Could not find configs/thesis.yaml. Pass --config explicitly or set "
        "$ML4PET_THESIS_CONFIG."
    )


def load_config(path: str | os.PathLike[str] | None = None, **overrides: Any) -> Config:
    """Load and validate the pipeline configuration.

    ``overrides`` are applied as dotted keys, e.g.
    ``load_config(**{"pvc.iterations_primary": 20})``.
    """
    resolved = find_config(path)
    with open(resolved, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    for dotted, value in overrides.items():
        node = raw
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    return Config(raw, source=resolved)
