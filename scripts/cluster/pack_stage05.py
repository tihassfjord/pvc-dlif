"""Bundle everything stage 05 needs into one folder that can be copied to a cluster.

Stage 05 is the only stage that wants a GPU per fold for days, and it is also
the only stage whose inputs are small enough to move: the network inputs from
stage 03, the arterial curves, the manifest, the group's model code and this
package.  Nothing else - no DICOM, no NIfTI, no PVC output.

    python scripts/cluster/pack_stage05.py --out E:/ML4PET/cluster_bundle

What comes out:

    <out>/
      pvc-dlif/            this package (src, scripts, pyproject) - pip install -e it
      dlif_repo/           src/ of the group's repository + COMMIT file
      data/AIF_SUV/        ground-truth curves for the usable scans
      work/manifest.json
      work/dlif_inputs/    one tree per condition (float32 unless --keep-float64)
      configs/cluster.yaml paths relative to <out>; run every command from <out>
      run/                 job templates: SLURM array and Kubernetes, one job per fold

Copy the folder to the cluster (rsync/scp), create the environment, and submit
the jobs in run/.  When the jobs finish, copy <out>/work/models back and run
scripts/cluster/collect_stage05.py to merge it into the local work tree.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pvc_dlif.config import load_config                      # noqa: E402
from pvc_dlif.config_edit import set_value                   # noqa: E402
from pvc_dlif.data import pkl_io                             # noqa: E402
from pvc_dlif.data.manifest import load_manifest             # noqa: E402
from pvc_dlif.dlif.adapter import DlifRepo                   # noqa: E402


def _copy_inputs(src_root: Path, dst_root: Path, scan_ids: list[str], shape, float32: bool) -> int:
    """Copy one condition's IMG tree, optionally downcast to float32.

    The training loader casts to float32 anyway, so the downcast changes
    nothing numerically for stage 05 and halves the transfer.
    """
    copied = 0
    for scan_id in scan_ids:
        src = pkl_io.img_path(src_root, scan_id, shape)
        if not src.exists():
            continue
        dst = pkl_io.img_path(dst_root, scan_id, shape)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not float32:
            shutil.copy2(src, dst)
        else:
            with open(src, "rb") as handle:
                payload = pickle.load(handle)
            payload[pkl_io.IMG_KEY] = np.asarray(payload[pkl_io.IMG_KEY], dtype=np.float32)
            with open(dst, "wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        copied += 1
    return copied


def _write_lf(path: Path, text: str) -> Path:
    """Write a file the cluster's shell will read, with Unix line endings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="path to thesis.yaml")
    parser.add_argument("--out", required=True, help="bundle folder to create")
    parser.add_argument("--keep-float64", action="store_true",
                        help="copy the input pickles byte-for-byte instead of downcasting")
    parser.add_argument("--conditions", nargs="*", default=None,
                        help="retrained conditions to include (default: all)")
    args = parser.parse_args()

    config = load_config(args.config)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(config.work / "manifest.json")
    scan_ids = manifest.usable_ids
    shape = config.retrain_shape
    conditions = [c for c in config.conditions if c.model == "retrained" and not c.motion
                  and (args.conditions is None or c.name in set(args.conditions))]
    tags = sorted({c.input_tag for c in conditions})

    # ---- this package ------------------------------------------------------
    pkg = out / "pvc-dlif"
    if pkg.exists():
        shutil.rmtree(pkg)
    for item in ("src", "scripts", "pyproject.toml", "environment.yml", "README.md", "LICENSE"):
        source = REPO_ROOT / item
        if source.is_dir():
            shutil.copytree(source, pkg / item, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        elif source.exists():
            shutil.copy2(source, pkg / item)
    (pkg / "configs").mkdir(exist_ok=True)
    shutil.copy2(REPO_ROOT / "configs" / "example.yaml", pkg / "configs" / "example.yaml")

    # ---- the group's model code --------------------------------------------
    repo = DlifRepo(config.dlif_repo)
    repo_dst = out / "dlif_repo"
    if repo_dst.exists():
        shutil.rmtree(repo_dst)
    shutil.copytree(repo.src, repo_dst / "src",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info", "checkpoints", "*.ipynb"))
    (repo_dst / "COMMIT").write_text((repo.commit() or "unknown") + "\n", encoding="utf-8")

    # ---- data ----------------------------------------------------------------
    (out / "data" / "AIF_SUV").mkdir(parents=True, exist_ok=True)
    n_aif = 0
    for scan_id in scan_ids:
        src = pkl_io.aif_path(config.dlif_data_root, scan_id)
        if src.exists():
            shutil.copy2(src, out / "data" / "AIF_SUV" / src.name)
            n_aif += 1

    (out / "work").mkdir(exist_ok=True)
    shutil.copy2(config.work / "manifest.json", out / "work" / "manifest.json")
    counts = {}
    for tag in tags:
        counts[tag] = _copy_inputs(config.dir_dlif_inputs / tag, out / "work" / "dlif_inputs" / tag,
                                   scan_ids, shape, float32=not args.keep_float64)

    # ---- config with bundle-relative paths ---------------------------------
    # The live config, verbatim, with only the paths rewritten: every training
    # setting the owner chose travels with the bundle.
    text = Path(config.source).read_text(encoding="utf-8")
    for dotted, rel in (("paths.dicom_root", "unused"), ("paths.dlif_data_root", "data"),
                        ("paths.dlif_repo", "dlif_repo"), ("paths.work", "work")):
        text = set_value(text, dotted, rel)
    text = set_value(text, "dlif.train.device", "cuda")
    (out / "configs").mkdir(exist_ok=True)
    (out / "configs" / "cluster.yaml").write_text(text, encoding="utf-8")

    # ---- job templates -------------------------------------------------------
    (out / "run").mkdir(exist_ok=True)
    for name in ("stage05_slurm.sh", "stage05_k8s.yaml", "stage05_one_fold.sh"):
        shutil.copy2(HERE / name, out / "run" / name)
    # newline="\n" matters: these are read by a shell on the cluster, and by
    # one inside the job's container.  Packed from Windows without it, Python
    # writes CRLF, and the reader gets "10\r" as a fold count and
    # "rl_retrained\r" as a condition name - which fails as an arithmetic
    # error in one place and, worse, as a silently wrong condition in another.
    _write_lf(out / "run" / "conditions.txt", "\n".join(c.name for c in conditions) + "\n")
    _write_lf(out / "run" / "n_folds.txt", str(config.get("dlif.cv.n_folds", 10)) + "\n")

    # The input pickles carry the module path of the numpy that wrote them, and
    # that path changed between generations: numpy 2 pickles reference
    # `numpy._core`, which numpy 1 cannot import, and the failure is a
    # ModuleNotFoundError on the first batch rather than anything about
    # versions.  Downcasting to float32 rewrites every pickle with the packing
    # machine's numpy, so the cluster has to match this generation - pin it
    # from here rather than leaving it to whatever the image happens to ship.
    major = int(np.__version__.split(".")[0])
    _write_lf(
        out / "run" / "requirements.txt",
        f"# numpy {np.__version__} wrote the input pickles in this bundle.\n"
        f"numpy>={major}.0,<{major + 1}\n",
    )

    # ---- report --------------------------------------------------------------
    size_gb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1024 ** 3
    summary = {
        "bundle": str(out),
        "scans": len(scan_ids),
        "aif_curves": n_aif,
        "input_trees": counts,
        "conditions": [c.name for c in conditions],
        "folds": int(config.get("dlif.cv.n_folds", 10)),
        "runs_per_fold": int(config.get("dlif.cv.n_runs", 10)),
        "epochs": int(config.get("dlif.train.epochs", 1000)),
        "dlif_repo_commit": repo.commit(),
        "float32": not args.keep_float64,
        "size_gb": round(size_gb, 2),
        "jobs": len(conditions) * int(config.get("dlif.cv.n_folds", 10)),
    }
    (out / "bundle.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
