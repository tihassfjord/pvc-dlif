"""QC -- draw the stage 01 crop window on the native image.

This answers a different question from ``preprocessing_agreement.csv``.  That
table asks whether our window reproduces the one the distributed inputs were
made with; it would report perfect agreement even if that window were badly
placed, because it only compares us against them.  This script asks whether the
window sits on the animal at all: it draws the box on the uncropped native
volume in three orthogonal planes, and puts the cropped result beside it.

What a good result looks like: the box contains the whole animal along the long
axis, the bright blood pool sits inside it with margin on every side, and the
cropped panel on the right shows the same anatomy as the region inside the box.
What a bad one looks like: the box clipping the animal at an edge, or the blood
pool sitting against a face of the box rather than away from it.

    python scripts/qc_crop_window.py --config configs/thesis.yaml
    python scripts/qc_crop_window.py --scans AA1 P3 AV1 --frames 30 41
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from _common import base_parser, start

from pvc_dlif.data.dicom_io import read_nifti_4d
from pvc_dlif.data.manifest import load_manifest
from pvc_dlif.data.preprocess import load_plan
from pvc_dlif.logging_utils import get_logger

LOGGER = get_logger("qc.crop")


def _panel(ax, image, title, box=None):
    """One greyscale panel, optionally with the crop window drawn on it.

    ``image`` is indexed ``[row, column]`` and drawn with ``origin="lower"``,
    so a box is given as ``((column_start, column_stop), (row_start, row_stop))``.
    """
    from matplotlib.patches import Rectangle

    ax.imshow(image, cmap="gray", origin="lower",
              vmin=0, vmax=float(np.percentile(image, 99.5)) or 1.0)
    if box is not None:
        (a0, a1), (b0, b1) = box
        ax.add_patch(Rectangle((a0, b0), a1 - a0, b1 - b0,
                               fill=False, edgecolor="#ea580c", lw=1.4))
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> int:
    parser = base_parser(__doc__ or "")
    parser.add_argument("--scans", nargs="*", default=None,
                        help="scan IDs to draw (default: a spread of six)")
    parser.add_argument("--frames", nargs=2, type=int, default=(30, 41), metavar=("FIRST", "LAST"),
                        help="inclusive native frame range to average; late frames show anatomy")
    parser.add_argument("--out", default=None, help="output PNG (default: <report>/crop_window_qc.png)")
    args = parser.parse_args()
    config = start(args, "qc_crop")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plan_path = config.work / "preprocessing_plan.json"
    if not plan_path.exists():
        raise SystemExit("Run scripts/01_prepare_data.py first")
    plan = load_plan(plan_path)

    manifest = load_manifest(config.work / "manifest.json")
    usable = [e.scan_id for e in manifest if e.usable and e.scan_id in plan.windows]
    if args.scans:
        scans = [s for s in args.scans if s in plan.windows]
        missing = sorted(set(args.scans) - set(scans))
        if missing:
            LOGGER.warning("no window on record for %s", ", ".join(missing))
    else:
        # A spread across the cohort rather than the first six alphabetically,
        # which would all come from the same acquisition group.
        step = max(1, len(usable) // 6)
        scans = usable[::step][:6]
    if not scans:
        raise SystemExit("nothing to draw")

    first, last = args.frames
    fig, axes = plt.subplots(len(scans), 4, figsize=(11.5, 2.6 * len(scans)), squeeze=False)

    for row, scan_id in enumerate(scans):
        scan = read_nifti_4d(config.dir_native / f"{scan_id}.nii.gz", scan_id)
        series = scan.data                                   # (T, Z, Y, X)
        lo, hi = max(0, first), min(series.shape[0] - 1, last)
        volume = series[lo:hi + 1].mean(axis=0)              # (Z, Y, X)

        transform = plan.transform(scan_id, config.reference_shape)
        (z0, z1), (y0, y1), (x0, x1) = transform.crop
        zc, yc, xc = (z0 + z1) // 2, (y0 + y1) // 2, (x0 + x1) // 2

        # Central slice through the middle of the window in each plane, so the
        # box is drawn where the data it selects actually lives.
        _panel(axes[row][0], volume[:, yc, :], f"{scan_id}  coronal (y={yc})",
               box=((x0, x1), (z0, z1)))
        _panel(axes[row][1], volume[:, :, xc], f"sagittal (x={xc})",
               box=((y0, y1), (z0, z1)))
        _panel(axes[row][2], volume[zc, :, :], f"axial (z={zc})",
               box=((x0, x1), (y0, y1)))

        cropped = transform.apply(volume)
        _panel(axes[row][3], cropped[:, cropped.shape[1] // 2, :],
               f"cropped {'x'.join(str(v) for v in cropped.shape)}")

        margin_z = min(z0, volume.shape[0] - z1)
        margin_y = min(y0, volume.shape[1] - y1)
        margin_x = min(x0, volume.shape[2] - x1)
        LOGGER.info(
            "%s: window z[%d:%d] y[%d:%d] x[%d:%d] of %s, edge margin (z,y,x) = (%d, %d, %d) voxels",
            scan_id, z0, z1, y0, y1, x0, x1, volume.shape, margin_z, margin_y, margin_x,
        )
        if min(margin_z, margin_y, margin_x) <= 0:
            LOGGER.warning("%s: the window touches the edge of the field of view", scan_id)

    fig.suptitle(
        f"Stage 01 crop window on the native image  -  frames {lo}-{hi} averaged",
        fontsize=10,
    )
    fig.tight_layout()

    out = Path(args.out) if args.out else (config.dir_report / "crop_window_qc.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    LOGGER.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
