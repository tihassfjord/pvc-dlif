"""Where does a training epoch actually go?  Measure it on this machine.

Runs a few epochs of one real (condition, fold, run) with a stopwatch around
each phase, then A/B tests a handful of settings that are safe to change.
Writes nothing into the work tree except its own report: no checkpoints, no
summaries, so it cannot disturb a run in progress.

    python scripts/profile_training.py                     # breakdown, 3 epochs
    python scripts/profile_training.py --epochs 5 --variants

The phases:

    load        pulling the batch out of the DataLoader (host-side copies)
    transfer    host -> device
    augment     Poisson noise, flips, average channel (on the device)
    forward     the model and the loss
    backward    gradients and the optimiser step
    validate    the whole validation pass

On a GPU every phase is timed with ``torch.cuda.synchronize()`` around it, so
the numbers are real rather than the usual "everything looks instant because
CUDA is asynchronous".
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

from _common import base_parser, start                       # noqa: E402

from pvc_dlif.data.manifest import load_manifest             # noqa: E402
from pvc_dlif.dlif.adapter import DlifRepo, build_model      # noqa: E402
from pvc_dlif.dlif.dataset import DlifDataset, make_folds, stratified_val_split  # noqa: E402
from pvc_dlif.dlif.train import (                            # noqa: E402
    TrainSettings, _build_loss, _build_optimizer, augment_on_device,
)
from pvc_dlif.logging_utils import get_logger                # noqa: E402

LOGGER = get_logger("stage.profile")


def _sync(device: str) -> float:
    """A timestamp that is true on a GPU as well as a CPU."""
    if device.startswith("cuda"):
        import torch
        torch.cuda.synchronize()
    return time.perf_counter()


def timed_epoch(model, loader, criterion, device, optimizer, scaler, amp,
                device_aug, generator) -> dict[str, float]:
    """One training epoch, with the stopwatch on every phase."""
    import torch

    timers: dict[str, float] = defaultdict(float)
    model.train(True)

    mark = _sync(device)
    for batch in loader:
        t = _sync(device); timers["load"] += t - mark; mark = t

        inputs = batch["INPUT"].to(device, dtype=torch.float32, non_blocking=True)
        targets = batch["AIF"].to(device, dtype=torch.float32, non_blocking=True)
        t = _sync(device); timers["transfer"] += t - mark; mark = t   # ~0 when prefetched

        if device_aug is not None:
            inputs = augment_on_device(
                inputs, generator, poisson_noise=device_aug["poisson_noise"],
                random_flip=device_aug["random_flip"], add_average=device_aug["add_average"])
            t = _sync(device); timers["augment"] += t - mark; mark = t

        if amp and device.startswith("cuda"):
            with torch.autocast("cuda", dtype=torch.float16):
                output = model(inputs)
                prediction = output["REG"] if isinstance(output, dict) else output
                loss = criterion(prediction.float(), targets)
        else:
            output = model(inputs)
            prediction = output["REG"] if isinstance(output, dict) else output
            loss = criterion(prediction, targets)
        t = _sync(device); timers["forward"] += t - mark; mark = t

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        t = _sync(device); timers["backward"] += t - mark; mark = t

    return dict(timers)


def validate_once(model, loader, criterion, device, amp, device_aug, generator) -> float:
    import torch

    model.train(False)
    mark = _sync(device)
    with torch.no_grad():
        for batch in loader:
            inputs = batch["INPUT"].to(device, dtype=torch.float32, non_blocking=True)
            targets = batch["AIF"].to(device, dtype=torch.float32, non_blocking=True)
            if device_aug is not None:
                inputs = augment_on_device(inputs, generator, poisson_noise=False,
                                           random_flip=False, add_average=device_aug["add_average"])
            output = model(inputs)
            prediction = output["REG"] if isinstance(output, dict) else output
            criterion(prediction, targets)
    return _sync(device) - mark


class StreamPrefetcher:
    """Overlap the host->device copy of the next batch with this batch's compute.

    NVIDIA's standard trick: issue the copy on a second CUDA stream while the
    default stream is still computing.  Two details make or break it -

    * no ``torch.cuda.empty_cache()`` in the loop.  It synchronises the whole
      device, which serialises exactly what the second stream was meant to
      overlap;
    * ``record_stream`` on each tensor, so the allocator does not reuse memory
      that the copy stream is still writing into.

    Only worth it when the transfer is a real share of the epoch, which is what
    this profiler is here to find out.
    """

    def __init__(self, loader, device: str):
        import torch
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream()

    def __len__(self) -> int:
        return len(self.loader)

    def _to_device(self, batch):
        import torch
        with torch.cuda.stream(self.stream):
            return {k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v)
                    for k, v in batch.items()}

    def __iter__(self):
        import torch

        iterator = iter(self.loader)
        try:
            nxt = self._to_device(next(iterator))
        except StopIteration:
            return
        while nxt is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            current = nxt
            for value in current.values():
                if torch.is_tensor(value):
                    value.record_stream(torch.cuda.current_stream())
            try:
                nxt = self._to_device(next(iterator))
            except StopIteration:
                nxt = None
            yield current


def build_everything(config, settings, condition_name, fold_index, seed_offset=0):
    """The same datasets and model stage 05 would build for one run."""
    import torch
    from torch.utils.data import DataLoader

    manifest = load_manifest(config.work / "manifest.json")
    repo = DlifRepo(config.dlif_repo)
    condition = config.condition(condition_name)
    data_root = config.dir_dlif_inputs / condition.input_tag
    base_ids = [e.scan_id for e in manifest if e.usable]
    folds = make_folds(base_ids, int(config.get("dlif.cv.n_folds", 10)), config.seed)
    fold = next(f for f in folds if f.index == fold_index)

    run_seed = config.seed + 1000 * fold.index + 1 + seed_offset
    group_of = {e.scan_id: e.group for e in manifest}
    train_ids, val_ids = stratified_val_split(
        fold.train_ids, group_of, float(config.get("dlif.cv.validation_size", 0.15)), seed=run_seed)

    device = settings.resolve_device()
    on_device = device.startswith("cuda")
    augmentation = dict(config.get("dlif.augmentation", {}))
    cache: dict = {}
    common = dict(aif_root=config.dlif_data_root, img_shape=config.retrain_shape,
                  add_average=config.retrain_add_average, seed=run_seed, cache=cache,
                  augment_on_device=on_device)
    train_dataset = DlifDataset(data_root, train_ids, mode="train", augment=True,
                                poisson_noise=bool(augmentation.get("poisson_noise", True)),
                                random_flip=bool(augmentation.get("random_flip", True)), **common)
    val_dataset = DlifDataset(data_root, val_ids, mode="val", augment=False, **common)

    LOGGER.info("loading %d scans into memory…", len(train_ids) + len(val_ids))
    started = time.perf_counter()
    for i in range(len(train_dataset)):
        train_dataset[i]
    for i in range(len(val_dataset)):
        val_dataset[i]
    LOGGER.info("  %.1f s", time.perf_counter() - started)

    train_loader = DataLoader(train_dataset.as_torch(), batch_size=settings.batch_size, shuffle=True,
                              num_workers=settings.num_workers, pin_memory=on_device)
    val_loader = DataLoader(val_dataset.as_torch(), batch_size=settings.batch_size, shuffle=False,
                            num_workers=settings.num_workers, pin_memory=on_device)

    model = build_model(repo, config.retrain_model_name, config.retrain_in_channels).to(device)
    device_aug = None
    generator = None
    if on_device:
        device_aug = {"poisson_noise": train_dataset.poisson_noise,
                      "random_flip": train_dataset.random_flip,
                      "add_average": train_dataset.add_average}
        generator = torch.Generator(device=device)
        generator.manual_seed(run_seed)

    return dict(model=model, train_loader=train_loader, val_loader=val_loader, device=device,
                device_aug=device_aug, generator=generator, n_train=len(train_ids), n_val=len(val_ids))


def measure(parts, settings, epochs: int, label: str) -> dict:
    """Run ``epochs`` epochs and return the average per-phase seconds."""
    import torch

    device = parts["device"]
    criterion = _build_loss(settings)
    optimizer = _build_optimizer(parts["model"], settings)
    scaler = torch.amp.GradScaler("cuda") if (settings.amp and device.startswith("cuda")) else None

    totals: dict[str, float] = defaultdict(float)
    wall = 0.0
    for epoch in range(epochs):
        started = _sync(device)
        timers = timed_epoch(parts["model"], parts["train_loader"], criterion, device,
                             optimizer, scaler, settings.amp, parts["device_aug"], parts["generator"])
        timers["validate"] = validate_once(parts["model"], parts["val_loader"], criterion, device,
                                           settings.amp, parts["device_aug"], parts["generator"])
        elapsed = _sync(device) - started
        # The first epoch pays for CUDA context, cudnn autotuning and warm-up.
        if epoch == 0 and epochs > 1:
            LOGGER.info("  %s: warm-up epoch %.1f s (discarded)", label, elapsed)
            continue
        for key, value in timers.items():
            totals[key] += value
        wall += elapsed

    n = max(1, epochs - 1 if epochs > 1 else 1)
    result = {key: value / n for key, value in totals.items()}
    result["total"] = wall / n
    return result


def report(name: str, result: dict) -> None:
    order = ["load", "transfer", "augment", "forward", "backward", "validate"]
    total = result.get("total", 0.0) or 1.0
    print(f"\n{name}   {total:.2f} s/epoch")
    for key in order:
        if key in result:
            print(f"   {key:10s} {result[key]:6.2f} s   {100 * result[key] / total:4.1f} %")
    accounted = sum(result.get(k, 0.0) for k in order)
    print(f"   {'other':10s} {total - accounted:6.2f} s   {100 * (total - accounted) / total:4.1f} %")
    if "augment" not in result:
        print("   (on CPU the augmentation runs in the dataset, so it is inside 'load')")


def main() -> int:
    parser = base_parser(__doc__ or "")
    parser.add_argument("--epochs", type=int, default=3,
                        help="epochs to time (the first is discarded as warm-up)")
    parser.add_argument("--condition", default=None, help="default: the reference condition")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--variants", action="store_true",
                        help="also A/B a few safe settings and print a comparison")
    args = parser.parse_args()
    config = start(args, "profile_training")

    settings = TrainSettings.from_config(config)
    if args.device:
        settings.device = args.device
    condition_name = args.condition or config.reference_condition.name

    import torch
    device = settings.resolve_device()
    print(f"\ndevice: {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.startswith("cuda") else "")
          + f"\nmodel:  {config.retrain_model_name}  input {tuple(config.retrain_shape)}"
          + f"  batch {settings.batch_size}  amp {settings.amp}"
          + f"\ncondition: {condition_name}, fold {args.fold}")

    parts = build_everything(config, settings, condition_name, args.fold)
    print(f"scans:  {parts['n_train']} train, {parts['n_val']} val")

    baseline = measure(parts, settings, args.epochs, "baseline")
    report("baseline (as configured)", baseline)
    results = {"baseline": baseline}

    if args.variants:
        # Each variant changes exactly one thing.  None of them changes the
        # protocol: same model, same data, same loss, same optimiser.
        variants = []

        if device.startswith("cuda"):
            variants.append(("cudnn.benchmark", dict(cudnn_benchmark=True)))
            variants.append(("no AMP", dict(amp=False)))
            variants.append(("stream prefetch", dict(prefetch=True)))
        variants.append(("batch 16", dict(batch_size=16)))
        variants.append(("batch 4", dict(batch_size=4)))

        for label, change in variants:
            print(f"\n--- {label} …", flush=True)
            trial = TrainSettings(**{**settings.__dict__})
            rebuild = False
            for key, value in change.items():
                if key == "cudnn_benchmark":
                    torch.backends.cudnn.benchmark = bool(value)
                elif key == "batch_size":
                    trial.batch_size = int(value); rebuild = True
                else:
                    setattr(trial, key, value)
            trial_parts = build_everything(config, trial, condition_name, args.fold) if rebuild else parts
            if change.get("prefetch"):
                # Wrap the loaders; the timed loop then sees batches already on
                # the device, so "transfer" collapses into "load".
                trial_parts = {**trial_parts,
                               "train_loader": StreamPrefetcher(trial_parts["train_loader"], device),
                               "val_loader": StreamPrefetcher(trial_parts["val_loader"], device)}
            try:
                results[label] = measure(trial_parts, trial, args.epochs, label)
                report(label, results[label])
            except Exception as exc:                          # noqa: BLE001
                print(f"   failed: {exc}")
                results[label] = {"failed": str(exc)}
            torch.backends.cudnn.benchmark = False
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        print("\n" + "=" * 60)
        print(f"{'variant':22s} {'s/epoch':>9s}  {'vs baseline':>12s}")
        base_total = baseline["total"]
        for label, result in results.items():
            if "failed" in result:
                print(f"{label:22s} {'failed':>9s}")
                continue
            change = 100 * (result["total"] - base_total) / base_total
            print(f"{label:22s} {result['total']:9.2f}  {change:+11.1f} %")

    out = config.dir_results / "training_profile.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "condition": condition_name, "fold": args.fold,
        "batch_size": settings.batch_size, "amp": settings.amp,
        "n_train": parts["n_train"], "n_val": parts["n_val"],
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
