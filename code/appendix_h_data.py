"""Read-only Appendix H loaders; raw traffic is never padded, sorted or truncated.

load_appendix_h() loads one synthetic job; load_real_tbb() selects the first
1,000 samples of one original real file. Both return data/label lists compatible
with the official DataGenerator. timestamp=True retains original time arrays.
generate_batches() delegates direction-only preprocessing to that generator.
"""

import argparse
from pathlib import Path
import pickle
import re
import resource
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
JOBS = {
    "train_6tab": ("train", 6, 40000),
    "ratio_test_2tab": ("test", 2, 1000),
    "ratio_test_3tab": ("test", 3, 1000),
    "ratio_test_4tab": ("test", 4, 1000),
}
REAL_JOBS = {"real_2tab": 2, "real_3tab": 3, "real_4tab": 4}


def _pad_labels(labels, count, tabs):
    labels = np.asarray(labels)
    if labels.shape != (count, tabs):
        raise ValueError(f"Raw label shape {labels.shape}; expected {(count, tabs)}")
    if not np.issubdtype(labels.dtype, np.integer) or not np.all((labels >= 0) & (labels <= 50)):
        raise ValueError("Raw labels must be integers in 0-50")
    # Same concatenate(..., ones(...) * 50).tolist() convention as Appendix F.
    return np.concatenate([labels, np.ones((count, 6 - tabs)) * 50], axis=-1).tolist()


def _validate_traces(data, times, context, synthetic=False):
    if len(data) != len(times):
        raise ValueError(f"{context}: data/time sample counts differ")
    allowed = (-1, 1) if synthetic else (-1, 0, 1)
    for index, (direction, time) in enumerate(zip(data, times)):
        d, t = np.asarray(direction), np.asarray(time)
        if d.ndim != 1 or t.ndim != 1 or not d.size or d.size != t.size:
            raise ValueError(f"{context}, sample {index}: empty, non-1D, or mismatched data/time")
        if not np.all(np.isin(d, allowed)):
            raise ValueError(f"{context}, sample {index}: invalid packet direction")
        if not np.all(np.isfinite(t)):
            raise ValueError(f"{context}, sample {index}: non-finite timestamp")


def load_appendix_h(job, timestamp=False):
    """Load one synthetic job in numeric chunk order, preserving raw arrays."""
    if job not in JOBS:
        raise ValueError(f"Unknown synthetic dataset {job!r}; choose from {tuple(JOBS)}")
    source, tabs, expected = JOBS[job]
    directory = ROOT / "dataset" / "generated_appendix_h" / job
    if not directory.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {directory}")
    pattern = re.compile(rf"COCO_{source}_(\d+)")
    chunks = []
    for path in directory.glob("COCO_*"):
        match = pattern.fullmatch(path.name)
        if not match or not path.is_file():
            raise ValueError(f"Unexpected COCO chunk: {path.name}")
        chunks.append((int(match.group(1)), path))
    chunks.sort(key=lambda item: item[0])
    if not chunks or [number for number, _ in chunks] != list(range(len(chunks))):
        raise ValueError("COCO chunk numbering must be contiguous from zero, without duplicates")
    if len(chunks) != expected // 1000 + 1:
        raise ValueError("Expected 1,000-sample chunks followed by the original empty final chunk")
    label_files = list(directory.glob("BAPM_*.npz"))
    label_path = directory / f"BAPM_{source}_{chunks[-1][0]}.npz"
    if len(label_files) != 1 or label_files[0] != label_path:
        raise ValueError(f"Expected exactly the matching label archive: {label_path.name}")
    with np.load(label_path, allow_pickle=False) as archive:
        labels = _pad_labels(archive["label"], expected, tabs)
    result = {"data": [], "label": labels}
    if timestamp:
        result["time"] = []
    for position, (_, path) in enumerate(chunks):
        with path.open("rb") as handle:
            chunk = pickle.load(handle)
        data, times = chunk["data"], chunk["time"]
        required = 0 if position == len(chunks) - 1 else 1000
        if len(data) != required or len(times) != required:
            raise ValueError(f"{path.name}: expected {required} data/time samples")
        _validate_traces(data, times, path.name, synthetic=True)
        if required:
            result["data"].extend(data)
            if timestamp:
                result["time"].extend(times)
        del chunk, data, times
    if len(result["data"]) != expected:
        raise ValueError(f"Expected {expected} combined samples")
    return result


def load_real_tbb(tabs, timestamp=False):
    """Load exactly the first 1,000 original TBB 2/3/4-tab rows, in file order.

    No Chrome default, tab-list indexing, or 2tabs-extra selection is used.
    Timestamp order is preserved, including any pre-existing reversals.
    """
    if type(tabs) is not int or tabs not in (2, 3, 4):
        raise ValueError("Real TBB tabs must be 2, 3, or 4")
    path = ROOT / "dataset" / "tbb realworld" / f"{tabs}tabs"
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    available = len(raw["data"])
    if available < 1000 or len(raw["time"]) != available or len(raw["label"]) != available:
        raise ValueError(f"{path.name}: need at least 1,000 matching data/time/label rows")
    raw_labels = np.asarray(raw["label"])
    # Validate the original label matrix before selecting the published subset.
    _pad_labels(raw_labels, available, tabs)
    labels = _pad_labels(raw_labels[:1000], 1000, tabs)
    data, times = list(raw["data"][:1000]), list(raw["time"][:1000])
    _validate_traces(data, times, path.name)
    result = {"data": data, "label": labels}
    if timestamp:
        result["time"] = times
    return result


def generate_batches(dataset, indices=None, batch_size=80):
    """Return the original repeating direction-only batch iterator.

    Inputs are truncated/zero-padded to 6 * 5120 = 30720 only in batches.
    Default ordering is unchanged; explicit indices are forwarded unchanged.
    This does not instantiate, train, or evaluate a model.
    """
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    # Lazy import keeps ordinary loading/checks limited to stdlib and NumPy.
    sys.dont_write_bytecode = True
    from simulation_openworld import DataGenerator
    if indices is None:
        indices = np.arange(len(dataset["data"]))
    return DataGenerator(batch_size, 5120, 6, False).generate(dataset, indices)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", required=True, choices=tuple(JOBS) + tuple(REAL_JOBS))
    parser.add_argument("--timestamp", action="store_true", help="Retain original timestamp arrays.")
    args = parser.parse_args()
    try:
        if args.check in JOBS:
            dataset = load_appendix_h(args.check, timestamp=args.timestamp)
            _, tabs, expected = JOBS[args.check]
        else:
            tabs, expected = REAL_JOBS[args.check], 1000
            dataset = load_real_tbb(tabs, timestamp=args.timestamp)
        print(f"PASS {args.check}: samples={len(dataset['data'])}, expected={expected}")
        print(f"Raw label shape: {(expected, tabs)}; padded label shape: {np.shape(dataset['label'])}")
        print(f"First/last raw trace lengths: {len(dataset['data'][0])}/{len(dataset['data'][-1])}")
        print(f"Timestamps validated: {expected}; retained: {'time' in dataset}")
        print("No timestamp sorting or raw traffic padding/truncation. "
              "Annotation alignment and source provenance are not re-audited by this loader.")
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin" or sys.platform.startswith("linux"):
            peak_bytes = peak if sys.platform == "darwin" else peak * 1024
            print(f"Approximate process peak RSS: {peak_bytes / 1_000_000:.2f} MB "
                  "(not current RSS; MB = 1,000,000 bytes)")
        else:
            print("Peak RSS unavailable: platform units are not defined here.")
        return 0
    except Exception as exc:
        print(f"FAIL {args.check}: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
