"""Load one generated Appendix F dataset without changing files or traffic arrays."""

import argparse
from pathlib import Path
import pickle
import re
import resource
import sys

import numpy as np


JOBS = {
    "train_6tab": ("train", 6, 20000),
    "test_2tab": ("test", 2, 5000),
    "test_4tab": ("test", 4, 5000),
    "test_6tab": ("test", 6, 5000),
}


def load_appendix_f(job, timestamp=False):
    """Return data/label lists, plus unmodified time arrays if timestamp=True.

    Labels follow the official loader's concatenate(..., ones(...) * 50).tolist()
    convention. Traffic arrays retain their original lengths, values and dtypes.
    """
    if job not in JOBS:
        raise ValueError(f"Unknown dataset {job!r}; choose from {tuple(JOBS)}")
    source, tabs, expected_count = JOBS[job]
    directory = Path(__file__).resolve().parents[1] / "dataset" / "generated_appendix_f" / job
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

    label_files = list(directory.glob("BAPM_*.npz"))
    if len(label_files) != 1:
        raise ValueError(f"Expected one BAPM label archive, found {len(label_files)}")
    with np.load(label_files[0], allow_pickle=False) as archive:
        labels = archive["label"]
    if labels.shape != (expected_count, tabs):
        raise ValueError(f"Raw label shape {labels.shape}; expected {(expected_count, tabs)}")
    if not np.issubdtype(labels.dtype, np.integer) or not np.all((labels >= 0) & (labels <= 50)):
        raise ValueError("Raw labels must be integers in 0-50")

    result = {"data": [], "label": []}
    if timestamp:
        result["time"] = []
    for _, path in chunks:
        with path.open("rb") as handle:
            chunk = pickle.load(handle)
        if len(chunk["data"]) != len(chunk["time"]):
            raise ValueError(f"{path.name}: data/time sample counts differ")
        if any(len(data) != len(time) for data, time in zip(chunk["data"], chunk["time"])):
            raise ValueError(f"{path.name}: data/time trace lengths differ")
        if len(chunk["data"]) == 0:
            del chunk
            continue
        result["data"].extend(chunk["data"])
        if timestamp:
            result["time"].extend(chunk["time"])
        del chunk
    count = len(result["data"])
    if count != expected_count or count != len(labels):
        raise ValueError(f"Combined samples={count}, label rows={len(labels)}, expected={expected_count}")
    result["label"] = np.concatenate(
        [labels, np.ones((count, 6 - tabs)) * 50], axis=-1
    ).tolist()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", required=True, choices=JOBS, metavar="JOB")
    parser.add_argument("--timestamp", action="store_true", help="Include original timestamp arrays.")
    args = parser.parse_args()
    dataset = load_appendix_f(args.check, timestamp=args.timestamp)
    print(f"Dataset: {args.check}")
    print(f"Sample count: {len(dataset['data'])}")
    print(f"Padded label shape: {np.shape(dataset['label'])}")
    print(f"First raw trace length: {len(dataset['data'][0])}")
    print(f"Last raw trace length: {len(dataset['data'][-1])}")
    print(f"Timestamps included: {'time' in dataset}")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin" or sys.platform.startswith("linux"):
        peak_bytes = peak if sys.platform == "darwin" else peak * 1024
        print(f"Approximate process peak RSS: {peak_bytes / 1_000_000:.2f} MB "
              "(not current RSS; MB = 1,000,000 bytes)")
    else:
        print("Process memory report unavailable: platform RSS units not defined here.")


if __name__ == "__main__":
    main()
