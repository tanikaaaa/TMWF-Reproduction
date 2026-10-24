"""Inspect official synthesis output using automatically removed temporary data."""

import os
from pathlib import Path
import pickle
import random
import sys
import tempfile

sys.dont_write_bytecode = True

import numpy as np

import MergeSingleTraces_openworld as synthesis
from MergeSingleTraces_openworld import merge_single_traces, save_merged_traces


def inspect_outputs(output_dir):
    print("Temporary output files:")
    for path in sorted(output_dir.iterdir()):
        print(f"  output/{path.name}")

    chunks = sorted(output_dir.glob("COCO_*"))
    label_files = sorted(output_dir.glob("BAPM_*.npz"))
    total_data = total_times = total_labels = 0
    empty_chunks = []
    valid = bool(chunks and label_files)
    for path in chunks:
        with path.open("rb") as handle:
            chunk = pickle.load(handle)
        data_count = len(chunk["data"])
        time_count = len(chunk["time"])
        total_data += data_count
        total_times += time_count
        valid = valid and data_count == time_count
        if data_count == 0:
            empty_chunks.append(path.name)
        print(f"{path.name}: data samples={data_count}, timestamp samples={time_count}")

    for path in label_files:
        with np.load(path, allow_pickle=False) as archive:
            labels = archive["label"]
            rows = labels.shape[0] if labels.ndim else 0
            total_labels += rows
            shape_ok = labels.ndim == 2 and labels.shape[1] == 6
            values_ok = (np.issubdtype(labels.dtype, np.integer) and
                         bool(np.all((labels >= 0) & (labels <= 50))))
            valid = valid and shape_ok and values_ok
            print(f"{path.name}: label rows={rows}, label shape={labels.shape}, "
                  f"six labels per row={shape_ok}, integer labels in 0-50={values_ok}")

    counts_agree = total_data == total_times == total_labels == 20
    print(f"Combined data samples={total_data}, timestamp samples={total_times}, "
          f"label rows={total_labels}; expected=20")
    print(f"Combined counts agree with expected total: {counts_agree}")
    print(f"Empty output chunks: {empty_chunks or 'none'}")
    print("Label-file suffixes do not establish per-chunk alignment; "
          "the totals above combine all chunks and label files.")
    return valid and counts_agree


def main():
    source_path = Path(__file__).resolve().parents[1] / "dataset" / "tbb single tab" / "train"
    with source_path.open("rb") as handle:
        source = pickle.load(handle)
    labels = np.asarray(source["label"])
    monitored = np.flatnonzero((labels >= 0) & (labels < 50))[:20]
    unmonitored = np.flatnonzero(labels == -1)[:20]
    if len(monitored) != 20 or len(unmonitored) != 20:
        raise ValueError("Source must contain at least 20 monitored and 20 unmonitored traces")
    indices = np.concatenate((monitored, unmonitored))
    tiny_source = {
        "time": [source["time"][i] for i in indices],
        "data": [source["data"][i] for i in indices],
        "label": labels[indices],
    }

    with tempfile.TemporaryDirectory(prefix="tmwf_synthesis_test_") as temporary:
        root = Path(temporary)
        input_dir = root / "input"
        output_dir = root / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        with (input_dir / "train").open("wb") as handle:
            pickle.dump(tiny_source, handle)
        print("Temporary input: input/train (20 monitored, then 20 unmonitored traces)")
        settings = {
            "input_dir": str(input_dir) + os.sep,
            "output_dir": str(output_dir) + os.sep,
            "page_num": 6,
            "expand_rate": 1,
            "devide_num": 10,
            "merge_type": "duration",
            "overlapped_ratios": [0.1, 0.2, 0.3, 0.4, 0.5],
            "pure_ratio_threshold": 0.1,
            "complete_randomization": True,
            "save_format": "CONJUNCTION",
            "task_type": "multi",
            "padding": False,
        }
        original_settings = {key: getattr(synthesis, key) for key in settings}
        try:
            for key, value in settings.items():
                setattr(synthesis, key, value)
            random.seed(2023)
            np.random.seed(2023)
            # The original entry point calls the original save function internally.
            if synthesis.save_merged_traces is not save_merged_traces:
                raise RuntimeError("The imported original save function has been replaced")
            merge_single_traces()
            valid = inspect_outputs(output_dir)
        finally:
            for key, value in original_settings.items():
                setattr(synthesis, key, value)
    print("Temporary input/output files removed; synthesis module settings restored.")
    print("PASS: combined counts and labels verified." if valid else
          "FAIL: output counts or labels failed verification.")
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
