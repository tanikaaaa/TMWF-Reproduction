"""Run one explicitly selected Appendix H TBB duration-overlap synthesis job.

Seeds, multipliers, and chunk size are explicit reproduction choices.
Delay synthesis is intentionally not supported pending separate protocol review.
"""

import argparse
import os
from pathlib import Path
import random
import sys
import tempfile
import time

sys.dont_write_bytecode = True

import numpy as np

import MergeSingleTraces_openworld as synthesis
from MergeSingleTraces_openworld import merge_single_traces


JOBS = {
    "train_6tab": ("train", 6, 10, 40000),
    "ratio_test_2tab": ("test", 2, 1, 1000),
    "ratio_test_3tab": ("test", 3, 1, 1000),
    "ratio_test_4tab": ("test", 4, 1, 1000),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, choices=JOBS,
                        help="Select exactly one duration-overlap synthesis job.")
    args = parser.parse_args()
    source_name, page_num, expand_rate, expected_count = JOBS[args.job]
    root = Path(__file__).resolve().parents[1]
    source = root / "dataset" / "tbb single tab" / source_name
    output = root / "dataset" / "generated_appendix_h" / args.job
    if not source.is_file():
        raise FileNotFoundError(f"Original source file not found: {source}")
    # Reject redirected output paths, including symlinked parent directories.
    for directory in (root / "dataset", output.parent, output):
        if directory.is_symlink():
            raise FileExistsError(f"Refusing a symlink output path: {directory}")
        if directory.exists() and not directory.is_dir():
            raise FileExistsError(f"Refusing an existing file at output path: {directory}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output path: {output}")
    output.mkdir(parents=True, exist_ok=True)

    # Keep even temporary input staging inside this job's output directory.
    with tempfile.TemporaryDirectory(prefix=".input_", dir=output) as temporary:
        input_dir = Path(temporary)
        (input_dir / source_name).symlink_to(source)
        settings = {
            "input_dir": str(input_dir) + os.sep,
            "output_dir": str(output) + os.sep,
            "page_num": page_num,
            "expand_rate": expand_rate,
            "devide_num": 1000,
            "merge_type": "duration",
            "overlapped_ratios": [0.1, 0.2, 0.3, 0.4, 0.5],
            "pure_ratio_threshold": 0.1,
            "complete_randomization": True,
            "task_type": "multi",
            "save_format": "CONJUNCTION",
            "padding": False,
        }
        original_settings = {key: getattr(synthesis, key) for key in settings}
        print(f"Job: {args.job}; source: {source}", flush=True)
        print(f"Output directory: {output}", flush=True)
        print(f"Expected successful trace count: {expected_count}", flush=True)
        print("Explicit reproduction choices: settings below; "
              "Python random seed: 2023; NumPy seed: 2023", flush=True)
        for key, value in settings.items():
            print(f"{key}={value!r}", flush=True)
        try:
            for key, value in settings.items():
                setattr(synthesis, key, value)
            random.seed(2023)
            np.random.seed(2023)
            start = time.perf_counter()
            try:
                merge_single_traces()
            finally:
                print(f"Elapsed synthesis time: {time.perf_counter() - start:.2f} s",
                      flush=True)
        finally:
            for key, value in original_settings.items():
                setattr(synthesis, key, value)
    print(f"Job completed. Original generator filenames and empty final chunks "
          f"retained unchanged in {output}", flush=True)


if __name__ == "__main__":
    main()
