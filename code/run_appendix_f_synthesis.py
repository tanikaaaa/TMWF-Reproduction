"""Run one explicitly selected Appendix F synthesis job using the original code."""

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
    "train_6tab": ("train", 6, 20000),
    "test_2tab": ("test", 2, 5000),
    "test_4tab": ("test", 4, 5000),
    "test_6tab": ("test", 6, 5000),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, choices=JOBS,
                        help="Select exactly one synthesis job.")
    args = parser.parse_args()
    source_name, page_num, expected_count = JOBS[args.job]
    root = Path(__file__).resolve().parents[1]
    source = root / "dataset" / "tbb single tab" / source_name
    output = root / "dataset" / "generated_appendix_f" / args.job
    if not source.is_file():
        raise FileNotFoundError(f"Original source file not found: {source}")
    if output.is_symlink():
        raise FileExistsError(f"Refusing a symlink output directory: {output}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite nonempty output path: {output}")
    output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="tmwf_appendix_f_input_") as temporary:
        input_dir = Path(temporary)
        (input_dir / source_name).symlink_to(source)
        settings = {
            "input_dir": str(input_dir) + os.sep,
            "output_dir": str(output) + os.sep,
            "page_num": page_num,
            "expand_rate": 5,
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
        print(f"Expected successful trace count: {expected_count}", flush=True)
        print("Python random seed: 2023; NumPy seed: 2023", flush=True)
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
    print(f"Job completed. Original generator outputs retained unchanged in {output}",
          flush=True)


if __name__ == "__main__":
    main()
