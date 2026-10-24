"""Measure official duration-based synthesis in memory without saving traces."""

from pathlib import Path
import pickle
import random
import resource
import sys
import time

# Import the existing implementation without creating bytecode files.
sys.dont_write_bytecode = True

import numpy as np

from MergeSingleTraces_openworld import rm_zero, merge_with_durationscale


def peak_rss_mb():
    # macOS reports ru_maxrss in bytes; Linux reports it in KiB.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return peak / 1_000_000
    if sys.platform.startswith("linux"):
        return peak * 1024 / 1_000_000
    raise RuntimeError("Peak RSS unit conversion is supported only on macOS and Linux")


def main():
    path = Path(__file__).resolve().parents[1] / "dataset" / "tbb single tab" / "train"
    with path.open("rb") as handle:
        source = pickle.load(handle)
    raw_times = source["time"]
    raw_datas = source["data"]
    total_num = len(raw_times)
    if total_num != 8000 or len(raw_datas) != 8000 or len(source["label"]) != 8000:
        raise ValueError("Expected 8000 time traces, direction traces, and labels in train")

    random.seed(2023)
    np.random.seed(2023)
    page_num = 6
    overlapped_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
    target_successes = 200
    max_attempts = 20000
    attempts = 0
    lengths = []
    retained_directions = []
    retained_timestamps = []
    print("Duration synthesis probe: 6 tabs, complete_randomization=True", flush=True)
    print("Sampling uniformly from all 8000 traces with replacement.", flush=True)

    memory_before = peak_rss_mb()
    start = time.perf_counter()
    while len(lengths) < target_successes and attempts < max_attempts:
        attempts += 1
        # Match the original complete_randomization=True sampling and trimming.
        picked_indexes = [np.random.randint(0, total_num) for _ in range(page_num)]
        times = [rm_zero(raw_times[j]) for j in picked_indexes]
        datas = [rm_zero(raw_datas[j]) for j in picked_indexes]
        ratio = random.choice(overlapped_ratios)
        merged = merge_with_durationscale(times, datas, ratio)
        if merged is not None:
            lengths.append(len(merged[1]))
            retained_timestamps.append(merged[0])
            retained_directions.append(merged[1])
        # Keep successful sequences in Python lists until this probe finishes.
        del merged
    elapsed = time.perf_counter() - start
    memory_after = peak_rss_mb()

    successes = len(lengths)
    print(f"Total attempts: {attempts}")
    print(f"Successful merges: {successes}")
    print(f"Acceptance rate: {successes / attempts:.2%}")
    print(f"Elapsed synthesis time: {elapsed:.6f} s (includes retries; excludes loading)")
    print("Memory measurement: process peak RSS (ru_maxrss), not current RSS; "
          "MB = 1,000,000 bytes.")
    print(f"Peak RSS before generation (source data loaded): {memory_before:.2f} MB")
    print(f"Peak RSS after retaining {successes} traces: {memory_after:.2f} MB")
    additional_mb = memory_after - memory_before
    print(f"Approximate additional peak RSS observed for {successes} retained traces: "
          f"{additional_mb:.2f} MB")
    if successes:
        for trace_count in (1000, 10000):
            projected_mb = additional_mb * trace_count / successes
            print(f"Rough extrapolation for retaining {trace_count} traces: "
                  f"{projected_mb:.2f} MB additional memory above the baseline")
    else:
        print("Memory extrapolations for 1000 and 10000 traces: N/A (no successes)")
    print("Peak-RSS differences include temporary allocations and allocator effects; "
          "an earlier process peak can hide growth. These are not isolated retained-data "
          "sizes. Memory scaling may not be linear; extrapolations are rough estimates.")
    if successes:
        mean_seconds = elapsed / successes
        estimate = mean_seconds * 20000
        print(f"Mean time per successful merge (including retries): {mean_seconds:.6f} s")
        print("Resulting direction trace lengths (min / median / max): "
              f"{min(lengths)} / {np.median(lengths):g} / {max(lengths)}")
        print("Approximate time for 20000 successful training merges: "
              f"{estimate:.2f} s ({estimate / 60:.2f} min)")
        print("Linear throughput estimate from this small probe; includes in-memory "
              "retention, but excludes loading, serialization, and disk writes.")
    else:
        print("Mean time per successful merge: N/A")
        print("Resulting trace lengths (min / median / max): N/A")
        print("Approximate time for 20000 successful training merges: N/A (no successes)")
    if successes != target_successes:
        print(f"FAIL: reached the {max_attempts}-attempt limit before 200 successful merges.")
        return 1
    print("PASS: 200 successful in-memory merges; no generated traces saved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
