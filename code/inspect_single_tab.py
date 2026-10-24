"""Read-only audit with bounded timestamp diagnostics for single-tab pickles."""

from collections import Counter
from pathlib import Path
import pickle

import numpy as np


def length_summary(traces):
    lengths = [len(trace) for trace in traces]
    if not lengths:
        return "N/A (no traces)"
    return f"{min(lengths)} / {np.median(lengths):g} / {max(lengths)}"


def audit_partition(path):
    # These are the existing local dataset pickles; loading does not rewrite them.
    with path.open("rb") as handle:
        partition = pickle.load(handle)
    directions = partition["data"]
    times = partition["time"]
    labels = partition["label"]

    invalid_values = invalid_traces = 0
    for trace in directions:
        count = int(np.count_nonzero(~np.isin(np.asarray(trace), (-1, 0, 1))))
        invalid_values += count
        invalid_traces += count > 0

    nonfinite_values = nonfinite_traces = nonmonotonic_traces = 0
    negative_differences = []
    affected_groups = Counter()
    examples = []
    for sample_index, trace in enumerate(times):
        timestamps = np.asarray(trace)
        count = int(np.count_nonzero(~np.isfinite(timestamps)))
        nonfinite_values += count
        nonfinite_traces += count > 0
        # Compare directly to avoid overflow from subtracting timestamps.
        # NaNs fail adjacent comparisons; non-finite values are also counted above.
        nonmonotonic_traces += not bool(
            np.all(timestamps[1:] >= timestamps[:-1])
        )
        # Use a floating copy to avoid integer subtraction wraparound. Never
        # sort or modify the original timestamps; position is the next index.
        with np.errstate(invalid="ignore", over="ignore"):
            differences = np.diff(timestamps.astype(np.longdouble))
        positions = np.flatnonzero(differences < 0)
        if positions.size:
            negative_differences.extend(differences[positions])
            label = labels[sample_index] if sample_index < len(labels) else None
            if label is None:
                group = "missing label"
            elif label == -1:
                group = "unmonitored"
            else:
                group = "monitored"
            affected_groups[group] += 1
            for position in positions[:max(0, 5 - len(examples))]:
                examples.append((sample_index, label, int(position) + 1,
                                 timestamps[position], timestamps[position + 1]))

    paired_mismatches = sum(
        len(direction) != len(time)
        for direction, time in zip(directions, times)
    )
    unpaired = abs(len(directions) - len(times))
    print(f"\nPartition: {path.name}")
    print(f"Sample count (direction traces): {len(directions)}")
    print(f"Timestamp trace count: {len(times)}; label count: {len(labels)}")
    print(f"Sample/label counts agree: {len(directions) == len(times) == len(labels)}")
    print("Label distribution (label: count):")
    for label, count in sorted(Counter(labels.tolist()).items()):
        print(f"  {label}: {count}")
    print(f"Direction trace lengths (min / median / max): {length_summary(directions)}")
    print(f"Timestamp trace lengths (min / median / max): {length_summary(times)}")
    print(f"Empty direction traces: {sum(len(trace) == 0 for trace in directions)}")
    print(f"Empty timestamp traces: {sum(len(trace) == 0 for trace in times)}")
    print(f"Direction/time length mismatches (paired traces): {paired_mismatches}")
    print(f"Unpaired direction/time traces: {unpaired}")
    print(f"Invalid direction values: {invalid_values} in {invalid_traces} traces")
    print(f"Non-finite timestamps: {nonfinite_values} in {nonfinite_traces} traces")
    print(f"Timestamp traces failing nondecreasing order: {nonmonotonic_traces}")
    print(f"Negative adjacent timestamp differences: {len(negative_differences)}")
    print(f"Traces with negative differences: {sum(affected_groups.values())}")
    if negative_differences:
        print(f"Minimum negative difference: {min(negative_differences)}")
        print("Median absolute negative difference: "
              f"{np.median(np.abs(negative_differences))}")
    else:
        print("Minimum negative difference: N/A (no negative differences)")
        print("Median absolute negative difference: N/A (no negative differences)")
    print("Affected traces by label group:")
    print(f"  monitored (label != -1): {affected_groups['monitored']}")
    print(f"  unmonitored (label == -1): {affected_groups['unmonitored']}")
    if affected_groups['missing label']:
        print(f"  missing label: {affected_groups['missing label']}")
    print("Violation examples (up to 5; zero-based sample index and position;")
    print("position is the next timestamp index, following position - 1):")
    if not examples:
        print("  None")
    for sample_index, label, position, previous, following in examples:
        print(f"  sample index={sample_index}, label={label}, "
              f"violation position={position}, previous timestamp={previous}, "
              f"next timestamp={following}")


def main():
    dataset_dir = Path(__file__).resolve().parents[1] / "dataset" / "tbb single tab"
    print("Single-tab dataset audit (aggregates and up to 5 violation examples per partition)")
    print("Order check: every adjacent timestamp must be >= its predecessor;")
    print("empty/singleton traces pass; adjacent comparisons with NaN fail.")
    for name in ("train", "test"):
        audit_partition(dataset_dir / name)


if __name__ == "__main__":
    main()
