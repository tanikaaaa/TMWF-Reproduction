"""Read-only integrity audit of the four existing TBB real-world pickles."""

from pathlib import Path
import pickle

import numpy as np


def length_summary(traces):
    lengths = [len(trace) for trace in traces]
    if not lengths:
        return "N/A (no traces)"
    return f"{min(lengths)} / {np.median(lengths):g} / {max(lengths)}"


def audit_file(path, expected_tabs):
    with path.open("rb") as handle:
        dataset = pickle.load(handle)
    directions = dataset["data"]
    times = dataset["time"]
    labels = dataset["label"]

    bad_rows = noninteger_labels = out_of_range_labels = 0
    monitored = unmonitored = 0
    for row in labels:
        values = np.asarray(row)
        bad_rows += values.ndim != 1 or values.size != expected_tabs
        for label in values.flat:
            # Validate stored integer types without silently coercing floats.
            is_integer = isinstance(label, (int, np.integer)) and not isinstance(
                label, (bool, np.bool_)
            )
            if not is_integer:
                noninteger_labels += 1
            elif not 0 <= label <= 50:
                out_of_range_labels += 1
            elif label == 50:
                unmonitored += 1
            else:
                monitored += 1

    invalid_directions = invalid_direction_traces = 0
    for trace in directions:
        count = int(np.count_nonzero(~np.isin(np.asarray(trace), (-1, 0, 1))))
        invalid_directions += count
        invalid_direction_traces += count > 0

    nonfinite_timestamps = nonfinite_traces = 0
    order_violations = order_violation_traces = 0
    negative_differences = backward_traces = 0
    maximum_backward_jump = np.longdouble(0)
    for trace in times:
        timestamps = np.asarray(trace)
        count = int(np.count_nonzero(~np.isfinite(timestamps)))
        nonfinite_timestamps += count
        nonfinite_traces += count > 0
        # Equal timestamps pass. NaN comparisons fail this order check and
        # are also captured by the separate non-finite timestamp check.
        count = int(np.count_nonzero(~(timestamps[1:] >= timestamps[:-1])))
        order_violations += count
        order_violation_traces += count > 0
        # Subtract a floating copy to avoid integer wraparound; never sort or
        # modify the loaded arrays. NaNs are not negative differences.
        with np.errstate(invalid="ignore", over="ignore"):
            differences = np.diff(timestamps.astype(np.longdouble))
        backwards = differences[differences < 0]
        negative_differences += backwards.size
        backward_traces += backwards.size > 0
        if backwards.size:
            maximum_backward_jump = max(maximum_backward_jump, -backwards.min())

    mismatches = sum(len(direction) != len(time)
                     for direction, time in zip(directions, times))
    unpaired = abs(len(directions) - len(times))
    try:
        label_shape = str(np.shape(labels))
    except ValueError:
        label_shape = f"ragged ({len(labels)} rows)"

    print(f"\nFile: {path.name}")
    print(f"Sample count (direction traces): {len(directions)}")
    print(f"Timestamp trace count: {len(times)}; label rows: {len(labels)}")
    print(f"Sample/time/label counts agree: {len(directions) == len(times) == len(labels)}")
    print(f"Label shape: {label_shape}")
    print(f"Expected tab labels per row: {expected_tabs}")
    print(f"Label rows with wrong shape/tab count: {bad_rows}")
    print(f"Non-integer labels (including bools): {noninteger_labels}")
    print(f"Integer labels outside 0-50: {out_of_range_labels}")
    valid_labels = (len(labels) == len(directions) and
                    bad_rows == noninteger_labels == out_of_range_labels == 0)
    print(f"Label integrity: {'PASS' if valid_labels else 'FAIL'}")
    print(f"Monitored tab labels (0-49): {monitored}")
    print(f"Unmonitored tab labels (50): {unmonitored}")
    print(f"Direction trace lengths (min / median / max): {length_summary(directions)}")
    print(f"Timestamp trace lengths (min / median / max): {length_summary(times)}")
    print(f"Empty direction traces: {sum(len(trace) == 0 for trace in directions)}")
    print(f"Empty timestamp traces: {sum(len(trace) == 0 for trace in times)}")
    print(f"Direction/time length mismatches (paired traces): {mismatches}")
    print(f"Unpaired direction/time traces: {unpaired}")
    print(f"Invalid direction values: {invalid_directions} in {invalid_direction_traces} traces")
    print(f"Non-finite timestamps: {nonfinite_timestamps} in {nonfinite_traces} traces")
    print(f"Timestamp-order violations (failed adjacent >= comparisons): "
          f"{order_violations} in {order_violation_traces} traces")
    print(f"Negative adjacent timestamp differences: "
          f"{negative_differences} in {backward_traces} traces")
    print(f"Maximum backward jump (original timestamp units; 0 if none): "
          f"{maximum_backward_jump}")


def main():
    dataset_dir = Path(__file__).resolve().parents[1] / "dataset" / "tbb realworld"
    print("TBB real-world multi-tab integrity audit (aggregate results only)")
    print("Label counts count individual tab labels, not samples.")
    print("Order check allows equal timestamps; NaN comparisons fail;")
    print("empty/singleton traces have no adjacent order violations.")
    for name, tabs in (("2tabs", 2), ("2tabs-extra", 2), ("3tabs", 3), ("4tabs", 4)):
        audit_file(dataset_dir / name, tabs)


if __name__ == "__main__":
    main()
