"""Read-only structural and cross-file audit of Appendix H duration datasets."""

import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import re

import numpy as np


JOBS = {"train_6tab": 40000, "ratio_test_2tab": 1000,
        "ratio_test_3tab": 1000, "ratio_test_4tab": 1000}
RAW_LABEL_WIDTHS = {"train_6tab": 6, "ratio_test_2tab": 2,
                    "ratio_test_3tab": 3, "ratio_test_4tab": 4}
CHUNK_SIZE = 1000


def inspect_chunk(path, annotation_path, final, errors):
    with path.open("rb") as handle:
        chunk = pickle.load(handle)
    directions, times = chunk["data"], chunk["time"]
    count = len(directions)
    print(f"{path.name}: data samples={count}, timestamp samples={len(times)}")
    if count != len(times):
        errors.append(f"{path.name}: unequal data/time sample counts")
    if count == 0:
        print(f"  Empty {'final' if final else 'non-final'} chunk: {path.name}")
        if not final:
            errors.append(f"{path.name}: empty non-final chunk")
    elif final:
        print("  Empty final chunk: no")

    stats = Counter()
    for trace in directions:
        array = np.asarray(trace)
        stats["non-1D direction traces"] += array.ndim != 1
        stats["empty direction traces"] += array.size == 0
        stats["invalid direction values"] += int(np.count_nonzero(~np.isin(array, (-1, 1))))
    for trace in times:
        array = np.asarray(trace)
        stats["non-1D timestamp traces"] += array.ndim != 1
        stats["empty timestamp traces"] += array.size == 0
        stats["non-finite timestamps"] += int(np.count_nonzero(~np.isfinite(array)))
    for direction, time in zip(directions, times):
        stats["data/time length mismatches"] += np.size(direction) != np.size(time)
    for name in ("non-1D direction traces", "non-1D timestamp traces",
                 "empty direction traces", "empty timestamp traces",
                 "invalid direction values", "non-finite timestamps",
                 "data/time length mismatches"):
        print(f"  {name}: {stats[name]}")
        if stats[name]:
            errors.append(f"{path.name}: {stats[name]} {name}")

    if not annotation_path.is_file():
        errors.append(f"Missing annotation file: {annotation_path.name}")
        return count, len(times)
    with annotation_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    traces, annotations = metadata["traces"], metadata["annotations"]
    print(f"  {annotation_path.name}: trace records={len(traces)}, "
          f"annotation records={len(annotations)}")
    ids = [trace.get("id") for trace in traces]
    if (len(traces) != count or any(type(value) is not int for value in ids)
            or set(ids) != set(range(count))):
        errors.append(f"{annotation_path.name}: trace count/IDs do not match chunk")
    else:
        if any(trace.get("width") != np.size(directions[trace["id"]]) for trace in traces):
            errors.append(f"{annotation_path.name}: trace widths do not match data lengths")
    annotation_ids = [annotation.get("id") for annotation in annotations]
    if (any(type(value) is not int for value in annotation_ids)
            or len(set(annotation_ids)) != len(annotation_ids)):
        errors.append(f"{annotation_path.name}: invalid or duplicate annotation IDs")
    for annotation in annotations:
        trace_id = annotation.get("trace_id")
        category = annotation.get("category_id")
        if type(trace_id) is not int or not 0 <= trace_id < count:
            errors.append(f"{annotation_path.name}: invalid annotation trace reference")
            break
        if type(category) is not int or not 0 <= category < 50:
            errors.append(f"{annotation_path.name}: invalid monitored category ID")
            break
    return count, len(times)


def check_alignment(chunk_counts, labels, errors):
    print("Alignment checks cross-file annotation/label consistency, "
          "not reconstruction of the original traffic.")
    global_sample_offset = 0
    total_checked = total_mismatches = 0
    alignment_valid = True
    for path, count in chunk_counts:
        checked = mismatches = 0
        annotation_path = path.with_name(
            path.name.removeprefix("COCO_") + "_annotations.json"
        )
        try:
            with annotation_path.open(encoding="utf-8") as handle:
                metadata = json.load(handle)
            categories = [Counter() for _ in range(count)]
            invalid_annotations = 0
            for annotation in metadata["annotations"]:
                trace_id = annotation.get("trace_id")
                category = annotation.get("category_id")
                if (type(trace_id) is not int or not 0 <= trace_id < count
                        or type(category) is not int or not 0 <= category < 50):
                    invalid_annotations += 1
                    continue
                categories[trace_id][category] += 1
            if invalid_annotations:
                alignment_valid = False
                errors.append(f"{annotation_path.name}: {invalid_annotations} invalid "
                              "annotation trace references/category IDs in alignment check")
            for local_trace_id in range(count):
                row = labels[global_sample_offset + local_trace_id]
                monitored = Counter(int(label) for label in row if 0 <= label < 50)
                checked += 1
                # Empty Counters agree for all-no-tab rows without annotations.
                mismatches += monitored != categories[local_trace_id]
            del metadata, categories
        except Exception as exc:
            alignment_valid = False
            errors.append(f"{path.name}: alignment {type(exc).__name__}: {exc}")
        print(f"{path.name}: alignment offset={global_sample_offset}, "
              f"traces checked={checked}/{count}, mismatches={mismatches}")
        total_checked += checked
        total_mismatches += mismatches
        global_sample_offset += count
    print(f"Overall alignment: traces checked={total_checked}, mismatches={total_mismatches}")
    if (alignment_valid and chunk_counts and total_mismatches == 0
            and total_checked == global_sample_offset == len(labels)):
        print("Alignment PASS: zero monitored-label multiset mismatches; "
              "cross-file consistency only, not traffic reconstruction.")
    else:
        errors.append(f"Alignment failed or incomplete: {total_mismatches} mismatches, "
                      f"{total_checked}/{len(labels)} label rows checked")
        print("Alignment FAIL")


def audit(job):
    directory = Path(__file__).resolve().parents[1] / "dataset" / "generated_appendix_h" / job
    errors = []
    print(f"Job: {job}\nDirectory: {directory}")
    print("Structural checks do not independently prove row-wise traffic/label correspondence.")
    print("Unverified: original-traffic reconstruction, source partition provenance, "
          "RNG seed, overlap ratios, timestamp ordering, and annotation geometry. "
          "These checks cannot establish the generation history. Delay datasets "
          "are unsupported; only duration jobs are accepted.")
    expected_width = RAW_LABEL_WIDTHS[job]
    print(f"Required raw label width: {expected_width}; model-input width: 6.")
    print("Model-input padding is checked on an in-memory copy only; raw labels are unchanged.")
    if not directory.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {directory}")
    source = "train" if job == "train_6tab" else "test"
    pattern = re.compile(rf"COCO_{source}_(\d+)")
    chunks = []
    for path in directory.glob("COCO_*"):
        match = pattern.fullmatch(path.name)
        if match and path.is_file():
            chunks.append((int(match.group(1)), path))
        else:
            errors.append(f"Unrecognized COCO chunk: {path.name}")
    chunks.sort(key=lambda item: (item[0], item[1].name))
    numbers = [number for number, _ in chunks]
    contiguous = bool(chunks) and numbers == list(range(len(chunks)))
    print(f"Chunk suffixes in numeric order: {numbers}")
    print(f"Contiguous numbering starting at zero: {contiguous}")
    expected_chunks = JOBS[job] // CHUNK_SIZE + 1
    print(f"Expected chunks including empty final chunk: {expected_chunks}")
    if len(chunks) != expected_chunks:
        errors.append("Chunk count does not include the expected empty final chunk")
    if not contiguous:
        errors.append("Chunk numbering is absent, duplicated, or noncontiguous")
    expected_annotations = {path.name.removeprefix("COCO_") + "_annotations.json"
                            for _, path in chunks}
    actual_annotations = {path.name for path in directory.glob("*_annotations.json")}
    for name in sorted(actual_annotations - expected_annotations):
        errors.append(f"Annotation file without corresponding COCO chunk: {name}")

    total_data = total_times = 0
    chunk_counts = []
    for position, (_, path) in enumerate(chunks):
        annotation = directory / (path.name.removeprefix("COCO_") + "_annotations.json")
        try:
            data_count, time_count = inspect_chunk(
                path, annotation, position == len(chunks) - 1, errors
            )
            expected_chunk_count = 0 if position == expected_chunks - 1 else CHUNK_SIZE
            if data_count != expected_chunk_count or time_count != expected_chunk_count:
                errors.append(f"{path.name}: expected {expected_chunk_count} direction/time "
                              "samples under devide_num=1000, including an empty final chunk")
            total_data += data_count
            total_times += time_count
            chunk_counts.append((path, data_count))
        except Exception as exc:
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
    print(f"Combined COCO samples: {total_data}; timestamp samples: {total_times}")
    print(f"Expected total: {JOBS[job]}")
    if total_data != JOBS[job] or total_times != JOBS[job]:
        errors.append("Combined trace counts do not match expected total")

    label_files = sorted(directory.glob("BAPM_*.npz"))
    print(f"Original BAPM label file candidates: {[path.name for path in label_files]}")
    if len(label_files) != 1:
        errors.append("Expected exactly one original BAPM label file")
    else:
        path = label_files[0]
        print(f"Exact BAPM label filename: {path.name}")
        if chunks and path.name != f"BAPM_{source}_{numbers[-1]}.npz":
            errors.append("BAPM filename does not match the original final-chunk suffix")
        with np.load(path, allow_pickle=False) as archive:
            labels = archive["label"]
            rows = labels.shape[0] if labels.ndim else 0
            width_ok = labels.ndim == 2 and labels.shape[1] == expected_width
            integer_ok = np.issubdtype(labels.dtype, np.integer)
            range_ok = bool(np.all((labels >= 0) & (labels <= 50)))
            print(f"Label shape: {labels.shape}; dtype: {labels.dtype}; rows: {rows}")
            print(f"Label rows match COCO total: {rows == total_data}; "
                  f"match expected total: {rows == JOBS[job]}")
            if rows != JOBS[job]:
                errors.append("Label row count does not match expected job total")
            print(f"{expected_width} raw labels per row: {width_ok}; integer labels: {integer_ok}; "
                  f"labels in 0-50: {range_ok}")
            if rows != total_data:
                errors.append("Label row count does not match combined COCO total")
            raw_valid = width_ok and integer_ok and range_ok and rows == total_data
            print(f"Raw label validity: {raw_valid}")
            if not width_ok or not integer_ok or not range_ok:
                errors.append(f"Labels must be a two-dimensional integer array with {expected_width} columns and values 0-50")
            if raw_valid:
                padded = np.concatenate(
                    (labels, np.full((rows, 6 - expected_width), 50, dtype=labels.dtype)),
                    axis=1,
                )
                padding_valid = (
                    padded.shape == (rows, 6)
                    and np.issubdtype(padded.dtype, np.integer)
                    and bool(np.all((padded >= 0) & (padded <= 50)))
                    and np.array_equal(padded[:, :expected_width], labels)
                    and bool(np.all(padded[:, expected_width:] == 50))
                )
                print(f"Model-input padding validity: {padding_valid}; shape={padded.shape}; "
                      f"appended class-50 columns={6 - expected_width}; raw columns preserved")
                if not padding_valid:
                    errors.append("In-memory model-input label padding failed verification")
                del padded
            else:
                print("Model-input padding validity: NOT CHECKED (invalid raw labels)")
            print("Label distribution (individual tab labels: count):")
            print(dict(sorted(Counter(labels.ravel().tolist()).items())))
            if labels.ndim == 2:
                monitored = np.any((labels >= 0) & (labels < 50), axis=1)
                print(f"Samples containing at least one monitored label (0-49): "
                      f"{int(np.count_nonzero(monitored))}")
            else:
                print("Samples containing monitored labels: unavailable (invalid label shape)")
            if raw_valid and contiguous and len(chunk_counts) == len(chunks):
                # Retain only chunk counts; reread one annotation file at a time.
                check_alignment(chunk_counts, labels, errors)
            else:
                print("Alignment NOT CHECKED: invalid raw labels or incomplete chunk ordering/counts")
                errors.append("Alignment verification is incomplete")
    if errors:
        print("FAIL:")
        for error in errors:
            print(f"  {error}")
        return 1
    print(f"PASS {job}: direction traces={total_data}, timestamp traces={total_times}, "
          f"label rows={rows}; alignment checked={total_data}, mismatches=0; "
          "raw labels and in-memory six-slot padding valid. "
          "Cross-file consistency does not prove original-traffic reconstruction.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, choices=JOBS)
    args = parser.parse_args()
    try:
        return audit(args.job)
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
