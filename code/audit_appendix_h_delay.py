"""Read-only audit of copy-isolated Appendix H delay outputs and their manifest."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import resource
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
JOBS = {"delay_test_2tab": 2, "delay_test_3tab": 3, "delay_test_4tab": 4}
PROTOCOL = "copy-isolated intended-protocol reproduction; NOT literal released-code reproduction"
FILES = {"COCO_test_0", "COCO_test_1", "BAPM_test_1.npz",
         "test_0_annotations.json", "test_1_annotations.json"}
SOURCES = ("dataset/tbb single tab/test", "code/run_appendix_h_delay_synthesis.py",
           "code/MergeSingleTraces_openworld.py", "code/test_appendix_h_delay.py",
           "code/run_appendix_h_synthesis.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    with path.open(encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=unique)


def verify_manifest(manifest, job, directory, root):
    require(isinstance(manifest, dict), "Manifest must be an object")
    expected = dict(job=job, protocol=PROTOCOL, source_partition="test",
                    expected_count=1000, status="completed",
                    seeds={"python_random": 2023, "numpy": 2023})
    for key, value in expected.items():
        require(manifest.get(key) == value, f"Manifest mismatch: {key}")
    require(manifest.get("deviation") == "Independent copies of every selected timestamp/direction occurrence before original delay merger",
            "Missing/mismatched copy-isolation deviation")
    settings = manifest.get("configuration", {})
    expected_settings = dict(valid_category_num=50, trace_length=5120, page_num=JOBS[job],
                             expand_rate=1, devide_num=1000, merge_type="delay", min_delay=2,
                             max_delay=9, overlapped_ratios=[0.1, 0.2, 0.3, 0.4, 0.5],
                             pure_ratio_threshold=0.1, complete_randomization=True,
                             task_type="multi", save_format="CONJUNCTION", padding=False)
    require(set(settings) == set(expected_settings) | {"input_dir", "output_dir"},
            "Unexpected/missing manifest configuration fields")
    for key, value in expected_settings.items():
        require(type(settings[key]) is type(value) and settings[key] == value,
                f"Manifest configuration mismatch: {key}")
    require(Path(settings["output_dir"]).resolve() == directory.resolve(), "Manifest output directory mismatch")
    staging = Path(settings["input_dir"])
    require(staging.parent.resolve() == directory.resolve() and staging.name.startswith(".input_"),
            "Manifest temporary input path mismatch")
    require(not staging.exists() and not staging.is_symlink(), "Temporary input staging was not cleaned up")
    software = manifest.get("software", {})
    require(all(isinstance(software.get(k), str) and software[k] for k in ("python", "numpy", "tqdm", "platform")),
            "Missing software version metadata")
    require(isinstance(manifest.get("elapsed_seconds"), (int, float))
            and np.isfinite(manifest["elapsed_seconds"]) and manifest["elapsed_seconds"] >= 0,
            "Invalid elapsed time metadata")
    before, after = manifest.get("fingerprints"), manifest.get("fingerprints_after")
    require(isinstance(before, dict) and set(before) == set(SOURCES), "Missing/unexpected source/code fingerprint entries")
    require(before == after, "Source/code fingerprints changed during generation")
    actual = {name: sha256(root / name) for name in SOURCES}
    require(actual == before, "Current source/code hashes differ from generation fingerprints")
    print("Manifest protocol/configuration/seeds verified; recorded source/code hashes match current files.")
    print(f"Recorded software: {software}")
    return actual


def inspect_chunk(path, expected):
    with path.open("rb") as handle:
        chunk = pickle.load(handle)
    require(isinstance(chunk, dict) and set(chunk) == {"data", "time"}, f"{path.name}: invalid chunk schema")
    data, times = chunk["data"], chunk["time"]
    require(len(data) == len(times) == expected, f"{path.name}: expected {expected} data/time samples")
    affected = reversals = 0
    largest = 0.0
    for index, (direction, time) in enumerate(zip(data, times)):
        d, t = np.asarray(direction), np.asarray(time)
        context = f"{path.name} sample {index}"
        require(d.ndim == t.ndim == 1 and d.size > 0 and t.size > 0, f"{context}: empty or non-1D trace")
        require(d.size == t.size, f"{context}: data/time length mismatch")
        require(np.all(np.isin(d, (-1, 1))), f"{context}: invalid raw direction (only -1/+1 allowed)")
        require(np.issubdtype(t.dtype, np.number) and not np.iscomplexobj(t), f"{context}: non-real timestamps")
        require(np.all(np.isfinite(t)), f"{context}: non-finite timestamps")
        delta = np.diff(t.astype(np.float64, copy=False))
        negative = delta[delta < 0]
        if negative.size:
            affected += 1
            reversals += int(negative.size)
            largest = max(largest, float(-negative.min()))
    print(f"{path.name}: directions={len(data)}, timestamps={len(times)}; "
          f"reversal traces={affected}, negative differences={reversals}, maximum backward jump={largest:g}")
    if expected == 0:
        print(f"Original empty final chunk verified: {path.name}")
    return {"data": len(data), "time": len(times)}, (affected, reversals, largest)


def audit(job, root=None):
    """Optional root enables temporary fixture tests; CLI always uses project ROOT."""
    print("Annotation-label alignment unavailable for delay synthesis.")
    print("Structural checks do not prove original-source reconstruction, sample independence, "
          "traffic-label provenance, or exact paper-level delay-distribution fidelity.")
    try:
        require(job in JOBS, f"Unsupported job: {job}")
        root = ROOT if root is None else Path(root)
        directory = root / "dataset/generated_appendix_h_delay/copy_isolated" / job
        if not directory.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {directory}")
        print(f"Job: {job}; directory: {directory}")
        require({p.name for p in directory.iterdir()} == FILES | {"manifest.json"},
                "Missing/unexpected files; required COCO numbering is exactly [0, 1]")
        require(all((directory / name).is_file() for name in FILES | {"manifest.json"}), "Non-file dataset entry")
        baseline = {name: sha256(directory / name) for name in FILES | {"manifest.json"}}
        manifest = read_json(directory / "manifest.json")
        source_baseline = verify_manifest(manifest, job, directory, root)
        require(manifest.get("output_sha256") == {name: baseline[name] for name in FILES},
                "Output file hashes differ from manifest")
        print("COCO numbering: [0, 1], contiguous from zero; generated-file hashes verified.")
        counts, reversal_stats = {}, []
        for suffix, expected in ((0, 1000), (1, 0)):
            name = f"COCO_test_{suffix}"
            counts[name], stats = inspect_chunk(directory / name, expected)
            reversal_stats.append(stats)
            annotation = f"test_{suffix}_annotations.json"
            require(read_json(directory / annotation) == {"traces": [], "annotations": [], "categories": []},
                    f"{annotation}: expected original empty delay annotation lists")
            print(f"{annotation}: empty traces/annotations/categories verified")
        with np.load(directory / "BAPM_test_1.npz", allow_pickle=False) as archive:
            require(set(archive.files) == {"label"}, "Unexpected BAPM archive fields")
            labels = archive["label"]
        tabs = JOBS[job]
        require(labels.shape == (1000, tabs), f"Invalid raw label shape: {labels.shape}")
        require(np.issubdtype(labels.dtype, np.integer) and np.all((labels >= 0) & (labels <= 50)),
                "Raw labels must be integers in 0-50")
        original = labels.copy()
        padded = np.concatenate((labels, np.full((1000, 6-tabs), 50, dtype=labels.dtype)), axis=1)
        require(padded.shape == (1000, 6) and np.array_equal(padded[:, :tabs], original)
                and np.all(padded[:, tabs:] == 50) and np.array_equal(labels, original), "In-memory padding failed")
        print(f"BAPM_test_1.npz: raw label validity PASS, shape={labels.shape}; "
              f"in-memory padding validity PASS, shape={padded.shape}")
        counts.update(label_shape=list(labels.shape), label_rows=1000,
                      total_data=sum(counts[k]["data"] for k in ("COCO_test_0", "COCO_test_1")),
                      total_time=sum(counts[k]["time"] for k in ("COCO_test_0", "COCO_test_1")))
        require(manifest.get("generated_counts") == counts, "Manifest generated counts differ from actual files")
        require({name: sha256(root / name) for name in SOURCES} == source_baseline, "Source/code changed during audit")
        require({p.name: sha256(p) for p in directory.iterdir()} == baseline, "Dataset files changed during audit")
        print(f"Timestamp reversals (reported, not repaired): traces={sum(x[0] for x in reversal_stats)}, "
              f"differences={sum(x[1] for x in reversal_stats)}, maximum backward jump={max(x[2] for x in reversal_stats):g}")
        print(f"PASS {job}: 1000 direction traces, 1000 timestamp traces, 1000 label rows; "
              "structural and manifest checks only; source/code and dataset hashes unchanged.")
        return 0
    except Exception as exc:
        print(f"FAIL {job}: {type(exc).__name__}: {exc}")
        return 1
    finally:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin" or sys.platform.startswith("linux"):
            size = peak if sys.platform == "darwin" else peak * 1024
            print(f"Approximate process peak RSS: {size / 1e6:.2f} MB (not current RSS)")
        else:
            print("Peak RSS unavailable: unknown platform units")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, choices=JOBS)
    return audit(parser.parse_args().job)


if __name__ == "__main__":
    raise SystemExit(main())
