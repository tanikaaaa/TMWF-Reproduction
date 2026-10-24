"""One copy-isolated Appendix H delay job; not literal released-code reproduction.

Only per-occurrence input ownership changes. The original orchestration, sampling,
unused overlap choice, delay draws, packet order and output formats are retained.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import pickle
import platform
import random
import sys
import tempfile
import time
import traceback

sys.dont_write_bytecode = True
import numpy as np
import MergeSingleTraces_openworld as synthesis

ROOT = Path(__file__).resolve().parents[1]
JOBS = {"delay_test_2tab": 2, "delay_test_3tab": 3, "delay_test_4tab": 4}
PROTOCOL = "copy-isolated intended-protocol reproduction; NOT literal released-code reproduction"


def configuration(job, input_dir, output_dir):
    if job not in JOBS:
        raise ValueError(f"Unsupported delay job: {job}")
    return dict(input_dir=str(input_dir) + os.sep, output_dir=str(output_dir) + os.sep,
                valid_category_num=50, trace_length=5120, page_num=JOBS[job],
                expand_rate=1, devide_num=1000, merge_type="delay",
                min_delay=2, max_delay=9, overlapped_ratios=[0.1, 0.2, 0.3, 0.4, 0.5],
                pure_ratio_threshold=0.1, complete_randomization=True,
                task_type="multi", save_format="CONJUNCTION", padding=False)


def isolated_merger(original):
    def merge(times, datas, min_interval, max_interval):
        # Copy each occurrence separately: deepcopy of a whole list can preserve aliases.
        return original([np.array(t, copy=True) for t in times],
                        [np.array(d, copy=True) for d in datas], min_interval, max_interval)
    return merge


@contextmanager
def configured_generator(settings):
    originals = {key: getattr(synthesis, key) for key in settings}
    original_merge = synthesis.merge_with_delaytime
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        for key, value in settings.items():
            setattr(synthesis, key, value)
        synthesis.merge_with_delaytime = isolated_merger(original_merge)
        random.seed(2023)
        np.random.seed(2023)
        yield
    finally:
        synthesis.merge_with_delaytime = original_merge
        for key, value in originals.items():
            setattr(synthesis, key, value)
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def reserve_output(output):
    # Refuse even empty existing job directories, broken symlinks and redirected parents.
    for path in (output,) + tuple(output.parents):
        if path.is_symlink():
            raise FileExistsError(f"Symlink output path refused: {path}")
        if path.exists() and not path.is_dir():
            raise FileExistsError(f"File at output path refused: {path}")
    if output.exists():
        raise FileExistsError(f"Existing job directory refused: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_manifest(path, contents):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".manifest_", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(contents, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Atomic, no-clobber publication on the same filesystem.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def generated_counts(output, tabs):
    counts = {}
    for suffix, expected in ((0, 1000), (1, 0)):
        name = f"COCO_test_{suffix}"
        with (output / name).open("rb") as handle:
            chunk = pickle.load(handle)
        counts[name] = {"data": len(chunk["data"]), "time": len(chunk["time"])}
        if counts[name] != {"data": expected, "time": expected}:
            raise ValueError(f"Unexpected counts: {name}: {counts[name]}")
        del chunk
        with (output / f"test_{suffix}_annotations.json").open() as handle:
            if json.load(handle) != {"traces": [], "annotations": [], "categories": []}:
                raise ValueError("Unexpected delay annotation structure")
    with np.load(output / "BAPM_test_1.npz", allow_pickle=False) as archive:
        labels = archive["label"]
        if labels.shape != (1000, tabs):
            raise ValueError(f"Unexpected label shape: {labels.shape}")
        counts["label_shape"] = list(labels.shape)
    counts["total_data"] = counts["total_time"] = counts["label_rows"] = 1000
    return counts


def run(job):
    if job not in JOBS:
        raise ValueError(f"Unsupported delay job: {job}")
    source = ROOT / "dataset/tbb single tab/test"
    output = ROOT / "dataset/generated_appendix_h_delay/copy_isolated" / job
    if not source.is_file():
        raise FileNotFoundError(source)
    # Verify the monitored-first partition assumption underlying expand_rate=1.
    with source.open("rb") as handle:
        raw = pickle.load(handle)
    labels = np.asarray(raw["label"])
    if (len(raw["data"]) != 2000 or len(raw["time"]) != 2000 or labels.shape != (2000,)
            or not np.all((labels[:1000] >= 0) & (labels[:1000] < 50))
            or not np.all(np.isin(labels[1000:], (-1, 50)))):
        raise ValueError("Expected held-out 2000-source partition, monitored 1000 first")
    del raw, labels
    paths = [source, Path(__file__).resolve(), Path(synthesis.__file__).resolve(),
             ROOT / "code/test_appendix_h_delay.py", ROOT / "code/run_appendix_h_synthesis.py"]
    fingerprints = {str(p.relative_to(ROOT)): sha256(p) for p in paths}
    reserve_output(output)
    manifest = dict(protocol=PROTOCOL, job=job, source_partition="test", expected_count=1000,
                    deviation="Independent copies of every selected timestamp/direction occurrence before original delay merger",
                    seeds={"python_random": 2023, "numpy": 2023}, fingerprints=fingerprints,
                    software={"python": platform.python_version(), "numpy": np.__version__,
                              "tqdm": version("tqdm"), "platform": platform.platform()},
                    started_utc=datetime.now(timezone.utc).isoformat(),
                    cautions=["Source timestamp reversals and packet order preserved",
                              "Original unused overlap choice and trailing uniform draws retained",
                              "No delay purity rejection or fabricated annotations",
                              "Counts do not establish traffic-label provenance"])
    start = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix=".input_", dir=output) as temporary:
            input_dir = Path(temporary)
            (input_dir / "test").symlink_to(source)
            settings = configuration(job, input_dir, output)
            manifest["configuration"] = settings
            print(f"Protocol: {PROTOCOL}\nJob: {job}\nSource: {source}\nOutput: {output}", flush=True)
            print(f"Expected count: 1000; seeds: 2023\nConfiguration: {settings}", flush=True)
            with configured_generator(settings):
                synthesis.merge_single_traces()
        manifest["generated_counts"] = generated_counts(output, JOBS[job])
        manifest["fingerprints_after"] = {str(p.relative_to(ROOT)): sha256(p) for p in paths}
        if manifest["fingerprints_after"] != fingerprints:
            raise RuntimeError("Source/code fingerprints changed during generation")
        manifest["output_sha256"] = {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["partial_output_files"] = sorted(p.name for p in output.iterdir() if p.is_file())
        raise
    finally:
        manifest["elapsed_seconds"] = time.perf_counter() - start
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_manifest(output / "manifest.json", manifest)
        print(f"Status: {manifest['status']}; elapsed: {manifest['elapsed_seconds']:.2f} s", flush=True)
    print(f"Generated counts: {manifest['generated_counts']}; original outputs retained unchanged.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, choices=JOBS)
    args = parser.parse_args()
    try:
        run(args.job)
        return 0
    except Exception:
        traceback.print_exc()
        print("FAIL: no automatic retry, output deletion, or protocol changes.", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
