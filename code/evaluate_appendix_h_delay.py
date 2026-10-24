"""Read-only Table 9 mixed-delay evaluation; --check-only never runs predictions."""

import argparse
import gc
import os
from pathlib import Path
import pickle
import sys
import time
import traceback

sys.dont_write_bytecode = True
import numpy as np
import torch
import torch.nn.functional as F

import audit_appendix_h_delay as delay_audit
from appendix_h_data import _pad_labels, _validate_traces, generate_batches
from evaluate_appendix_h import (
    ROOT, CHECKPOINT, MODEL, SETTINGS, METRICS, METRIC_NAMES,
    load_final_model, sha256, synchronize, report_memory, verify_model_unchanged, score,
)

JOBS = {"delay_test_2tab": 2, "delay_test_3tab": 3, "delay_test_4tab": 4}
PUBLISHED = (31.6, 63.8, 31.3)


def protocol_report():
    print("Copy-isolated reproduction of the released TMWF delay-merging algorithm.")
    print("NOT literal released-code execution. Configured TBB random-delay range: 2–9 seconds.")
    print("Original unusual delay-draw schedule preserved, including unused draws. "
          "Inputs independently copied per sampled occurrence to prevent cross-sample mutation.")
    print("Existing timestamp reversals were not corrected. Annotation-label alignment unavailable for delay synthesis.")
    print("Numerical agreement with Table 9 does not establish exact protocol identity.")


def directory_for(job):
    if job not in JOBS:
        raise ValueError(f"Unsupported delay dataset: {job}")
    return ROOT / "dataset/generated_appendix_h_delay/copy_isolated" / job


def dataset_hashes(job):
    directory = directory_for(job)
    return {p.name: sha256(p) for p in sorted(directory.iterdir()) if p.is_file()}


def load_delay(job):
    """Private adapter: audit first, retain raw direction arrays in original order."""
    if delay_audit.audit(job, root=ROOT) != 0:
        raise ValueError(f"{job}: delay manifest/data audit failed")
    directory = directory_for(job)
    with np.load(directory / "BAPM_test_1.npz", allow_pickle=False) as archive:
        labels = _pad_labels(archive["label"], 1000, JOBS[job])
    data = []
    for suffix, expected in ((0, 1000), (1, 0)):
        with (directory / f"COCO_test_{suffix}").open("rb") as handle:
            chunk = pickle.load(handle)
        if len(chunk["data"]) != expected or len(chunk["time"]) != expected:
            raise ValueError(f"{job}: unexpected chunk sample counts")
        _validate_traces(chunk["data"], chunk["time"], f"{job}/{suffix}", synthetic=True)
        data.extend(chunk["data"])
        del chunk
    if len(data) != 1000 or np.shape(labels) != (1000, 6):
        raise ValueError(f"{job}: expected 1000 samples and six label slots")
    print(f"{job}: samples=1000; raw labels=(1000, {JOBS[job]}); padded labels=(1000, 6)")
    return {"data": data, "label": labels}


def predict_component(job, tabs, dataset, model, device):
    synchronize(device)
    start = time.perf_counter()
    labels = np.asarray(dataset["label"])
    if len(dataset["data"]) != 1000 or labels.shape != (1000, 6):
        raise ValueError(f"{job}: expected exactly 1000 samples with six label slots")
    if (not np.isfinite(labels).all() or not np.all(labels == np.floor(labels))
            or not np.all((labels >= 0) & (labels <= 50)) or not np.all(labels[:, tabs:] == 50)):
        raise ValueError(f"{job}: invalid labels/padding")
    generator = generate_batches(dataset, np.arange(1000), batch_size=80)
    outputs, truths = [], []
    evaluated = 0
    model.eval()
    with torch.inference_mode():
        for _ in range((1000 + 79) // 80):
            xb, yb = next(generator)
            batch = min(80, 1000 - evaluated)
            if xb.shape != (batch, 30720) or yb.shape != (batch, 6):
                raise ValueError(f"{job}: invalid batch shapes {xb.shape}, {yb.shape}")
            if not np.array_equal(yb, labels[evaluated:evaluated + batch]):
                raise ValueError(f"{job}: label ordering changed")
            if not np.isin(xb, (-1, 0, 1)).all():
                raise ValueError(f"{job}: invalid direction input")
            logits = model(torch.tensor(xb, dtype=torch.float).to(device))
            if tuple(logits.shape) != (batch, 6, 51):
                raise ValueError(f"{job}: invalid output shape {tuple(logits.shape)}")
            if not torch.isfinite(logits).all().item():
                raise FloatingPointError(f"{job}: non-finite logits")
            outputs.append(logits.cpu().numpy())
            truths.append(F.one_hot(torch.tensor(yb, dtype=torch.long), 51).numpy())
            evaluated += batch
    predictions, targets = np.concatenate(outputs), np.concatenate(truths)
    if evaluated != 1000 or predictions.shape != (1000, 6, 51) or targets.shape != (1000, 6, 51):
        raise ValueError(f"{job}: incomplete inference")
    synchronize(device)
    elapsed = time.perf_counter() - start
    print(f"{job}: 1000 samples; inference time={elapsed:.2f} s", flush=True)
    report_memory(device)
    return predictions, targets, elapsed


def print_comparison(values):
    print("\nAppendix H / Table 9 mixed delay-synthetic: 3000 samples.")
    print("Original metric rounding preserved; delta = reproduced minus published.")
    print(f"{'Metric':<18} {'Reproduced %':>13} {'Published %':>12} {'Delta pp':>10}")
    for name, value, target in zip(METRIC_NAMES, values, PUBLISHED):
        print(f"{name:<18} {value*100:>13.1f} {target:>12.1f} {value*100-target:>+10.1f}")
    protocol_report()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check-only", action="store_true", help="Audit/load all inputs and checkpoint; no predictions.")
    modes.add_argument("--evaluate", action="store_true", help="Predict once for each of 3000 delay samples.")
    args = parser.parse_args()
    start = time.perf_counter()
    try:
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
            raise RuntimeError("Automatic MPS CPU fallback enabled externally; unset before evaluation")
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        print(f"Selected device: {device}; batch size=80; epoch 50 only", flush=True)
        protocol_report()
        print("Cross-backend numerical differences are possible.")
        checkpoint_before = sha256(CHECKPOINT)
        before = {job: dataset_hashes(job) for job in JOBS}
        source_before = {name: sha256(ROOT / name) for name in delay_audit.SOURCES}
        model, reference = load_final_model(device)
        predictions, truths = [], []
        loaded = 0
        for job, tabs in JOBS.items():
            component_start = time.perf_counter()
            dataset = load_delay(job)
            loaded += len(dataset["data"])
            if args.evaluate:
                p, y, _ = predict_component(job, tabs, dataset, model, device)
                predictions.append(p)
                truths.append(y)
            else:
                # Exercise original batching on a single batch, without any model call.
                xb, yb = next(generate_batches(dataset, np.arange(1000), batch_size=80))
                if xb.shape != (80, 30720) or yb.shape != (80, 6):
                    raise ValueError(f"{job}: incompatible original DataGenerator shapes")
                if not np.array_equal(yb, np.asarray(dataset["label"][:80])):
                    raise ValueError(f"{job}: label order changed during batching")
                print(f"{job}: input/label batch shapes verified; no inference")
                del xb, yb
            verify_model_unchanged(model, reference)
            print(f"{job}: elapsed including loading/audit/checks={time.perf_counter()-component_start:.2f} s", flush=True)
            del dataset
            gc.collect()
            report_memory(device)
        if loaded != 3000:
            raise ValueError(f"Expected exactly 3000 combined samples, loaded {loaded}")
        if args.evaluate:
            # Joint scoring over all samples, never an average of component metrics.
            values = score(np.concatenate(predictions), np.concatenate(truths), 3000)
        verify_model_unchanged(model, reference)
        if sha256(CHECKPOINT) != checkpoint_before:
            raise AssertionError("Checkpoint changed")
        if {job: dataset_hashes(job) for job in JOBS} != before:
            raise AssertionError("Delay dataset/manifest hashes changed")
        if {name: sha256(ROOT / name) for name in delay_audit.SOURCES} != source_before:
            raise AssertionError("Generation source/code hashes changed")
        if args.evaluate:
            print_comparison(values)
            print("PASS: 3000 predictions; checkpoint, model parameters/buffers, datasets and generation sources unchanged.")
        else:
            print("CHECK PASS: epoch-50 compatibility, manifests, provenance, data format, "
                  "3000 samples and batching verified. No predictions performed; hashes and model state unchanged.")
        return 0
    except Exception:
        traceback.print_exc()
        print("FAIL: stopped without training, regeneration, metric substitution or automatic device fallback.")
        return 1
    finally:
        print(f"Total elapsed time: {time.perf_counter()-start:.2f} s", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
