"""Read-only final-epoch Appendix H / Table 9 evaluation (complete PDF pp.20-21).

Use --check-only for checkpoint/configuration checks without any predictions.
Use --evaluate explicitly for the 6,000 component-sample evaluation.
"""

import argparse
import gc
import hashlib
import os
from pathlib import Path
import resource
import sys
import time
import traceback

sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from appendix_h_data import load_appendix_h, load_real_tbb, generate_batches
from models import TMWF_DFNet
from metrics import overall_basic_accuracy, overall_basic_precision, overall_basic_recall

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "outputs/appendix_h/full/run_8mrp3k_2/epoch_050.pth"
MODEL = dict(embed_dim=256, nhead=8, dim_feedforward=1024,
             num_encoder_layers=2, num_decoder_layers=2, max_len=121,
             num_queries=6, cls=51, dropout=0.1)
VALIDATION_JOBS = ("ratio_test_2tab", "ratio_test_3tab", "ratio_test_4tab")
REPRODUCTION_CHOICES = dict(
    seed=2023,
    validation="Concatenate 1000 samples each from ratio_test_2tab/3tab/4tab; "
               "compute original Basic metrics jointly, for monitoring only",
    checkpoint_selection="Final epoch 50; no best-model selection or early stopping",
)
SETTINGS = dict(experiment="appendix_h_duration", epochs=50, batch_size=80,
                lr=0.0005, optimizer="Adam", loss="summed six-slot CrossEntropyLoss",
                seed=2023, train="train_6tab", train_count=40000,
                validation=VALIDATION_JOBS, validation_count=3000,
                timestamp=False, input_length=30720, model=MODEL,
                reproduction_choices=REPRODUCTION_CHOICES)
METRICS = (overall_basic_accuracy, overall_basic_precision, overall_basic_recall)

METRIC_NAMES = ("Basic Accuracy", "Basic Precision", "Basic Recall")
# TMWF rows in complete_paper_version_of_TMWF.pdf, Table 9, p.21.
PUBLISHED = {
    "mixed_ratio": (3000, (61.3, 75.3, 73.5)),
    "mixed_real": (3000, (23.0, 58.9, 24.8)),
    "real_2tab": (1000, (12.7, 44.3, 15.1)),
    "real_3tab": (1000, (16.0, 54.3, 17.3)),
    "real_4tab": (1000, (34.4, 63.2, 35.6)),
}
CODE_FILES = ("train_appendix_h.py", "appendix_h_data.py", "models.py",
              "metrics.py", "simulation_openworld.py")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def report_memory(device):
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin" or sys.platform.startswith("linux"):
        size = peak if sys.platform == "darwin" else peak * 1024
        print(f"Process lifetime peak RSS: {size / 1e6:.2f} MB (not current RSS)")
    if device.type == "cuda":
        for name in ("memory_allocated", "memory_reserved", "max_memory_allocated"):
            print(f"CUDA {name}: {getattr(torch.cuda, name)(device) / 1e6:.2f} MB")
    elif device.type == "mps":
        for name in ("current_allocated_memory", "driver_allocated_memory"):
            query = getattr(torch.mps, name, None)
            if query is None:
                print(f"MPS {name}: unavailable")
            else:
                print(f"MPS {name}: {query() / 1e6:.2f} MB (snapshot)")


def verify_model_unchanged(model, reference):
    state = model.state_dict()
    if state.keys() != reference.keys():
        raise AssertionError("Model state keys changed during inference")
    for name, value in state.items():
        if not torch.equal(value.detach().cpu(), reference[name]):
            raise AssertionError(f"Model parameter/buffer changed during inference: {name}")


def input_hashes(job):
    if job.startswith("real_"):
        tabs = {"real_2tab": 2, "real_3tab": 3, "real_4tab": 4}[job]
        paths = [ROOT / "dataset/tbb realworld" / f"{tabs}tabs"]
    else:
        if job not in VALIDATION_JOBS:
            raise ValueError(f"Unsupported evaluation dataset: {job}")
        directory = ROOT / "dataset/generated_appendix_h" / job
        paths = [p for p in directory.iterdir() if p.is_file()]
        if not paths:
            raise FileNotFoundError(f"No inputs in {directory}")
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def load_final_model(device):
    # Trusted local training checkpoint contains NumPy/Python RNG objects.
    # Deserialize on CPU; never construct an optimizer or restore its state.
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    required = {"state_dict", "completed_epoch", "mode", "settings", "fingerprints",
                "environment", "reproduction_choices"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError("Unsupported checkpoint structure: missing required metadata")
    if checkpoint["completed_epoch"] != 50 or checkpoint["mode"] != "full":
        raise ValueError("Only full-run epoch 50 is accepted")
    if checkpoint["settings"] != SETTINGS or checkpoint["reproduction_choices"] != REPRODUCTION_CHOICES:
        raise ValueError("Checkpoint configuration differs from inspected Appendix H settings")
    fingerprints = checkpoint["fingerprints"]
    for name in CODE_FILES:
        relative = f"code/{name}"
        if fingerprints.get(relative) != sha256(ROOT / relative):
            raise ValueError(f"Source differs from training fingerprint: {relative}")
    for job in VALIDATION_JOBS:
        current = input_hashes(job)
        prefix = f"dataset/generated_appendix_h/{job}/"
        recorded = {k: v for k, v in fingerprints.items() if k.startswith(prefix)}
        # Training fingerprinted COCO and BAPM files, not annotation JSON.
        consumed = {k: v for k, v in current.items()
                    if Path(k).name.startswith(("COCO_", "BAPM_"))}
        if not recorded or consumed != recorded:
            raise ValueError(f"{job}: inputs differ from training fingerprints")
    print(f"Checkpoint: {CHECKPOINT}; epoch=50; mode=full", flush=True)
    print(f"Recorded training environment: {checkpoint['environment']}")
    print(f"Verified configuration: {SETTINGS}")
    model = TMWF_DFNet(**MODEL)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    reference = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if not all(torch.isfinite(v).all().item() for v in reference.values()):
        raise FloatingPointError("Checkpoint contains non-finite model state")
    del checkpoint
    return model, reference


def predict_component(job, tabs, model, device):
    synchronize(device)
    start = time.perf_counter()
    dataset = (load_real_tbb(tabs) if job.startswith("real_") else load_appendix_h(job))
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
    print(f"{job}: 1000 samples; loading/inference time={elapsed:.2f} s", flush=True)
    report_memory(device)
    return predictions, targets, elapsed


def score(predictions, targets, expected):
    if predictions.shape != (expected, 6, 51) or targets.shape != (expected, 6, 51):
        raise ValueError(f"Metrics require exactly {expected} samples")
    scores = tuple(metric(predictions, targets, 51) for metric in METRICS)
    if not np.isfinite(scores).all():
        raise FloatingPointError("Non-finite original Basic metric")
    return scores


def print_comparison(results):
    print("\nAppendix H / Table 9, TMWF: original metric rounding preserved.")
    print("Percentages; delta = reproduced minus published, in percentage points.")
    print(f"{'Dataset':<15} {'Samples':>7} {'Metric':<17} {'Reproduced %':>13} {'Published %':>12} {'Delta pp':>10}")
    for job, (count, published) in PUBLISHED.items():
        for name, value, target in zip(METRIC_NAMES, results[job], published):
            print(f"{job:<15} {count:>7} {name:<17} {100*value:>13.1f} {target:>12.1f} {100*value-target:>+10.1f}")
    print("Mixed delay-synthetic: NOT YET PERFORMED; no substituted results.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check-only", action="store_true", help="Check compatibility; no predictions.")
    modes.add_argument("--evaluate", action="store_true", help="Evaluate epoch 50 on 6000 component samples.")
    args = parser.parse_args()
    start = time.perf_counter()
    try:
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
            raise RuntimeError("Automatic MPS CPU fallback is enabled externally; unset it before evaluation")
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        print(f"Selected device: {device}; batch size=80; epoch 50 only", flush=True)
        print("Cross-backend numerical differences are possible. Mixed delay-synthetic: NOT YET PERFORMED.")
        checkpoint_before = sha256(CHECKPOINT)
        jobs = VALIDATION_JOBS + ("real_2tab", "real_3tab", "real_4tab")
        before = {job: input_hashes(job) for job in jobs}
        model, reference = load_final_model(device)
        results = {}
        if args.evaluate:
            for family, components in (("mixed_ratio", VALIDATION_JOBS), ("mixed_real", jobs[3:])):
                group_start = time.perf_counter()
                predictions, targets = [], []
                for tabs, job in zip((2, 3, 4), components):
                    p, y, elapsed = predict_component(job, tabs, model, device)
                    verify_model_unchanged(model, reference)
                    predictions.append(p)
                    targets.append(y)
                    if family == "mixed_real":
                        metric_start = time.perf_counter()
                        results[job] = score(p, y, 1000)
                        print(f"{job}: evaluation time={elapsed + time.perf_counter() - metric_start:.2f} s; predictions reused for mixed real")
                    gc.collect()
                # Joint metric computation, never an average of subgroup scores.
                results[family] = score(np.concatenate(predictions), np.concatenate(targets), 3000)
                print(f"{family}: 3000 samples; elapsed={time.perf_counter() - group_start:.2f} s", flush=True)
        verify_model_unchanged(model, reference)
        if sha256(CHECKPOINT) != checkpoint_before:
            raise AssertionError("Checkpoint hash changed")
        if {job: input_hashes(job) for job in jobs} != before:
            raise AssertionError("Source dataset hashes changed")
        report_memory(device)
        if args.evaluate:
            print_comparison(results)
            print("PASS: 6000 component predictions; 3000 samples per mixed set; "
                  "model parameters/buffers, checkpoint and source dataset hashes unchanged.")
        else:
            print("CHECK PASS: configuration, strict state loading, source fingerprints, "
                  "checkpoint/dataset hashes and model state verified. No predictions performed.")
        return 0
    except Exception:
        traceback.print_exc()
        print("FAIL: stopped without metric approximations, training, or automatic device fallback.")
        return 1
    finally:
        print(f"Total elapsed time including setup/integrity checks: {time.perf_counter() - start:.2f} s", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
