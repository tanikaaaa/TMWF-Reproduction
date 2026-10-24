"""Read-only epoch-50 evaluation against TMWF Appendix F, Table 7 (PDF p.20)."""

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

from appendix_f_data import load_appendix_f
from models import TMWF_DFNet
from simulation_openworld import DataGenerator
from metrics import (
    overall_basic_accuracy, overall_basic_precision, overall_basic_recall,
    overall_advanced_accuracy, overall_advanced_precision, overall_advanced_recall,
    previous_accuracy, previous_precision, previous_recall,
)


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "outputs/appendix_f/full/run_97g8i7wd/epoch_050.pth"
MODEL = dict(embed_dim=256, nhead=8, dim_feedforward=1024,
             num_encoder_layers=2, num_decoder_layers=2, max_len=121,
             num_queries=6, cls=51, dropout=0.1)
OVERALL = (
    ("Basic Accuracy", overall_basic_accuracy),
    ("Basic Precision", overall_basic_precision),
    ("Basic Recall", overall_basic_recall),
    ("Advanced Accuracy", overall_advanced_accuracy),
    ("Advanced Precision", overall_advanced_precision),
    ("Advanced Recall", overall_advanced_recall),
)
# TMWF rows, not Adaptive-BAPM: complete_paper_version_of_TMWF.pdf, Table 7, p.20.
# Overall values follow OVERALL order; Previous values are Acc/Pre/Rec by page.
PUBLISHED = {
    "test_2tab": {
        "tabs": 2, "overall": [55.6, 71.9, 75.2, 19.3, 28.3, 40.2],
        "previous": [[80.2, 51.6], [84.2, 54.0], [71.8, 15.0]],
    },
    "test_4tab": {
        "tabs": 4, "overall": [64.7, 74.2, 73.4, 17.7, 25.4, 29.1],
        "previous": [[83.4, 59.8, 40.3, 29.7], [84.0, 63.0, 17.5, 2.2],
                     [80.1, 33.9, 11.0, 2.1]],
    },
    "test_6tab": {
        "tabs": 6, "overall": [65.1, 77.8, 67.6, 52.1, 64.8, 59.9],
        "previous": [[84.0, 69.5, 64.3, 62.4, 68.1, 80.4],
                     [83.9, 63.0, 54.2, 52.8, 59.6, 71.8],
                     [81.8, 61.5, 51.2, 48.8, 52.0, 67.2]],
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_hashes(job):
    directory = ROOT / "dataset/generated_appendix_f" / job
    paths = list(directory.glob("COCO_*")) + list(directory.glob("BAPM_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No generated inputs found in {directory}")
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(paths)}


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


def load_final_model(device):
    # This trusted local checkpoint includes NumPy/Python RNG and optimizer objects.
    # CPU deserialization avoids allocating the unused optimizer state on the GPU.
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    required = {"state_dict", "completed_epoch", "mode", "settings", "fingerprints", "environment"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError("Unsupported checkpoint structure; required training metadata is missing")
    if checkpoint["completed_epoch"] != 50 or checkpoint["mode"] != "full":
        raise ValueError("Only the completed full-run epoch-50 checkpoint is accepted")
    settings = checkpoint["settings"]
    if (settings.get("model") != MODEL or settings.get("batch_size") != 80
            or settings.get("timestamp") is not False or settings.get("epochs") != 50
            or settings.get("train") != "train_6tab" or settings.get("validation") != "test_6tab"):
        raise ValueError("Checkpoint settings do not match the inspected Appendix F training run")
    fingerprints = checkpoint["fingerprints"]
    for name in ("models.py", "metrics.py", "simulation_openworld.py", "appendix_f_data.py"):
        relative = f"code/{name}"
        if relative not in fingerprints or sha256(ROOT / relative) != fingerprints[relative]:
            raise ValueError(f"Source differs from training checkpoint fingerprint: {relative}")
    print(f"Checkpoint: {CHECKPOINT}; completed_epoch=50; mode=full")
    print(f"Recorded training environment: {checkpoint['environment']}")
    print(f"Evaluation PyTorch: {torch.__version__}; model settings: {settings['model']}")
    model = TMWF_DFNet(**settings["model"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    reference = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    # No optimizer is constructed, loaded or stepped. Discard its deserialized state.
    del checkpoint
    return model, reference, fingerprints


def evaluate_dataset(job, model, device, reference, training_fingerprints):
    synchronize(device)
    start = time.perf_counter()
    before = input_hashes(job)
    if job == "test_6tab":
        recorded = {name: value for name, value in training_fingerprints.items()
                    if name.startswith("dataset/generated_appendix_f/test_6tab/")}
        if before != recorded:
            raise ValueError("test_6tab input files differ from the training checkpoint fingerprints")
    dataset = load_appendix_f(job, timestamp=False)
    labels = np.asarray(dataset["label"])
    tabs = PUBLISHED[job]["tabs"]
    if len(dataset["data"]) != 5000 or labels.shape != (5000, 6):
        raise AssertionError(f"{job}: expected 5000 traces and label shape (5000, 6)")
    if (not np.isfinite(labels).all() or not np.all(labels == np.floor(labels))
            or not np.all((labels >= 0) & (labels <= 50)) or not np.all(labels[:, tabs:] == 50)):
        raise AssertionError(f"{job}: invalid label values or no-tab padding")
    # Fixed original numeric-chunk/sample order; no shuffle, resampling or filtering.
    generator = DataGenerator(batch_size=80, dim=5120, page=6, timestamp=False).generate(
        dataset, np.arange(5000)
    )
    outputs, truths = [], []
    evaluated = 0
    with torch.inference_mode():
        for _ in range((5000 + 79) // 80):
            xb, yb = next(generator)
            batch = min(80, 5000 - evaluated)
            if xb.shape != (batch, 30720) or yb.shape != (batch, 6):
                raise AssertionError(f"{job}: unexpected generator shapes {xb.shape}, {yb.shape}")
            if not np.array_equal(yb, labels[evaluated:evaluated + batch]):
                raise AssertionError(f"{job}: generator label ordering changed")
            logits = model(torch.tensor(xb, dtype=torch.float).to(device))
            if tuple(logits.shape) != (batch, 6, 51):
                raise AssertionError(f"{job}: unexpected output shape {tuple(logits.shape)}")
            if not torch.isfinite(logits).all().item():
                raise FloatingPointError(f"{job}: non-finite model output")
            targets = torch.tensor(yb, dtype=torch.long)
            outputs.append(logits.cpu().numpy())
            truths.append(F.one_hot(targets, 51).numpy())
            evaluated += batch
    predictions, targets = np.concatenate(outputs), np.concatenate(truths)
    if evaluated != 5000 or predictions.shape != (5000, 6, 51) or targets.shape != (5000, 6, 51):
        raise AssertionError(f"{job}: incomplete evaluation")
    print(f"\n{job}: original metric functions on exactly {evaluated} samples", flush=True)
    overall = [function(predictions, targets, 51) for _, function in OVERALL]
    # Table 7 excludes redundant positions only for Previous/per-page metrics.
    # Slicing is required: previous_accuracy iterates every supplied page.
    previous = [
        previous_accuracy(predictions[:, :tabs], targets[:, :tabs], tabs),
        previous_precision(predictions[:, :tabs], targets[:, :tabs], 51, tabs),
        previous_recall(predictions[:, :tabs], targets[:, :tabs], 51, tabs),
    ]
    if not np.isfinite(overall).all() or np.shape(previous) != (3, tabs) or not np.isfinite(previous).all():
        raise ValueError(f"{job}: invalid metric results")
    verify_model_unchanged(model, reference)
    if input_hashes(job) != before:
        raise AssertionError(f"{job}: input files changed during evaluation")
    synchronize(device)
    print(f"{job}: elapsed evaluation time including loading/metrics/integrity checks: "
          f"{time.perf_counter() - start:.2f} s", flush=True)
    report_memory(device)
    print(f"{job}: model parameters/buffers and input-file hashes unchanged.")
    return overall, previous


def print_comparison(results):
    print("\nAppendix F / Table 7 (complete paper p.20), TMWF rows")
    print("Original function rounding preserved; percentages; delta = reproduced - published (pp).")
    print(f"{'Dataset':<12} {'Samples':>7} {'Metric':<28} {'Reproduced %':>13} {'Published %':>12} {'Delta pp':>10}")
    for job, (overall, previous) in results.items():
        rows = [(name, value, published) for (name, _), value, published
                in zip(OVERALL, overall, PUBLISHED[job]["overall"])]
        for metric, values, published in zip(("Accuracy", "Precision", "Recall"), previous,
                                             PUBLISHED[job]["previous"]):
            rows.extend((f"Previous {metric} page {index}", value, target)
                        for index, (value, target) in enumerate(zip(values, published), 1))
        for name, value, published in rows:
            percentage = 100 * value
            print(f"{job:<12} {5000:>7} {name:<28} {percentage:>13.1f} {published:>12.1f} "
                  f"{percentage - published:>+10.1f}")


def main():
    start = time.perf_counter()
    try:
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
            raise RuntimeError("Automatic MPS CPU fallback is enabled externally; unset it before evaluation")
        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        )
        print(f"Selected device: {device}; batch size=80; epoch 50 only", flush=True)
        print("Evaluation keeps original sample order; cross-backend numerical differences are possible.")
        checkpoint_hash = sha256(CHECKPOINT)
        model, reference, fingerprints = load_final_model(device)
        results = {}
        for job in PUBLISHED:
            results[job] = evaluate_dataset(job, model, device, reference, fingerprints)
            gc.collect()  # Dataset/generator locals have been released before the next load.
        if sha256(CHECKPOINT) != checkpoint_hash:
            raise AssertionError("Checkpoint file changed during evaluation")
        print_comparison(results)
        print("PASS: exactly 5000 samples per dataset; checkpoint and model state unchanged. No files written.")
        return 0
    except Exception:
        traceback.print_exc()
        print("FAIL: evaluation stopped; no metric approximation, training or automatic device fallback.")
        return 1
    finally:
        print(f"Total elapsed time: {time.perf_counter() - start:.2f} s", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
