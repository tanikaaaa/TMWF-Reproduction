"""Explicit full Appendix H training or a separate one-epoch diagnostic profile."""

import argparse
import hashlib
import os
from pathlib import Path
import platform
import random
import resource
import sys
import tempfile
import time
import traceback

sys.dont_write_bytecode = True

import numpy as np
import torch
import torch.nn.functional as F

from appendix_h_data import load_appendix_h, generate_batches
from models import TMWF_DFNet
from metrics import (
    overall_basic_accuracy, overall_basic_precision, overall_basic_recall,
)


ROOT = Path(__file__).resolve().parents[1]
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


def check_dataset(dataset, expected, name):
    labels = np.asarray(dataset["label"])
    if len(dataset["data"]) != expected or labels.shape != (expected, 6):
        raise ValueError(f"{name}: expected {expected} traces and six labels per trace")
    if not np.all(np.isfinite(labels)) or not np.all((labels >= 0) & (labels <= 50)) or not np.all(labels == np.floor(labels)):
        raise ValueError(f"{name}: invalid labels")
    print(f"Dataset {name}: {expected} samples; label shape={labels.shape}", flush=True)


def check_batch(xb, yb):
    if xb.shape != (len(xb), 30720) or not 1 <= len(xb) <= 80 or yb.shape != (len(xb), 6):
        raise ValueError(f"Unexpected batch shapes: {xb.shape}, {yb.shape}")
    if not np.all(np.isfinite(yb)) or not np.all((yb >= 0) & (yb <= 50)) or not np.all(yb == np.floor(yb)):
        raise ValueError("Invalid batch labels")


def check_output(logits, count):
    if tuple(logits.shape) != (count, 6, 51):
        raise ValueError(f"Unexpected model output shape: {tuple(logits.shape)}")
    if not torch.isfinite(logits).all().item():
        raise FloatingPointError("Non-finite model output")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def memory_report(device, stage):
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin" or sys.platform.startswith("linux"):
        size = peak if sys.platform == "darwin" else peak * 1024
        print(f"{stage}: process peak RSS={size / 1e6:.2f} MB (not current RSS)", flush=True)
    else:
        print(f"{stage}: process RSS units unavailable on this platform", flush=True)
    if device.type == "cuda":
        for name in ("memory_allocated", "memory_reserved", "max_memory_allocated"):
            print(f"  CUDA {name}={getattr(torch.cuda, name)(device) / 1e6:.2f} MB", flush=True)
    elif device.type == "mps":
        for name in ("current_allocated_memory", "driver_allocated_memory"):
            query = getattr(torch.mps, name, None)
            print(f"  MPS {name}: {query() / 1e6 if query else 'unavailable'} MB", flush=True)
        print("  MPS readings are snapshots, not guaranteed peaks.", flush=True)


def fingerprints():
    """Record input/code bytes so resumption cannot silently use changed files."""
    paths = [ROOT / "code" / name for name in (
        "train_appendix_h.py", "appendix_h_data.py", "simulation_openworld.py",
        "models.py", "metrics.py",
    )]
    for job in ("train_6tab",) + VALIDATION_JOBS:
        directory = ROOT / "dataset" / "generated_appendix_h" / job
        paths.extend(directory.glob("COCO_*"))
        paths.extend(directory.glob("BAPM_*.npz"))
    result = {}
    for path in sorted(paths):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        result[str(path.relative_to(ROOT))] = digest.hexdigest()
    return result


def capture_rng(device):
    states = {"python": random.getstate(), "numpy": np.random.get_state(),
              "torch_cpu": torch.get_rng_state()}
    if device.type == "cuda":
        states["cuda"] = torch.cuda.get_rng_state_all()
    elif device.type == "mps":
        states["mps"] = torch.mps.get_rng_state()
    return states


def restore_rng(states, device):
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch_cpu"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state_all([state.cpu() for state in states["cuda"]])
    elif device.type == "mps":
        torch.mps.set_rng_state(states["mps"].cpu())


def save_checkpoint(path, payload):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {path}")
    temporary = path.with_suffix(".tmp")
    with temporary.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    # Atomic publication without replacing an existing destination, even in a race.
    os.link(temporary, path)
    temporary.unlink()


def run(args):
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("Automatic MPS CPU fallback is enabled externally; unset it before running")
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else
        "cpu"
    )
    mode = "full" if args.full else "profile"
    print(f"Mode: {mode}; selected device: {device}", flush=True)
    if mode == "profile":
        print("DIAGNOSTIC: one full-data epoch, not the official 50-epoch run.", flush=True)
    print(f"Configuration: {SETTINGS}", flush=True)
    print(f"Explicit reproduction choices: {REPRODUCTION_CHOICES}", flush=True)
    print("Resume is at completed epoch boundaries only. Mid-epoch work is lost on failure. "
          "Backend/kernel nondeterminism may prevent bit-for-bit results even with saved RNG states. "
          "No deterministic-backend settings are changed.", flush=True)
    random.seed(2023)
    np.random.seed(2023)
    torch.manual_seed(2023)
    # Fail before training if the selected backend cannot save/restore its RNG.
    if device.type == "mps" and not all(callable(getattr(torch.mps, name, None))
                                       for name in ("get_rng_state", "set_rng_state")):
        raise RuntimeError("This PyTorch version lacks MPS RNG checkpoint support")
    environment = dict(python=platform.python_version(), numpy=np.__version__,
                       torch=str(torch.__version__), platform=platform.platform(),
                       device=str(device), cuda_devices=torch.cuda.device_count(),
                       cuda_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None)
    file_hashes = fingerprints()
    checkpoint = None
    if args.resume:
        # Only load trusted checkpoints produced by this script (contains Python/NumPy RNG objects).
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if (checkpoint["mode"] != "full" or checkpoint["settings"] != SETTINGS
                or checkpoint["environment"] != environment or checkpoint["fingerprints"] != file_hashes):
            raise ValueError("Resume requires a full-run checkpoint with identical settings, files and environment")
        if not 0 < checkpoint["completed_epoch"] < 50:
            raise ValueError("Resume checkpoint must have completed between 1 and 49 epochs")

    parent = ROOT / "outputs" / "appendix_h" / mode
    parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run_", dir=parent))
    print(f"New persistent run directory: {run_dir}", flush=True)
    print("Existing runs are never overwritten; resume also writes to a new directory.", flush=True)
    memory_report(device, "Before loading")
    train = load_appendix_h("train_6tab")
    check_dataset(train, 40000, "train_6tab")
    validation = {"data": [], "label": []}
    for job in VALIDATION_JOBS:
        component = load_appendix_h(job)
        check_dataset(component, 1000, job)
        validation["data"].extend(component["data"])
        validation["label"].extend(component["label"])
        del component
    check_dataset(validation, 3000, "combined ratio validation")
    # Match train_test(): train shuffle first, validation shuffle second, once only.
    train_indices = np.arange(len(train["data"]))
    np.random.shuffle(train_indices)
    validation_indices = np.arange(len(validation["data"]))
    np.random.shuffle(validation_indices)
    model = TMWF_DFNet(**MODEL).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = torch.nn.CrossEntropyLoss()
    completed = 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        train_indices = checkpoint["train_indices"]
        validation_indices = checkpoint["validation_indices"]
        for indices, count in ((train_indices, 40000), (validation_indices, 3000)):
            if not np.array_equal(np.sort(indices), np.arange(count)):
                raise ValueError("Checkpoint sample ordering is not a complete permutation")
        completed = checkpoint["completed_epoch"]
        restore_rng(checkpoint["rng_states"], device)
        del checkpoint
    train_gen = generate_batches(train, train_indices, batch_size=80)
    validation_gen = generate_batches(validation, validation_indices, batch_size=80)
    memory_report(device, "After loading/model setup")
    durations = []
    final_epoch = 50 if mode == "full" else 1
    for epoch in range(completed + 1, final_epoch + 1):
        synchronize(device)
        start = time.perf_counter()
        model.train()
        losses = []
        train_seen = 0
        for _ in range((len(train_indices) + 79) // 80):
            xb, yb = next(train_gen)
            check_batch(xb, yb)
            train_seen += len(xb)
            logits = model(torch.tensor(xb, dtype=torch.float).to(device))
            check_output(logits, len(xb))
            targets = torch.tensor(yb, dtype=torch.long).to(device)
            optimizer.zero_grad()
            loss = 0
            for slot in range(6):
                loss = loss + criterion(logits[:, slot], targets[:, slot].to(device))
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}: {loss.item()}")
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if train_seen != 40000:
            raise ValueError(f"Expected 40000 training samples, saw {train_seen}")
        print(f"Epoch {epoch}/{final_epoch}: training loss={np.mean(losses):.8f}", flush=True)
        model.eval()
        probabilities, one_hot = [], []
        validation_seen = 0
        with torch.no_grad():
            for _ in range((len(validation_indices) + 79) // 80):
                xb, yb = next(validation_gen)
                check_batch(xb, yb)
                validation_seen += len(xb)
                logits = model(torch.tensor(xb, dtype=torch.float).to(device))
                check_output(logits, len(xb))
                targets = torch.tensor(yb, dtype=torch.long).to(device)
                if not torch.isfinite(logits).all().item():
                    raise FloatingPointError(f"Non-finite validation logits at epoch {epoch}")
                probabilities.append(logits.cpu().numpy())
                one_hot.append(F.one_hot(targets, 51).cpu().numpy())
        probabilities = np.concatenate(probabilities, axis=0)
        one_hot = np.concatenate(one_hot, axis=0)
        if validation_seen != 3000 or probabilities.shape != (3000, 6, 51) or one_hot.shape != (3000, 6, 51):
            raise ValueError("Combined validation must contain exactly 3000 samples")
        print(f"Epoch samples: train={train_seen}; combined ratio validation={validation_seen}", flush=True)
        scores = {metric.__name__: metric(probabilities, one_hot, 51) for metric in METRICS}
        if not all(np.isfinite(value) for value in scores.values()):
            raise FloatingPointError("Non-finite validation metric")
        print(f"Original Overall Basic validation metrics: {scores}", flush=True)
        synchronize(device)
        save_checkpoint(run_dir / f"epoch_{epoch:03d}.pth", {
            "state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "completed_epoch": epoch, "rng_states": capture_rng(device),
            "train_indices": train_indices, "validation_indices": validation_indices,
            "settings": SETTINGS, "mode": mode, "environment": environment,
            "fingerprints": file_hashes, "validation_metrics": scores,
            "parent_checkpoint": str(args.resume.resolve()) if args.resume else None,
            "reproduction_choices": REPRODUCTION_CHOICES,
        })
        synchronize(device)
        elapsed = time.perf_counter() - start
        durations.append(elapsed)
        print(f"Epoch wall time (training/validation/checkpoint): {elapsed:.2f} s; "
              f"estimated remaining: {np.mean(durations) * (final_epoch - epoch) / 60:.2f} min",
              flush=True)
        memory_report(device, f"After epoch {epoch}")
    if mode == "profile":
        print(f"DIAGNOSTIC approximate 50-epoch runtime: {durations[0] * 50 / 3600:.2f} hours "
              "plus initial setup; linear estimate, not a guarantee.", flush=True)
    print(f"Completed {mode} run; checkpoints: {run_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--full", action="store_true", help="Train through epoch 50.")
    modes.add_argument("--profile-one-epoch", action="store_true", help="Separate one-epoch full-data diagnostic.")
    parser.add_argument("--resume", type=Path, help="Trusted full-run epoch checkpoint; requires --full.")
    args = parser.parse_args()
    if args.resume and not args.full:
        parser.error("--resume requires --full; diagnostic profile checkpoints cannot resume full runs")
    start = time.perf_counter()
    try:
        run(args)
    except Exception:
        traceback.print_exc()
        print("FAILED: settings unchanged; no automatic device or batch-size fallback. "
              "Only successfully written epoch checkpoints are resumable.", flush=True)
        return 1
    finally:
        print(f"Invocation wall time including setup: {time.perf_counter() - start:.2f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
