"""Two-epoch diagnostic on tiny Appendix F subsets, not paper reproduction."""

import os
from pathlib import Path
import sys
import tempfile
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
)


def make_model(device):
    return TMWF_DFNet(
        embed_dim=256, nhead=8, dim_feedforward=1024,
        num_encoder_layers=2, num_decoder_layers=2,
        max_len=121, num_queries=6, cls=51, dropout=0.1,
    ).to(device)


def load_subset(job, count):
    full = load_appendix_f(job, timestamp=False)
    # Fixed leading samples only; release all other loaded samples on return.
    return {"data": full["data"][:count], "label": full["label"][:count]}


def summed_loss(logits, targets, criterion):
    loss = 0
    for slot in range(6):
        loss_slot = criterion(logits[:, slot], targets[:, slot])
        loss = loss + loss_slot
    if not torch.isfinite(loss).item():
        raise FloatingPointError(f"Non-finite summed six-slot loss: {loss.item()}")
    return loss


def run_diagnostic():
    print("DIAGNOSTIC TEST ONLY — not paper reproduction", flush=True)
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK=1 is set; automatic MPS CPU fallback is not allowed")
    np.random.seed(2023)
    torch.manual_seed(2023)
    device = torch.device(
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Selected device: {device}", flush=True)
    train = load_subset("train_6tab", 160)
    validation = load_subset("test_6tab", 80)
    if len(train["data"]) != 160 or len(validation["data"]) != 80:
        raise AssertionError("Diagnostic requires 160 training and 80 validation samples")
    print("Using first 160 training / first 80 validation samples; batch size=80; epochs=2.",
          flush=True)
    train_indices = np.arange(160)
    np.random.shuffle(train_indices)
    validation_indices = np.arange(80)
    np.random.shuffle(validation_indices)
    train_gen = DataGenerator(batch_size=80, dim=5120, page=6, timestamp=False).generate(
        train, train_indices
    )
    validation_gen = DataGenerator(batch_size=80, dim=5120, page=6, timestamp=False).generate(
        validation, validation_indices
    )
    model = make_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = torch.nn.CrossEntropyLoss()
    metric_functions = (
        overall_basic_accuracy, overall_basic_precision, overall_basic_recall,
        overall_advanced_accuracy, overall_advanced_precision, overall_advanced_recall,
    )
    for epoch in range(2):
        model.train()
        training_losses = []
        for _ in range(2):
            xb, yb = next(train_gen)
            logits = model(torch.tensor(xb, dtype=torch.float, device=device))
            targets = torch.tensor(yb, dtype=torch.long, device=device)
            optimizer.zero_grad()
            loss = summed_loss(logits, targets, criterion)
            loss.backward()
            optimizer.step()
            training_losses.append(loss.item())
        print(f"Diagnostic epoch {epoch + 1}/2 training loss: {np.mean(training_losses):.8f}",
              flush=True)
        model.eval()
        with torch.no_grad():
            xb, yb = next(validation_gen)
            logits = model(torch.tensor(xb, dtype=torch.float, device=device))
            targets = torch.tensor(yb, dtype=torch.long, device=device)
            validation_loss = summed_loss(logits, targets, criterion)
            probabilities = logits.detach().cpu().numpy()
            one_hot = F.one_hot(targets, 51).cpu().numpy()
        print(f"Diagnostic epoch {epoch + 1}/2 validation loss: {validation_loss.item():.8f}",
              flush=True)
        print("Original Basic/Advanced metrics (diagnostic subset):", flush=True)
        for metric in metric_functions:
            value = metric(probabilities, one_hot, 51)
            if not np.isfinite(value):
                raise FloatingPointError(f"Non-finite metric: {metric.__name__}={value}")

    with tempfile.TemporaryDirectory(prefix="tmwf_diagnostic_checkpoint_") as temporary:
        checkpoint = Path(temporary) / "checkpoint.pth"
        torch.save({"state_dict": model.state_dict()}, checkpoint)
        restored = make_model(device)
        saved = torch.load(checkpoint, map_location=device)
        restored.load_state_dict(saved["state_dict"], strict=True)
        original_state = model.state_dict()
        restored_state = restored.state_dict()
        if original_state.keys() != restored_state.keys():
            raise AssertionError("Checkpoint state_dict keys differ")
        for name, value in original_state.items():
            if not torch.equal(value.detach().cpu(), restored_state[name].detach().cpu()):
                raise AssertionError(f"Checkpoint state_dict mismatch: {name}")
        print("Checkpoint state_dict consistency verified (all parameters and buffers).", flush=True)
    print("Temporary checkpoint removed automatically.", flush=True)
    if device.type == "mps":
        torch.mps.synchronize()


def main():
    start = time.perf_counter()
    try:
        run_diagnostic()
    except Exception:
        traceback.print_exc()
        print("FAIL: diagnostic did not complete; no device fallback attempted.", flush=True)
        return 1
    else:
        print("PASS: two diagnostic epochs and checkpoint round-trip succeeded.", flush=True)
        return 0
    finally:
        print(f"Total elapsed time (including loading/checkpoint checks): "
              f"{time.perf_counter() - start:.2f} s", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
