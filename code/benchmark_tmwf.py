"""Benchmark one warm-up and three timed steps per batch on native Apple MPS."""

import gc
import os
import statistics
import sys
import time
import traceback

sys.dont_write_bytecode = True

if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
    raise RuntimeError(
        "PYTORCH_ENABLE_MPS_FALLBACK=1 is set; unset it before benchmarking "
        "native MPS without automatic CPU fallback."
    )

import numpy as np
import torch

from models import TMWF_DFNet


BATCH_SIZES = (32, 64, 80)
MEMORY_LIMIT = 10_000_000_000  # 10 GB, decimal bytes.


def reset_peak_statistics():
    reset = getattr(torch.mps, "reset_peak_memory_stats", None)
    queries = tuple(name for name in ("max_memory_allocated", "max_driver_allocated_memory")
                    if callable(getattr(torch.mps, name, None)))
    if callable(reset) and queries:
        try:
            reset()
            for name in queries:
                getattr(torch.mps, name)()
        except NotImplementedError:
            pass
        else:
            print("MPS peak-memory statistics reset for this batch.", flush=True)
            return queries
    print("MPS peak-memory statistics unavailable; reporting measured snapshots only.",
          flush=True)
    return ()


def memory_snapshot(peak_queries=()):
    result = {}
    for name in ("current_allocated_memory", "driver_allocated_memory") + peak_queries:
        query = getattr(torch.mps, name, None)
        result[name] = query() if query is not None else None
    return result


def step(model, optimizer, criterion, inputs, targets):
    outputs = model(inputs)
    expected = (inputs.shape[0], 6, 51)
    if tuple(outputs.shape) != expected:
        raise AssertionError(f"Expected output shape {expected}, got {tuple(outputs.shape)}")
    optimizer.zero_grad()
    loss = 0
    for slot in range(6):
        loss_slot = criterion(outputs[:, slot], targets[:, slot].to(inputs.device))
        loss = loss + loss_slot
    if not torch.isfinite(loss).item():
        raise FloatingPointError(f"Non-finite loss: {loss.item()}")
    loss.backward()
    gradient_count = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            gradient_count += 1
            if not torch.isfinite(parameter.grad).all().item():
                raise FloatingPointError(f"Non-finite gradient: {name}")
    if gradient_count == 0:
        raise AssertionError("Backward produced no parameter gradients")
    optimizer.step()


def benchmark_batch(batch_size):
    model = optimizer = criterion = inputs = targets = None
    try:
        torch.mps.synchronize()
        peak_queries = reset_peak_statistics()
        np.random.seed(2023)
        torch.manual_seed(2023)
        device = torch.device("mps")
        model = TMWF_DFNet(
            embed_dim=256,
            nhead=8,
            dim_feedforward=1024,
            num_encoder_layers=2,
            num_decoder_layers=2,
            max_len=121,
            num_queries=6,
            cls=51,
            dropout=0.1,
        ).to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        criterion = torch.nn.CrossEntropyLoss()
        inputs = torch.tensor(
            np.random.choice((-1, 1), size=(batch_size, 30720)).astype(np.float32),
            dtype=torch.float32, device=device,
        )
        targets = torch.tensor(
            np.random.randint(0, 51, size=(batch_size, 6)),
            dtype=torch.long, device=device,
        )
        snapshots = [memory_snapshot(peak_queries)]
        step(model, optimizer, criterion, inputs, targets)
        torch.mps.synchronize()
        snapshots.append(memory_snapshot(peak_queries))

        durations = []
        for _ in range(3):
            torch.mps.synchronize()
            start = time.perf_counter()
            step(model, optimizer, criterion, inputs, targets)
            torch.mps.synchronize()
            durations.append(time.perf_counter() - start)
            snapshots.append(memory_snapshot(peak_queries))

        print(f"Median timed step: {statistics.median(durations):.6f} s", flush=True)
        print("Timing includes shape/loss/gradient checks.", flush=True)
        for name, value in snapshots[-1].items():
            display = "unavailable" if value is None else f"{value} bytes ({value / 1e9:.3f} GB)"
            print(f"torch.mps.{name}(): {display}", flush=True)
            measured = [snapshot[name] for snapshot in snapshots if snapshot[name] is not None]
            if measured:
                print(f"  Highest measured: {max(measured)} bytes "
                      f"({max(measured) / 1e9:.3f} GB)", flush=True)
        observed = [value for snapshot in snapshots for value in snapshot.values()
                    if value is not None]
        exceeded = any(value > MEMORY_LIMIT for value in observed)
        print(f"Batch {batch_size}: PASS", flush=True)
        if exceeded:
            print("STOP: measured MPS allocated/driver/peak memory exceeded 10 GB; "
                  "no larger batch will be tested.", flush=True)
        return exceeded
    finally:
        # Drop all per-batch objects before releasing the MPS allocator cache.
        model = optimizer = criterion = inputs = targets = None
        gc.collect()
        torch.mps.empty_cache()


def main():
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS is unavailable; this benchmark requires Apple MPS and will not use CPU.")
    print("Device: mps; memory stop threshold: 10 GB (10,000,000,000 bytes).", flush=True)
    print("Snapshot readings are not guaranteed peaks; peak statistics are used if supported.", flush=True)
    for index, batch_size in enumerate(BATCH_SIZES):
        print(f"\nBatch {batch_size}: one warm-up and three timed steps", flush=True)
        try:
            exceeded = benchmark_batch(batch_size)
        except Exception:
            print(f"Batch {batch_size}: FAIL; stopping benchmark.", flush=True)
            traceback.print_exc()
            for remaining in BATCH_SIZES[index + 1:]:
                print(f"Batch {remaining}: NOT RUN", flush=True)
            return 1
        if exceeded:
            for remaining in BATCH_SIZES[index + 1:]:
                print(f"Batch {remaining}: NOT RUN (memory threshold)", flush=True)
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
