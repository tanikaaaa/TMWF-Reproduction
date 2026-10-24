"""Run one synthetic forward/backward/Adam step with the original TMWF_DFNet."""

import os
import sys
import time

# Keep imports from creating bytecode files beside the research implementation.
sys.dont_write_bytecode = True

if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
    raise RuntimeError(
        "PYTORCH_ENABLE_MPS_FALLBACK=1 is set; unset it to test native MPS "
        "execution without automatic CPU fallback."
    )

import numpy as np
import torch

from models import TMWF_DFNet


def main():
    np.random.seed(2023)
    torch.manual_seed(2023)
    device = torch.device(
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Selected device: {device}", flush=True)

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
    inputs = torch.tensor(
        np.random.choice((-1, 1), size=(1, 30720)).astype(np.float32),
        dtype=torch.float32,
        device=device,
    )
    targets = torch.tensor(
        np.random.randint(0, 51, size=(1, 6)), dtype=torch.long, device=device
    )
    print(f"Input shape: {tuple(inputs.shape)}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = torch.nn.CrossEntropyLoss()

    if device.type == "mps":
        torch.mps.synchronize()
    start = time.perf_counter()
    outputs = model(inputs)
    print(f"Output shape: {tuple(outputs.shape)}", flush=True)
    if tuple(outputs.shape) != (1, 6, 51):
        raise AssertionError(f"Expected output shape (1, 6, 51), got {tuple(outputs.shape)}")
    optimizer.zero_grad()
    loss = 0
    for slot in range(6):
        loss_slot = criterion(outputs[:, slot], targets[:, slot].to(device))
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
    if device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start

    print(f"Loss: {loss.item():.8f}")
    print(f"Finite parameter gradients checked: {gradient_count}")
    print(f"Forward/backward/optimizer elapsed time (including checks): {elapsed:.6f} s")
    print("PASS")


if __name__ == "__main__":
    # Let failures propagate with their exact exception and full traceback.
    main()
