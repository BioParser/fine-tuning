"""Training on more than one GPU of one computer (data parallel). Stdlib only at import.

`bioparser-train --num-gpus 2` starts torchrun, and torchrun starts one training process for each
GPU. Each process has a full copy of the model and trains on a different part of the examples.
Before each optimizer step, DDP calculates the mean of the gradients of all the processes.
torchrun gives each process the environment variables RANK, LOCAL_RANK and WORLD_SIZE. Unsloth
reads LOCAL_RANK and loads the model on that GPU.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from typing import Any


def world_size() -> int:
    """The number of training processes. It is 1 outside torchrun."""
    return int(os.environ.get("WORLD_SIZE", "1"))


def rank() -> int:
    """The number of this process, from 0 to world_size() - 1."""
    return int(os.environ.get("RANK", "0"))


def is_main_process() -> bool:
    """True for the process that writes the files and the progress messages."""
    return rank() == 0


def per_device_grad_accum(grad_accum: int, processes: int) -> int:
    """Return the gradient accumulation steps for each GPU.

    `grad_accum` is the value for one GPU. Each GPU accumulates grad_accum / processes steps, so the
    effective batch (batch_size * grad_accum) and the number of optimizer steps do not change.
    """
    if grad_accum < 1 or processes < 1:
        raise ValueError(f"grad_accum and processes must be 1 or more, not {grad_accum} and {processes}")
    if grad_accum % processes:
        raise ValueError(
            f"--grad-accum {grad_accum} is not a multiple of {processes} GPUs. Use a multiple, for "
            f"example {processes * max(1, round(grad_accum / processes))}, to keep the effective batch."
        )
    return grad_accum // processes


def torchrun_command(num_gpus: int, module: str, args: Sequence[str]) -> list[str]:
    """The command that starts `python -m module args` one time for each GPU."""
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",  # One computer: torchrun finds a free port for the processes
        f"--nproc-per-node={num_gpus}",
        "-m",
        module,
        *args,
    ]


def launch(num_gpus: int, module: str, args: Sequence[str]) -> int:
    """Start the training processes with torchrun, wait for them, and return the exit code."""
    import torch  # Only to count the GPUs. The parent process does not use a GPU.

    found = torch.cuda.device_count()  # This value obeys CUDA_VISIBLE_DEVICES
    if found < num_gpus:
        raise SystemExit(f"--num-gpus {num_gpus}: only {found} CUDA GPUs are available")
    env = dict(os.environ)
    # torchrun sets 1 thread for each process if OMP_NUM_THREADS is not set. The CPU work of the
    # training (activation offload, tokenization) is faster with a share of the CPU cores.
    env.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 1) // num_gpus)))
    return subprocess.run(torchrun_command(num_gpus, module, args), env=env, check=False).returncode


def gather_objects[T](obj: T) -> list[T]:
    """Return the `obj` of each process, in the sequence of the ranks. All processes must call it.

    Use only for small objects, for example metric counts: the function sends them with pickle.
    """
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return [obj]
    out: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(out, obj)
    return out


def destroy_process_group() -> None:
    """Release the connections between the processes at the end of the training."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
