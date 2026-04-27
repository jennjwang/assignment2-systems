"""
Modify your minimal DDP implementation to communicate a tensor with flattened gradients
from all parameters. Compare its performance with the minimal DDP implementation that issues
an all-reduce for each parameter tensor under the previously-used conditions (1 node x 2 GPUs,
xl model size as described in Section 2.1.2).
Deliverable: The measured time per training iteration and time spent communicating gradients
under distributed data parallel training with a single batched all-reduce call. 1-2 sentences
comparing the results when batching vs. individually communicating gradients.
"""

import modal
import torch
import timeit
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist
import os

app = modal.App("systems-jwang400")

vol1 = modal.Volume.from_name("baseline", create_if_missing=True)
traces_vol = modal.Volume.from_name("nsys-traces", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.13")
    .run_commands(
        "apt-get update && apt-get install -y wget",
        "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
    )
    .apt_install("libcap2-bin", "libdw1", "cuda-nsight-systems-13-2")
    .pip_install("torch~=2.11.0", "numpy", "regex", "wandb", "triton",
                    "omegaconf", "einops", "tqdm", "pandas", "pathlib")
    .add_local_dir("cs336-basics", "/root/cs336-basics", copy=True)
    .add_local_dir(".", "/root/cs336-systems", copy=True,
                    ignore=[".venv", ".git", "__pycache__", "*.pyc"])
    .run_commands("pip install /root/cs336-basics", "pip install /root/cs336-systems")
)

def benchmark(rank, world_size, warmup_steps, measurement_steps, results):
    from einops import rearrange
    from cs336_basics.model import BasicsTransformerLM
    from cs336_basics.optimizer import AdamW
    from cs336_basics.nn_utils import cross_entropy

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)

    model = BasicsTransformerLM(
        vocab_size=10_000,
        context_length=1024,
        d_model=2560,
        num_layers=32,
        num_heads=32,
        d_ff=10_240,
        rope_theta=10_000.0,
    ).to(f"cuda:{rank}")

    data   = torch.randint(0, 10_000, (4, 1024), device=f"cuda:{rank}")
    labels = torch.randint(0, 10_000, (4, 1024), device=f"cuda:{rank}")

    optimizer = AdamW(model.parameters(), lr=0.01)

    grad_params = [p for p in model.parameters() if p.requires_grad]
    flat = torch.zeros(sum(p.numel() for p in grad_params), 
                    device=f"cuda:{rank}", dtype=torch.float32)

    for _ in range(warmup_steps):
        optimizer.zero_grad()
        logits = model(data)
        loss = cross_entropy(rearrange(logits, "b t v -> (b t) v"), rearrange(labels, "b t -> (b t)"))
        loss.backward()

        offset = 0
        for p in grad_params:
            n = p.numel()
            if p.grad is not None:
                flat[offset:offset + n].copy_(p.grad.flatten())
            offset += n

        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(world_size)

        offset = 0
        for p in grad_params:
            n = p.numel()
            if p.grad is not None:
                p.grad.copy_(flat[offset:offset + n].view_as(p.grad))
            offset += n

        optimizer.step()
        torch.cuda.synchronize()

    training_times = []
    communication_times = []

    for _ in range(measurement_steps):
        optimizer.zero_grad()
        start_time = timeit.default_timer()

        logits = model(data)
        loss = cross_entropy(rearrange(logits, "b t v -> (b t) v"), rearrange(labels, "b t -> (b t)"))
        loss.backward()
        torch.cuda.synchronize()
        
        communication_start_time = timeit.default_timer()
        
        offset = 0
        for p in grad_params:
            n = p.numel()
            if p.grad is not None:
                flat[offset:offset + n].copy_(p.grad.flatten())
            offset += n

        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(world_size)

        offset = 0
        for p in grad_params:
            n = p.numel()
            if p.grad is not None:
                p.grad.copy_(flat[offset:offset + n].view_as(p.grad))
            offset += n

        torch.cuda.synchronize()
        communication_end_time = timeit.default_timer()

        optimizer.step()
        torch.cuda.synchronize()
        end_time = timeit.default_timer()

        training_times.append(end_time - start_time)
        communication_times.append(communication_end_time - communication_start_time)

    all_training_times = [None] * world_size
    all_communication_times = [None] * world_size
    dist.all_gather_object(all_training_times, training_times)
    dist.all_gather_object(all_communication_times, communication_times)
    dist.destroy_process_group()

    if rank == 0:
        results[0] = (all_training_times, all_communication_times)


@app.function(
    image=image,
    gpu="B200:2",
    timeout=60 * 120,
    volumes={"/data": vol1, "/traces": traces_vol},
)
def benchmark_distributed(warmup_steps, measurement_steps):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12390"

    world_size = 2
    with mp.Manager() as manager:
        results = manager.dict()
        mp.spawn(benchmark, args=(world_size, warmup_steps, measurement_steps, results), 
                 nprocs=world_size, join=True, daemon=False)

        all_training_times, all_communication_times = results[0]
        training_times     = np.concatenate(all_training_times)
        communication_times = np.concatenate(all_communication_times)

        ret = {
            "world_size": world_size,
            "mean_training_time": np.mean(training_times),
            "std_training_time": np.std(training_times),
            "mean_communication_time": np.mean(communication_times),
            "std_communication_time": np.std(communication_times),
            "communication_pct": np.mean(communication_times) / np.mean(training_times) * 100,
        }
        print(ret)
        return ret


@app.local_entrypoint()
def main():
    import pandas as pd
    result = benchmark_distributed.remote(warmup_steps=5, measurement_steps=10)
    df = pd.DataFrame([result])
    df.to_csv("flat_ddp_benchmark.csv", index=False)
    print(df)

