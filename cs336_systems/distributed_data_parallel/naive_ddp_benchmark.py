"""
In this naïve DDP implementation, parameter gradients are individually all-reduced across ranks
after each backward pass. To better understand the overhead of data parallel training, create a
script to benchmark your previously-implemented language model when trained with this naïve
implementation of DDP. Measure the total time per training step and the proportion of time
spent on communicating gradients. Collect measurements in the single-node setting (1 node x 2
GPUs) for the xl model size described in Section 2.1.2.
Deliverable: A description of your benchmarking setup, along with the measured time per
training iteration and time spent communicating gradients for each setting.
"""

import modal
import torch
import torch.cuda.nvtx as nvtx
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

def benchmark(rank, num_gpu, warmup_steps, measurement_steps, results):
    from einops import rearrange
    from cs336_basics.model import BasicsTransformerLM
    from cs336_basics.optimizer import AdamW
    from cs336_basics.nn_utils import cross_entropy
    from cs336_systems.distributed_data_parallel.ddp_overlap import DDPOverlap

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", init_method="env://", world_size=num_gpu, rank=rank)

    # model = DDPOverlap(BasicsTransformerLM(
    #     vocab_size=10_000,
    #     context_length=1024,
    #     d_model=2560,
    #     num_layers=32,
    #     num_heads=32,
    #     d_ff=10_240,
    #     rope_theta=10_000.0,
    # ).to(f"cuda:{rank}"))
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

    for _ in range(warmup_steps):
        optimizer.zero_grad()
        logits = model(data)
        loss = cross_entropy(rearrange(logits, "b t v -> (b t) v"), rearrange(labels, "b t -> (b t)"))
        loss.backward()
        # model.finish_gradient_synchronization()
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(num_gpu)
        optimizer.step()
        torch.cuda.synchronize()

    training_times = []
    communication_times = []

    for _ in range(measurement_steps):
        optimizer.zero_grad()
        start_time = timeit.default_timer()

        with nvtx.range("forward"):
            logits = model(data)
            loss = cross_entropy(rearrange(logits, "b t v -> (b t) v"), rearrange(labels, "b t -> (b t)"))
        with nvtx.range("backward"):
            loss.backward()
        torch.cuda.synchronize()

        communication_start_time = timeit.default_timer()
        # with nvtx.range("finish_sync"):
        #     model.finish_gradient_synchronization()
        with nvtx.range("allreduce"):
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                    param.grad.div_(num_gpu)
        torch.cuda.synchronize()
        communication_end_time = timeit.default_timer()

        with nvtx.range("optimizer"):
            optimizer.step()
        torch.cuda.synchronize()
        end_time = timeit.default_timer()

        training_times.append(end_time - start_time)
        communication_times.append(communication_end_time - communication_start_time)

    all_training_times = [None] * num_gpu
    all_communication_times = [None] * num_gpu
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

    num_gpu = 2
    with mp.Manager() as manager:
        results = manager.dict()
        mp.spawn(benchmark, args=(num_gpu, warmup_steps, measurement_steps, results), nprocs=num_gpu, join=True)

        all_training_times, all_communication_times = results[0]
        training_times     = np.concatenate(all_training_times)
        communication_times = np.concatenate(all_communication_times)

        ret = {
            "num_gpu": num_gpu,
            "mean_training_time": np.mean(training_times),
            "std_training_time": np.std(training_times),
            "mean_communication_time": np.mean(communication_times),
            "std_communication_time": np.std(communication_times),
            "communication_pct": np.mean(communication_times) / np.mean(training_times) * 100,
        }
        print(ret)
        return ret


@app.function(
    image=image,
    gpu="B200:2",
    timeout=60 * 120,
    volumes={"/data": vol1, "/traces": traces_vol},
)
def profile_ddp(trace_name: str = "ddp"):
    import subprocess, sys
    sys.path.insert(0, "/root/cs336-systems")
    trace_path = f"/traces/{trace_name}"
    cmd = [
        "nsys", "profile",
        "--trace=cuda,nvtx,cublas",
        "--force-overwrite=true",
        "-o", trace_path,
        sys.executable, "-m",
        "cs336_systems.distributed_data_parallel.naive_ddp_benchmark",
    ]
    subprocess.run(cmd, check=False)
    traces_vol.commit()
    return f"{trace_path}.nsys-rep"


@app.local_entrypoint()
def main():
    import pandas as pd
    # Timing benchmark
    result = benchmark_distributed.remote(warmup_steps=5, measurement_steps=10)
    df = pd.DataFrame([result])
    df.to_csv("naive_ddp_benchmark.csv", index=False)
    print(df)
    # Run profile_ddp after swapping the implementation manually
    print(profile_ddp.remote("ddp_naive"))


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12390"
    num_gpu = 2
    import torch.multiprocessing as mp
    mp.spawn(
        benchmark,
        args=(num_gpu, 5, 3, {}),
        nprocs=num_gpu,
        join=True,
    )
