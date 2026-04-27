"""
Write a script to benchmark the runtime of the all-reduce operation in the single-node multi-
process setup. The example code above may provide a reasonable starting point. Experiment with
varying the following settings:
all-reduce data size float32 data tensors ranging over 1MB, 10MB, 100MB, 1GB.
Number of GPUs/processes 2, 4, or 6.
Resource requirements: Up to 6 GPUs. Each benchmarking run should take less than 5
minutes.
Deliverable: Plot(s) and/or table(s) comparing the various settings, with 2-3 sentences of
commentary about your results and thoughts about how the various factors interact.

5.1.1 Best Practices for Benchmarking Distributed Applications
Throughout this portion of the assignment you will be benchmarking distributed applications to better
understand the overhead from communication. Here are a few best practices:
• Whenever possible, run benchmarks on the same machine to facilitate controlled comparisons.
• Perform several warm-up steps before timing the operation of interest. This is especially important for
NCCL communication calls. 5 iterations of warmup is generally sufficient.
• Call torch.cuda.synchronize() to wait for CUDA operations to complete when benchmarking on
GPUs. Note that this is necessary even when calling communication operations with async_op=False,
which returns when the operation is queued on the GPU (as opposed to when the communication
actually finishes).3
• Timings may vary slightly across different ranks, so it’s common to aggregate measurements across
ranks to improve estimates. You may find the all-gather collective (specifically the
dist.all_gather_object function) to be useful for collecting results from all ranks.
• In general, debug locally with Gloo on CPU, and then as required in a given problem, benchmark with
NCCL on GPU.

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
    # .add_local_dir("cs336_systems/configs", "/root/project/cs336_systems/configs")
)

def benchmark(rank, data_size, num_gpu, warmup_steps, measurement_steps, results):
    
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=num_gpu, rank=rank)

    data = torch.randn(data_size, device=f"cuda:{rank}")

    for _ in range(warmup_steps):
        torch.distributed.all_reduce(data)
        torch.cuda.synchronize()
    
    times = []
    for _ in range(measurement_steps):
        start_time = timeit.default_timer()
        torch.distributed.all_reduce(data)
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        times.append(end_time - start_time)

    all_results = [None] * num_gpu
    torch.distributed.all_gather_object(all_results, times)
    torch.distributed.destroy_process_group()
    if rank == 0:
        # print(f"Rank {rank} results: {all_results}")
        results[rank] = all_results

# @app.local_entrypoint()
@app.function(
    image=image,
    gpu="B200:6",
    timeout=60 * 120,
    # secrets=[modal.Secret.from_name("wandb-secret")],
    volumes={"/data": vol1, "/traces": traces_vol},
    max_containers=3,
)
def benchmark_distributed(data_size, num_gpu, warmup_steps, measurement_steps):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12390"
    with mp.Manager() as manager:
        results = manager.dict()
        mp.spawn(benchmark, args=(data_size, num_gpu, warmup_steps, measurement_steps, results), nprocs=num_gpu, join=True)
        # flatten results
        results = np.concatenate(results[0])
        ret = {
            "data_size": data_size,
            "num_gpu": num_gpu,
            "mean_time": np.mean(results),
            "std_time": np.std(results),
        }
        return ret

@app.local_entrypoint()
def main():
    import pandas as pd
    
    scale = 1024 * 1024 // 4
    data_sizes = [1 * scale, 10 * scale, 100 * scale, 1024 * scale]
    num_gpus = [2, 4, 6]
    warmup_steps = 5
    measurement_steps = 100

    results = []
    futures = []
    for data_size in data_sizes:
        for num_gpu in num_gpus:
            futures.append(benchmark_distributed.spawn(data_size, num_gpu, warmup_steps, measurement_steps))
    
    for future in futures:
        results.append(future.get())
    
    df = pd.DataFrame(results)
    df.to_csv("benchmark_distributed.csv", index=False)
    print(df)
