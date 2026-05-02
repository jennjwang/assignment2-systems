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
    from cs336_systems.fsdp.fsdp import FSDP
    
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

    model = FSDP(model, compute_dtype=torch.bfloat16)

    data   = torch.randint(0, 10_000, (4, 1024), device=f"cuda:{rank}")
    labels = torch.randint(0, 10_000, (4, 1024), device=f"cuda:{rank}")

    optimizer = AdamW(model.parameters(), lr=0.01)

    for _ in range(warmup_steps):
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(data)
            loss = cross_entropy(rearrange(logits, "b t v -> (b t) v"), rearrange(labels, "b t -> (b t)"))
        loss.backward()
        model.finish_gradient_synchronization()
        optimizer.step()
        torch.cuda.synchronize()

    training_times = []
    sync_stall_times = []

    for _ in range(measurement_steps):
        optimizer.zero_grad()
        start_time = timeit.default_timer()

        with nvtx.range("forward"):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(data)
                loss = cross_entropy(rearrange(logits, "b t v -> (b t) v"), rearrange(labels, "b t -> (b t)"))
        with nvtx.range("backward"):
            loss.backward()

        sync_stall_start_time = timeit.default_timer()
        with nvtx.range("finish_sync"):
            model.finish_gradient_synchronization()
        torch.cuda.synchronize()
        sync_stall_end_time = timeit.default_timer()

        with nvtx.range("optimizer"):
            optimizer.step()
        torch.cuda.synchronize()
        end_time = timeit.default_timer()

        training_times.append(end_time - start_time)
        sync_stall_times.append(sync_stall_end_time - sync_stall_start_time)

    all_training_times = [None] * num_gpu
    all_sync_stall_times = [None] * num_gpu
    dist.all_gather_object(all_training_times, training_times)
    dist.all_gather_object(all_sync_stall_times, sync_stall_times)
    dist.destroy_process_group()

    if rank == 0:
        results[0] = (all_training_times, all_sync_stall_times)


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

        all_training_times, all_sync_stall_times = results[0]
        training_times     = np.concatenate(all_training_times)
        sync_stall_times = np.concatenate(all_sync_stall_times)

        ret = {
            "num_gpu": num_gpu,
            "mean_training_time": np.mean(training_times),
            "std_training_time": np.std(training_times),
            "mean_sync_stall_time": np.mean(sync_stall_times),
            "std_sync_stall_time": np.std(sync_stall_times),
            "sync_stall_pct": np.mean(sync_stall_times) / np.mean(training_times) * 100,
            "communication_note": "sync_stall excludes overlapped forward all-gathers; use the nsys trace for total NCCL time",
        }
        print(ret)
        return ret


@app.function(
    image=image,
    gpu="B200:2",
    timeout=60 * 120,
    volumes={"/data": vol1, "/traces": traces_vol},
)
def profile_fsdp(trace_name: str = "fsdp_accounting", warmup_steps: int = 1, measurement_steps: int = 3):
    import subprocess, sys
    sys.path.insert(0, "/root/cs336-systems")
    trace_path = f"/traces/{trace_name}"
    rep_path = f"{trace_path}.nsys-rep"
    stats_path = f"{trace_path}_stats.txt"

    profile_cmd = [
        "nsys", "profile",
        "--trace=cuda,nvtx,cublas,cudnn,nccl,osrt",
        "--cuda-memory-usage=true",
        "--gpu-metrics-devices=all",
        "--trace-fork-before-exec=true",
        "--force-overwrite=true",
        "-o", trace_path,
        sys.executable, "-m",
        "cs336_systems.fsdp.fsdp_accounting",
        "--profile-target",
        "--warmup_steps", str(warmup_steps),
        "--measurement_steps", str(measurement_steps),
    ]
    subprocess.run(profile_cmd, check=False)

    reports = [
        "cuda_gpu_trace",
        "cuda_gpu_kern_sum",
        "cuda_gpu_mem_time_sum",
        "cuda_gpu_mem_size_sum",
        "cuda_api_sum",
        "nvtx_sum",
        "nvtx_gpu_proj_sum",
        "nccl_sum",
        "osrt_sum",
    ]
    with open(stats_path, "w") as f:
        for report in reports:
            f.write(f"\n\n===== nsys stats: {report} =====\n")
            f.flush()
            subprocess.run(
                ["nsys", "stats", "--report", report, rep_path],
                stdout=f,
                stderr=subprocess.STDOUT,
                check=False,
            )

    traces_vol.commit()
    return {"trace": rep_path, "stats": stats_path}


def run_profile_target(warmup_steps: int, measurement_steps: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12391"

    num_gpu = min(torch.cuda.device_count(), 2)
    if num_gpu < 1:
        raise RuntimeError("No CUDA devices available for FSDP profiling")

    with mp.Manager() as manager:
        results = manager.dict()
        mp.spawn(benchmark, args=(num_gpu, warmup_steps, measurement_steps, results), nprocs=num_gpu, join=True)
        if 0 in results:
            print(results[0], flush=True)


@app.local_entrypoint()
def profile(trace_name: str = "fsdp_accounting", warmup_steps: int = 1, measurement_steps: int = 3):
    print(profile_fsdp.remote(trace_name, warmup_steps, measurement_steps))


@app.local_entrypoint()
def main():
    import csv
    result = benchmark_distributed.remote(warmup_steps=5, measurement_steps=10)
    with open("fsdp_accounting.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow(result)
    for k, v in result.items():
        print(f"{k}: {v}")
    print(profile_fsdp.remote("fsdp_accounting"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-target", action="store_true")
    parser.add_argument("--warmup_steps", type=int, default=1)
    parser.add_argument("--measurement_steps", type=int, default=3)
    args = parser.parse_args()

    if args.profile_target:
        run_profile_target(args.warmup_steps, args.measurement_steps)
