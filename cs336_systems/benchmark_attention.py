'''
Benchmark your attention implementation at different scales. Write a script that will:
(i) Fix the batch size to 8 and don’t use multihead attention (i.e. remove the head
dimension).
(ii) Iterate through the cartesian product of [16, 32, 64, 128] for the head embedding
dimension 𝑑model, and [256, 1024, 4096, 8192, 16384] for the sequence length.
(iii) Create random inputs 𝑄, 𝐾, 𝑉 for the appropriate size.
(iv) Time 100 forward passes through attention using the inputs.
(v) Measure how much memory is in use before the backward pass starts, and time 100
backward passes.
(vi) Make sure to warm up, and to call torch.cuda.synchronize() after each forward/
backward pass.
Depending on your GPU, some of these configurations are expected to run out of memory.
Report the timings (or out-of-memory errors) you get for these configurations. At what size
do you get out-of-memory errors? Do the accounting for the memory usage of attention in one
of the smallest configurations you find that runs out of memory (you can use the equations
for memory usage of Transformers from Assignment 1). How does the memory saved for
backward change with the sequence length? What would you do to eliminate this memory
cost?
Deliverable: A table with your timings, your calculations for the memory usage, and a 1-2
paragraph response.
'''


import torch
import timeit
import numpy as np
import modal

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
    .pip_install("torch~=2.11.0", "numpy", "regex", "wandb",
                    "omegaconf", "einops", "tqdm", "pandas", "pathlib")
    .add_local_dir("cs336-basics", "/root/cs336-basics", copy=True)
    .add_local_dir(".", "/root/cs336-systems", copy=True,
                    ignore=[".venv", ".git", "__pycache__", "*.pyc"])
    .run_commands("pip install /root/cs336-basics", "pip install /root/cs336-systems")
    .add_local_dir("cs336_systems/configs", "/root/project/cs336_systems/configs")
)

@app.function(
    image=image,
    gpu="B200",
    timeout=60 * 120,
    # secrets=[modal.Secret.from_name("wandb-secret")],
    volumes={"/data": vol1, "/traces": traces_vol},
    max_containers=3,
)
def benchmark_attention(d_model, seq, batch_size, warmup_steps, device):
    from cs336_basics.model import scaled_dot_product_attention
    scaled_dot_product_attention = torch.compile(scaled_dot_product_attention)
    try:
        for _ in range(warmup_steps):
            q = torch.randn(batch_size, seq, d_model, device=device, requires_grad=True)
            k = torch.randn(batch_size, seq, d_model, device=device, requires_grad=True)
            v = torch.randn(batch_size, seq, d_model, device=device, requires_grad=True)
            scaled_dot_product_attention(q, k, v)

        fwd_times = []
        for _ in range(100):
            start_time = timeit.default_timer()
            scaled_dot_product_attention(q, k, v)
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            fwd_times.append(end_time - start_time)

        output = scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        mem = torch.cuda.memory_allocated() / (1024 ** 2)
        del output

        bwd_times = []
        for _ in range(100):
            q = torch.randn(batch_size, seq, d_model, device=device, requires_grad=True)
            k = torch.randn(batch_size, seq, d_model, device=device, requires_grad=True)
            v = torch.randn(batch_size, seq, d_model, device=device, requires_grad=True)
            output = scaled_dot_product_attention(q, k, v)
            torch.cuda.synchronize()
            start_time = timeit.default_timer()
            output.sum().backward()
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            bwd_times.append(end_time - start_time)

        print(
            f"d_model={d_model:4d} seq={seq:6d}: "
            f"fwd={np.mean(fwd_times)*1000:.2f}ms ± {np.std(fwd_times)*1000:.2f}ms  "
            f"bwd={np.mean(bwd_times)*1000:.2f}ms ± {np.std(bwd_times)*1000:.2f}ms  "
            f"mem_before_bwd={mem:.1f}MB"
        )

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if isinstance(e, RuntimeError) and "out of memory" not in str(e).lower():
            raise
        print(f"d_model={d_model:4d} seq={seq:6d}: OOM")
        torch.cuda.empty_cache()

@app.local_entrypoint()
def main():
    d_models = [16, 32, 64, 128]
    seqs = [256, 1024, 4096, 8192, 16384]
    batch_size = 8
    warmup_steps = 5
    device = "cuda"
    spawns = [
        benchmark_attention.spawn(d_model, seq, batch_size, warmup_steps, device)
        for d_model in d_models
        for seq in seqs
    ]
    for spawn in spawns:
        spawn.get()
