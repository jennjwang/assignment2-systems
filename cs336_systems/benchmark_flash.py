"""
Write a benchmarking script using triton.testing.do_bench that compares the performance
of your (partially) Triton implementation of FlashAttention-2 forward and backward passes
with a regular PyTorch implementation (i.e., not using FlashAttention).
Specifically, you will report a table that includes latencies for forward, backward, and the end-
to-end forward-backward pass, for both your Triton and PyTorch implementations. Randomly
generate any necessary inputs before you start benchmarking, and run the benchmark on a
single B200. Always use batch size 1 and causal masking. Sweep over the cartesian product of
sequence lengths of various powers of 2 from 128 up to 65536, embedding dimension sizes of
various powers of 2 from 16 up to size 128, and precisions of torch.bfloat16 and
torch.float32. You will likely need to adjust tile sizes depending on the input sizes.
Deliverable: A table of results comparing your implementation of FlashAttention-2 with the
PyTorch implementation, using the settings above and reporting forward, backward, and end-
to-end latencies.
"""

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
import pandas as pd

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
def benchmark_flash(d_model, seq, batch_size, dtype_str, device):

    from cs336_systems.triton_ff import TritonFlashAttentionAutograd
    import triton

    if d_model <= 32:
        B_q = 16
        B_k = 16
    elif d_model <= 64:
        B_q = 32
        B_k = 32
    else:
        B_q = 64
        B_k = 64

    dtype = getattr(torch, dtype_str)
    q = torch.randn(batch_size, seq, d_model, device=device, requires_grad=True, dtype=dtype)
    k = torch.randn(batch_size, seq, d_model, device=device, requires_grad=True, dtype=dtype)
    v = torch.randn(batch_size, seq, d_model, device=device, requires_grad=True, dtype=dtype)
    torch_attention = torch.nn.functional.scaled_dot_product_attention

    fwd_pytorch_ms = triton.testing.do_bench(lambda: torch_attention(q, k, v, is_causal=True))
    fwd_triton_ms = triton.testing.do_bench(lambda: TritonFlashAttentionAutograd.apply(q, k, v, True, B_q, B_k))
    
    output_pytorch = torch_attention(q, k, v, is_causal=True)
    output_triton = TritonFlashAttentionAutograd.apply(q, k, v, True, B_q, B_k)
    
    bwd_pytorch_ms = triton.testing.do_bench(lambda: output_pytorch.sum().backward(retain_graph=True))
    bwd_triton_ms = triton.testing.do_bench(lambda: output_triton.sum().backward(retain_graph=True))

    e2e_pytorch_ms = triton.testing.do_bench(
        lambda: torch_attention(
            q.detach().requires_grad_(True), k.detach().requires_grad_(True), 
            v.detach().requires_grad_(True), is_causal=True
        ).sum().backward())
    
    e2e_triton_ms = triton.testing.do_bench(
        lambda: TritonFlashAttentionAutograd.apply(
            q.detach().requires_grad_(True), k.detach().requires_grad_(True), 
            v.detach().requires_grad_(True), True, B_q, B_k
        ).sum().backward())

    return {
        "d_model": d_model,
        "seq": seq,
        "dtype": dtype_str,
        "fwd_pytorch": fwd_pytorch_ms,
        "fwd_triton": fwd_triton_ms,
        "bwd_pytorch": bwd_pytorch_ms,
        "bwd_triton": bwd_triton_ms,
        "e2e_pytorch": e2e_pytorch_ms,
        "e2e_triton": e2e_triton_ms,
    }

@app.local_entrypoint()
def main():
    d_models = [16, 32, 64, 128]
    seqs = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    dtypes = ["bfloat16", "float32"]
    batch_size = 1
    device = "cuda"
    spawns = [
        benchmark_flash.spawn(d_model, seq, batch_size, dtype_str, device=device)
        for d_model in d_models
        for seq in seqs
        for dtype_str in dtypes
    ]
    results = [spawn.get() for spawn in spawns]
    df = pd.DataFrame(results)
    df.to_csv("benchmark_flash.csv", index=False)
    print(df)
