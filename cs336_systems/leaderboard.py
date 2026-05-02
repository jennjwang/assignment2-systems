import torch
import modal
import torch.distributed as dist

app = modal.App("systems-jwang400")

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
    .pip_install("torch~=2.11.0", "numpy", "regex",
                    "einops", "einx", "jaxtyping", "tqdm")
    .add_local_dir("cs336-basics", "/root/cs336-basics", copy=True)
    .add_local_dir(".", "/root/cs336-systems", copy=True,
                    ignore=[".venv", ".git", "__pycache__", "*.pyc"])
    .run_commands(
        "pip install --force-reinstall --no-cache-dir --no-deps /root/cs336-basics",
        "pip install --force-reinstall --no-cache-dir --no-deps /root/cs336-systems",
    )
)

class Config:
    ctx_len = 32768
    vocab_size = 151936
    d_model = 4096
    d_ff = 11008
    num_layers = 34
    num_heads = 32
    torch_dtype = torch.bfloat16
    batch_size = 2
cfg = Config()

activation_checkpointing = True
checkpoint_group_size = 6
checkpoint_num_layers = None
use_torch_compile = True
compile_mode = "default"
fsdp_prefetch_distance = 2

# attn_B_q = 128
# attn_B_k = 128
# attn_dq_B_q = 128
# attn_dq_B_k = 128
# attn_dkdv_B_q = 128
# attn_dkdv_B_k = 128

bench_warmup_ms = 100
bench_rep_ms = 100


def _set_config_if_exists(root, path, value):
    obj = root
    parts = path.split(".")
    for part in parts[:-1]:
        if not hasattr(obj, part):
            return False
        obj = getattr(obj, part)
    if not hasattr(obj, parts[-1]):
        return False
    setattr(obj, parts[-1], value)
    return True


def configure_torch_compile():
    import torch._dynamo.config as dynamo_config
    import torch._inductor.config as inductor_config

    # Static leaderboard shapes should not need dynamic-shape recompiles.
    _set_config_if_exists(dynamo_config, "cache_size_limit", 64)
    _set_config_if_exists(dynamo_config, "accumulated_cache_size_limit", 1024)
    _set_config_if_exists(dynamo_config, "dynamic_shapes", False)
    _set_config_if_exists(dynamo_config, "automatic_dynamic_shapes", False)
    _set_config_if_exists(dynamo_config, "assume_static_by_default", True)
    _set_config_if_exists(dynamo_config, "capture_scalar_outputs", True)

    # Do not silently fall back if compile breaks; a hidden eager fallback makes
    # benchmark comparisons misleading.
    _set_config_if_exists(dynamo_config, "suppress_errors", False)

    _set_config_if_exists(inductor_config, "fx_graph_cache", True)
    _set_config_if_exists(inductor_config, "triton.cudagraphs", False)


def run_leaderboard(rank, num_gpu, results, profile_mode=False, warmup_steps=1, measurement_steps=1):
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import cs336_basics.model as basics_model
    from torch.optim import AdamW
    from cs336_systems.fused_linear_ce import default_chunk_size as fused_ce_chunk_size
    from cs336_systems.triton_ff import flash_bwd_dkdv_kernel, flash_bwd_dq_kernel, flash_fwd_kernel
    import triton

    torch.cuda.set_device(rank)
    torch.set_float32_matmul_precision("high")
    configure_torch_compile()
    dist.init_process_group(backend="nccl", init_method="env://", world_size=num_gpu, rank=rank)

    labels, targets = torch.randint(high=cfg.vocab_size, size=(2, cfg.batch_size,
    cfg.ctx_len))
    labels = labels.to(f"cuda:{rank}")
    targets = targets.to(f"cuda:{rank}")

    transformer = basics_model.BasicsTransformerLM(
        context_length=cfg.ctx_len,
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        use_flash_attention=True,
        # flash_attention_B_q=attn_B_q,
        # flash_attention_B_k=attn_B_k,
        # flash_attention_dq_B_q=attn_dq_B_q,
        # flash_attention_dq_B_k=attn_dq_B_k,
        # flash_attention_dkdv_B_q=attn_dkdv_B_q,
        # flash_attention_dkdv_B_k=attn_dkdv_B_k,
        gradient_checkpointing=activation_checkpointing,
        checkpoint_group_size=checkpoint_group_size,
        checkpoint_num_layers=checkpoint_num_layers,
    ).to(device=f"cuda:{rank}")

    if use_torch_compile:
        # compile each TransformerBlock and the final norm independently before FSDP wrapping.
        for i in range(len(transformer.layers)):
            transformer.layers[i] = torch.compile(
                transformer.layers[i], mode=compile_mode, fullgraph=False, dynamic=False,
            )
        transformer.ln_final = torch.compile(
            transformer.ln_final, mode=compile_mode, fullgraph=False, dynamic=False,
        )
        if rank == 0:
            print(f"compiled {len(transformer.layers)} layers + ln_final with mode={compile_mode}", flush=True)

    from cs336_systems.fsdp.fsdp import FSDP
    model = FSDP(transformer, compute_dtype=cfg.torch_dtype, prefetch_distance=fsdp_prefetch_distance)
    fp32_master_params = True

    optimizer = AdamW(model.parameters(), lr=0.01, fused=True)

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        loss = model(labels, targets)
        loss.backward()
        model.finish_gradient_synchronization()
        optimizer.step()
    
    torch.cuda.synchronize()                                                                                                                  
    dist.barrier()
    timing_results = triton.testing.do_bench(train_step, rep=bench_rep_ms, warmup=bench_warmup_ms)

    if rank == 0:
        results[0] = timing_results
        print(
            f"leaderboard: checkpoint_group_size={checkpoint_group_size}, "
            f"checkpoint_num_layers={checkpoint_num_layers}, "
            f"activation_checkpointing={activation_checkpointing}, "
            f"fused_ce_chunk_size={fused_ce_chunk_size}, "
            f"compute_dtype={cfg.torch_dtype}, "
            f"fp32_master_params={fp32_master_params}, "
            f"fsdp_prefetch_distance={fsdp_prefetch_distance}, "
            f"use_torch_compile={use_torch_compile}, "
            f"compile_mode={compile_mode}",
            flush=True,
        )
        print(f'flash_fwd_kernel.best_config = "{flash_fwd_kernel.best_config}"', flush=True)
        print(f'flash_bwd_dq_kernel.best_config = "{flash_bwd_dq_kernel.best_config}"', flush=True)
        print(f'flash_bwd_dkdv_kernel.best_config = "{flash_bwd_dkdv_kernel.best_config}"', flush=True)
        print(timing_results, flush=True)
    dist.destroy_process_group()

@app.function(
    image=image,
    gpu="B200:2",
    timeout=60 * 120,
    volumes={"/traces": traces_vol},
    max_containers=3,
)
def run_leaderboard_distributed():
    import torch.multiprocessing as mp
    import os

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12390"

    num_gpu = 2
    with mp.Manager() as manager:
        results = manager.dict()
        mp.spawn(run_leaderboard, args=(num_gpu, results), nprocs=num_gpu, join=True)

        print(results[0])


def run_profile_target(warmup_steps: int, measurement_steps: int):
    import torch.multiprocessing as mp
    import os

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12391"

    num_gpu = 2
    with mp.Manager() as manager:
        results = manager.dict()
        mp.spawn(
            run_leaderboard,
            args=(num_gpu, results, True, warmup_steps, measurement_steps),
            nprocs=num_gpu,
            join=True,
        )
        if 0 in results:
            print(results[0], flush=True)


@app.function(
    image=image,
    gpu="B200:2",
    timeout=60 * 120,
    volumes={"/traces": traces_vol},
    max_containers=3,
)
def profile_leaderboard(trace_name: str = "leaderboard", warmup_steps: int = 1, measurement_steps: int = 1):
    import subprocess
    import sys

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
        sys.executable, "-m", "cs336_systems.leaderboard",
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


@app.local_entrypoint()
def main():
    run_leaderboard_distributed.remote()


@app.local_entrypoint()
def profile(trace_name: str = "leaderboard", warmup_steps: int = 1, measurement_steps: int = 1):
    print(profile_leaderboard.remote(trace_name, warmup_steps, measurement_steps))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-target", action="store_true")
    parser.add_argument("--warmup_steps", type=int, default=1)
    parser.add_argument("--measurement_steps", type=int, default=1)
    args = parser.parse_args()

    if args.profile_target:
        run_profile_target(args.warmup_steps, args.measurement_steps)
