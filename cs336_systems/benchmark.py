"""
Problem (benchmarking_script): Benchmarking Script (4 points)
(a) Write a script to perform basic end-to-end benchmarking of the forward pass, backward pass,
and optimizer step in your model. Specifically, your script should support the following:
• Given hyperparameters (e.g., number of layers), initialize a model.
• Generate a random batch of data.
• Run 𝑤 warm-up steps (before you start measuring time), then time the execution of 𝑛
steps (either only forward, forward and backward, or forward and backward with optimizer
step, depending on an argument). For timing, you can use the Python timeit module (e.g.,
either using the timeit function, or using timeit.default_timer(), which gives you the
system’s highest resolution clock, thus a better default for benchmarking than
time.time()).
3
• Call torch.cuda.synchronize() after each step.
Deliverable: A script that will initialize a basics Transformer model with the given
hyperparameters, create a random batch of data, and time forward-only, forward-and-
backward, and full training steps that include the optimizer step.
(b) Time the forward, backward, and optimizer step for the model sizes described in
Section 2.1.2. Use 5 warmup steps and compute the average and standard deviation of
timings over 10 measurement steps. How long does a forward pass take? How about a
backward pass? Do you see high variability across measurements, or is the standard deviation
small?
Deliverable: A 1-2 sentence response with your timings.
(c) One caveat of benchmarking is not performing the warm-up steps. Repeat your analysis
without the warm-up steps. How does this affect your results? Why do you think this
happens? Also try to run the script with 1 or 2 warm-up steps. Why might the result still be
different?
Deliverable: A 2-3 sentence response

"""


import timeit
import torch
# import modal
import argparse
import torch.cuda.nvtx as nvtx
from contextlib import nullcontext
from pathlib import Path
import json

import modal


@nvtx.range("profile_benchmark")
def benchmark(batch_size, context_length, vocab_size, device, d_model, num_layers, num_heads, d_ff, rope_theta, warmup_steps, measurement_steps, mode, mixed_precision=False, use_compile=True):

    from einops import rearrange
    from cs336_basics.model import BasicsTransformerLM
    from cs336_basics.nn_utils import cross_entropy, clip_gradient
    from cs336_basics.optimizer import AdamW
    import numpy as np

    torch.set_float32_matmul_precision('high')

    times = []
    model = BasicsTransformerLM(vocab_size=vocab_size, context_length=context_length, d_model=d_model, num_layers=num_layers, num_heads=num_heads, d_ff=d_ff, rope_theta=rope_theta)
    model.to(device)
    if use_compile:
        model = torch.compile(model)
    optimizer = AdamW(model.parameters(), lr=1e-3)

    data = torch.randint(0, vocab_size, (batch_size, context_length))
    labels = torch.randint(0, vocab_size, (batch_size, context_length))
    data = data.to(device)
    labels = labels.to(device)

    device_type = device.split(":")[0]
    ctx = (lambda: torch.autocast(device_type=device_type, dtype=torch.bfloat16)) if mixed_precision else nullcontext

    if mode == "forward_only":

        for i in range(warmup_steps):
            with ctx():
                model(data)
            torch.cuda.synchronize()

        # torch.cuda.memory._record_memory_history(max_entries=1000000)

        with nvtx.range("forward_only"):
            for i in range(measurement_steps):
                start_time = timeit.default_timer()
                with ctx():
                    model(data)
                torch.cuda.synchronize()
                end_time = timeit.default_timer()
                times.append(end_time - start_time)
            mean_time = float(np.mean(times))
            std_time = float(np.std(times))
            print(f"Forward pass time: {mean_time} seconds ± {std_time} seconds")

        # torch.cuda.memory._dump_snapshot(f"/traces/memory_snapshot_{mode}_ctx{context_length}{'_mp' if mixed_precision else ''}.pickle")
        # torch.cuda.memory._record_memory_history(enabled=None)

    elif mode == "forward_backward":

        for _ in range(warmup_steps):
            with ctx():
                logits = model(data)
                logits = rearrange(logits, "batch_size context_len vocab -> (batch_size context_len) vocab")
                labels_flat = rearrange(labels, "batch_size context_len -> (batch_size context_len)")
                loss = cross_entropy(logits, labels_flat)
            optimizer.zero_grad()
            loss.backward()
            torch.cuda.synchronize()

        with nvtx.range("forward_backward"):
            # with torch.autograd.profiler.emit_nvtx():
            with nullcontext():
                for _ in range(measurement_steps):
                    start_time = timeit.default_timer()
                    with nvtx.range("forward"):
                        with ctx():
                            logits = model(data)
                            logits = rearrange(logits, "batch_size context_len vocab -> (batch_size context_len) vocab")
                            labels_flat = rearrange(labels, "batch_size context_len -> (batch_size context_len)")
                            loss = cross_entropy(logits, labels_flat)
                    with nvtx.range("backward"):
                        optimizer.zero_grad()
                        loss.backward()
                    torch.cuda.synchronize()
                    end_time = timeit.default_timer()
                    times.append(end_time - start_time)
                mean_time = float(np.mean(times))
                std_time = float(np.std(times))
                print(f"Forward and backward pass time: {mean_time} seconds ± {std_time} seconds")

    elif mode == "full_training":
        for i in range(warmup_steps):
            with ctx():
                logits = model(data)
                logits = rearrange(logits, "batch_size context_len vocab -> (batch_size context_len) vocab")
                labels_flat = rearrange(labels, "batch_size context_len -> (batch_size context_len)")
                loss = cross_entropy(logits, labels_flat)
            optimizer.zero_grad()
            loss.backward()
            clip_gradient(model.parameters(), 1.0)
            optimizer.step()
            torch.cuda.synchronize()

        # torch.cuda.memory._record_memory_history(max_entries=1000000)

        with nvtx.range("full_training"):
            # with torch.autograd.profiler.emit_nvtx():
            with nullcontext():
                for _ in range(measurement_steps):
                    start_time = timeit.default_timer()
                    with nvtx.range("forward"):
                        with ctx():
                            logits = model(data)
                            logits = rearrange(logits, "batch_size context_len vocab -> (batch_size context_len) vocab")
                            labels_flat = rearrange(labels, "batch_size context_len -> (batch_size context_len)")
                            loss = cross_entropy(logits, labels_flat)
                    with nvtx.range("backward"):
                        optimizer.zero_grad()
                        loss.backward()
                    with nvtx.range("optimizer"):
                        clip_gradient(model.parameters(), 1.0)
                        optimizer.step()
                    torch.cuda.synchronize()
                    end_time = timeit.default_timer()
                    times.append(end_time - start_time)
                mean_time = float(np.mean(times))
                std_time = float(np.std(times))
                print(f"Full training time: {mean_time} seconds ± {std_time} seconds")
    
        # Save a pickle file to be loaded by PyTorch's online tool.
        # torch.cuda.memory._dump_snapshot(f"/traces/memory_snapshot_{mode}_ctx{context_length}{'_mp' if mixed_precision else ''}.pickle")

        # Stop recording history.
        # torch.cuda.memory._record_memory_history(enabled=None)

    return {
        "mode": mode,
        "compiled": bool(use_compile),
        "mixed_precision": bool(mixed_precision),
        "batch_size": batch_size,
        "context_length": context_length,
        "vocab_size": vocab_size,
        "device": device,
        "d_model": d_model,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "d_ff": d_ff,
        "rope_theta": rope_theta,
        "warmup_steps": warmup_steps,
        "measurement_steps": measurement_steps,
        "mean_seconds": float(mean_time),
        "std_seconds": float(std_time),
        "times_seconds": [float(t) for t in times],
    }

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
def profile_one(
    config_path: str,
    mixed_precision: bool = False,
    overrides: dict | None = None,
    save_json_dir: str | None = None,
    use_compile: bool = True,
):
    from pathlib import Path

    import yaml
    name = Path(config_path).stem
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg.update(overrides or {})

    ctx = cfg.get("context_length", "unknown")
    mode = cfg.get("mode", "full_training")
    mp_suffix = "_mp" if mixed_precision else ""
    compile_suffix = "compiled" if use_compile else "uncompiled"
    result_json_path = None
    if save_json_dir is not None:
        result_json_path = f"{save_json_dir}/{name}_ctx{ctx}_{mode}{mp_suffix}_{compile_suffix}.json"

    # os.environ["PYTORCH_ALLOC_CONF"] = "backend:cudaMallocAsync"
    # os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

    # subprocess.run(
    #     [
    #         "nsys", "profile",
    #         "--trace=cuda,cudnn,cublas,nvtx",
    #         "--cuda-memory-usage=true",
    #         "--gpu-metrics-devices=0",
    #         "--force-overwrite=true",
    #         "-o", f"/traces/{name}_ctx{ctx}_{mode}{mp_suffix}",
    #         sys.executable, "-m", "cs336_systems.benchmark",
    #         "--configs", tmp.name,
    #     ],
    #     env=env,
    #     check=False,
    # )
    # subprocess.run(["nsys", "stats", f"{trace_path}.nsys-rep"], check=False)
    # traces_vol.commit()
    result = benchmark(
        batch_size=cfg["batch_size"],
        context_length=cfg["context_length"],
        vocab_size=cfg["vocab_size"],
        device=cfg.get("device", "cuda"),
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        rope_theta=cfg.get("rope_theta", 10000.0),
        warmup_steps=cfg.get("warmup_steps", 5),
        measurement_steps=cfg.get("measurement_steps", 10),
        mode=mode,
        mixed_precision=mixed_precision,
        use_compile=use_compile,
    )
    result["model"] = name

    if result_json_path is not None:
        out_path = Path(result_json_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        vol1.commit()

    return result

# @app.function(
#     image=image,
#     gpu="B200",
#     timeout=60 * 120,
#     volumes={"/data": vol1},
#     max_containers=8,
# )
# def run_benchmark(config_path: str, mixed_precision: bool = False):
#     import sys
#     import yaml
#     sys.path.insert(0, "/root/project")
#     with open(config_path) as f:
#         cfg = yaml.safe_load(f)
#     benchmark(
#         batch_size=cfg["batch_size"],
#         context_length=cfg["context_length"],
#         vocab_size=cfg["vocab_size"],
#         device="cuda",
#         d_model=cfg["d_model"],
#         num_layers=cfg["num_layers"],
#         num_heads=cfg["num_heads"],
#         d_ff=cfg["d_ff"],
#         rope_theta=cfg.get("rope_theta", 10000.0),
#         warmup_steps=cfg.get("warmup_steps", 5),
#         measurement_steps=cfg.get("measurement_steps", 10),
#         mode=cfg.get("mode", "forward_backward"),
#         mixed_precision=mixed_precision,
#     )


configs = [
    "/root/project/cs336_systems/configs/small.yaml",
    "/root/project/cs336_systems/configs/medium.yaml",
    "/root/project/cs336_systems/configs/large.yaml",
    "/root/project/cs336_systems/configs/xl.yaml",
    "/root/project/cs336_systems/configs/10B.yaml",
]

@app.local_entrypoint()
def main(
    mixed_precision: bool = False,
    use_compile: bool = True,
    compare_compile: bool = False,
    modes: str = "forward_only,forward_backward,full_training",
    warmup_steps: int = 1,
    measurement_steps: int = 10,
    include_no_warmup: bool = False,
):
    selected_modes = [mode.strip() for mode in modes.split(",") if mode.strip()]
    warmup_values = [warmup_steps]
    if include_no_warmup and 0 not in warmup_values:
        warmup_values.append(0)
    compile_values = [False, True] if compare_compile else [use_compile]

    spawns = [
        (
            cfg,
            mode,
            warmup,
            compile_value,
            profile_one.spawn(
                cfg,
                mixed_precision=mixed_precision,
                overrides={
                    "mode": mode,
                    "warmup_steps": warmup,
                    "measurement_steps": measurement_steps,
                },
                save_json_dir=None,
                use_compile=compile_value,
            ),
        )
        for compile_value in compile_values
        for warmup in warmup_values
        for mode in selected_modes
        for cfg in configs
    ]
    results = []
    for cfg, mode, warmup, compile_value, spawn in spawns:
        model = Path(cfg).stem
        compile_label = "compiled" if compile_value else "uncompiled"
        try:
            result = spawn.get()
            results.append(result)
            mean = result["mean_seconds"]
            std = result["std_seconds"]
            print(f"{model:>6} | {compile_label:<10} | warmup={warmup:<2} | {mode:<16} | {mean:.6f} ± {std:.6f}s")
        except Exception as exc:
            message = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
            print(f"{model:>6} | {compile_label:<10} | warmup={warmup:<2} | {mode:<16} | ERROR: {message}")

    if compare_compile and results:
        import pandas as pd

        df = pd.DataFrame(results)
        df.to_csv("benchmark_transformer_compile.csv", index=False)
        print(df)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", type=str, default=None)
    parser.add_argument("--configs", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--measurement_steps", type=int, default=10)
    parser.add_argument("--mode", type=str, default="forward_only")
    parser.add_argument("--mixed_precision", action="store_true", default=False)
    parser.add_argument("--no_compile", action="store_true", default=False)
    parser.add_argument("--save_json", type=str, default=None)
    args = parser.parse_args()

    if args.stats is not None:
        import subprocess
        subprocess.run(["nsys", "stats", args.stats], check=True)
        exit(0)

    if args.configs is not None:
        import yaml
        with open(args.configs) as f:
            cfg = yaml.safe_load(f)
        for key, val in cfg.items():
            setattr(args, key, val)

    result = benchmark(args.batch_size, args.context_length, args.vocab_size, args.device,
                       args.d_model, args.num_layers, args.num_heads, args.d_ff,
                       args.rope_theta, args.warmup_steps, args.measurement_steps, args.mode,
                       args.mixed_precision, use_compile=(not args.no_compile))

    if args.save_json is not None:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"Saved benchmark results to {out_path}")
