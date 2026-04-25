import torch
import modal
from torch.utils.checkpoint import checkpoint

app = modal.App("systems-jwang400")
vol1 = modal.Volume.from_name("baseline", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("torch~=2.11.0", "numpy", "regex", "einops", "tqdm")
    .add_local_dir("cs336-basics", "/root/cs336-basics", copy=True)
    .add_local_dir(".", "/root/cs336-systems", copy=True,
                    ignore=[".venv", ".git", "__pycache__", "*.pyc"])
    .run_commands("pip install /root/cs336-basics", "pip install /root/cs336-systems")
)


def grouped_checkpoint_forward(layers, x, group_size):
    for i in range(0, len(layers), group_size):
        group = layers[i:i + group_size]
        def run_group(x, group=group):
            for layer in group:
                x = layer(x)
            return x
        x = checkpoint(run_group, x, use_reentrant=False)
    return x


@app.function(image=image, gpu="B200", timeout=60 * 60, volumes={"/data": vol1})
def profile_checkpointing(group_sizes: list[int], batch_size: int = 4, context_length: int = 2048):
    from einops import rearrange
    from cs336_basics.model import BasicsTransformerLM
    from cs336_basics.nn_utils import cross_entropy
    from cs336_basics.optimizer import AdamW

    device = "cuda"
    vocab_size, d_model, num_layers, num_heads, d_ff = 10000, 2560, 32, 32, 10240

    for group_size in group_sizes:
        model = BasicsTransformerLM(
            vocab_size=vocab_size, context_length=context_length,
            d_model=d_model, num_layers=num_layers, num_heads=num_heads,
            d_ff=d_ff, rope_theta=10000.0,
        ).to(device)
        optimizer = AdamW(model.parameters(), lr=1e-3)
        data = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
        labels = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
        layers = list(model.layers)

        def forward(x):
            x = model.token_embeddings(x)
            x = grouped_checkpoint_forward(layers, x, group_size)
            x = model.ln_final(x)
            return model.lm_head(x)

        for _ in range(2):  # warmup
            optimizer.zero_grad()
            logits = rearrange(forward(data), "b s v -> (b s) v")
            cross_entropy(logits, rearrange(labels, "b s -> (b s)")).backward()
            optimizer.step()
            torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad()
        logits = rearrange(forward(data), "b s v -> (b s) v")
        cross_entropy(logits, rearrange(labels, "b s -> (b s)")).backward()
        torch.cuda.synchronize()

        peak = torch.cuda.max_memory_allocated() / 1e9
        num_groups = num_layers // group_size
        print(f"group_size={group_size:3d} num_groups={num_groups:2d}: peak = {peak:.2f} GB")

        del model, optimizer
        torch.cuda.empty_cache()


@app.local_entrypoint()
def main():
    # Compare: too small, near-optimal (~sqrt(32)≈6), too large
    profile_checkpointing.remote([1, 2, 4, 6])
