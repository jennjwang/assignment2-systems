"""
Write a script to naively perform distributed data parallel training by all-reducing
individual parameter gradients after the backward pass. To verify the correctness of your DDP
implementation, use it to train a small toy model on randomly-generated data and verify that its
weights match the results from single-process training.
"""

from copy import deepcopy

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
import os

from tests.common import ToyModel

def naive_dpp_shard(rank, model, data, num_gpu, num_samples, labels, loss_fn, optimizer_class):

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", world_size=num_gpu, rank=rank)

    data_i = data[rank * num_samples : (rank + 1) * num_samples].to(f"cuda:{rank}")
    labels_i = labels[rank * num_samples : (rank + 1) * num_samples].to(f"cuda:{rank}")

    # each device runs forward pass and backward pass
    model = model.to(f"cuda:{rank}")
    optimizer = optimizer_class(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    outputs = model(data_i)
    loss = loss_fn(outputs, labels_i)
    loss.backward()

    # all-reduce the gradients
    for param in model.parameters():
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(num_gpu)

    # run optimizer step
    optimizer.step()
    dist.destroy_process_group()


def naive_dpp(model, data, num_gpu, labels, loss_fn, optimizer_class):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12390"
    # given a batch with n examples, the batch is sharded and each device receives n/d disjoint examples
    num_samples = data.shape[0] // num_gpu
    mp.spawn(naive_dpp_shard, args=(model, data, num_gpu, num_samples, labels, loss_fn, optimizer_class), nprocs=num_gpu, join=True)


def _test_naive_dpp_inner(rank, world_size, model_state, data, labels):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12391"
    dist.init_process_group(backend="gloo", world_size=world_size, rank=rank)

    loss_fn = nn.MSELoss()

    model = ToyModel().double()
    model.load_state_dict(model_state)

    n = data.shape[0] // world_size
    data_i = data[rank * n : (rank + 1) * n]
    labels_i = labels[rank * n : (rank + 1) * n]

    ddp_model = deepcopy(model)
    ddp_opt = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
    ddp_opt.zero_grad()
    ddp_loss = loss_fn(ddp_model(data_i), labels_i)
    ddp_loss.backward()

    for param in ddp_model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(world_size)

    ddp_opt.step()

    if rank == 0:
        single_model = deepcopy(model)
        single_opt = torch.optim.SGD(single_model.parameters(), lr=0.01)
        single_opt.zero_grad()
        single_loss = loss_fn(single_model(data), labels)
        single_loss.backward()
        single_opt.step()

        for (k, sp), (_, dp) in zip(single_model.named_parameters(), ddp_model.named_parameters()):
            assert torch.allclose(sp, dp, atol=1e-10), (
                f"Parameter '{k}' mismatch: max diff = {(sp - dp).abs().max():.2e}\n"
                f"  single={sp}\n  ddp={dp}"
            )

    dist.barrier()
    dist.destroy_process_group()


def test_naive_dpp():
    world_size = 2
    torch.manual_seed(42)

    model = ToyModel().double()
    data = torch.randn(100, 10, dtype=torch.float64)   # ToyModel input dim: 10
    labels = torch.randn(100, 5, dtype=torch.float64)  # ToyModel output dim: 5

    model_state = {k: v.clone() for k, v in model.state_dict().items()}

    mp.spawn(
        _test_naive_dpp_inner,
        args=(world_size, model_state, data, labels),
        nprocs=world_size,
        join=True,
    )
    print("test_naive_dpp passed!")


if __name__ == "__main__":
    test_naive_dpp()
