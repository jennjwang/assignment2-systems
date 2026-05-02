"""
With optimizer state sharding and data parallel, we're able to split the optimizer state and activations
across our data-parallel axis. However, our model weights remain duplicated — we're storing a full copy
of them all on each GPU.
We can solve this by turning our data parallel (DP) axis into a fully-sharded data parallel axis (FSDP).
With FSDP, each GPU stores only its own slice of every weight tensor, but has to pull slices from other
GPUs to form the full weight tensor using an all-gather to prepare for a forward or backward pass.
To avoid keeping GPU compute waiting around for communication to finish, most FSDP implementations
schedule the layer's all-gather in advance of the operation, meaning the relevant weights are ready before
they are needed, preventing communication from blocking computation. This keeps weight sharding
communication off the critical path, meaning it has no cost as long as communication can keep up with
compute and is scheduled well.
Some layers are small enough in memory and compute that the latency overhead of a transfer is not
worth it. You should mark these layers not to be sharded by FSDP. In our architecture, this will mostly
be the case for norms. This leaves us with the embedding layer and every linear layer.
While it is necessary to store master weights in FP32 (any values that are repeatedly accumulated into
are sensitive to precision), the weights do not need to be used in FP32. In mixed precision, we always
convert to the low-precision compute datatype before use, so we may as well convert even before the
weight is communicated to save on bandwidth
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from cs336_basics.model import Embedding, Linear, LMHead, get_active_layer_range


class FSDP(nn.Module):
    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None, prefetch_distance: int = 2):
        """
        Given an instantiated PyTorch nn.Module to be parallelized, construct an FSDP module that
        will handle weight all-gathers and gradient reduce-scatters. Make sure that your hooks or your
        module wrappers all-gather the weights in time for the forward pass. To limit memory use,
        38
        only start gathering after the layer two before the current one has completed its forward pass.
        In the backward pass, your hooks or module wrappers should all-gather to have the weights
        available for the computation. When the gradients are available, they should be reduce-
        scattered to the appropriate ranks. Make sure to free the gathered weights after use. When
        compute_dtype is provided, cast the weights to that dtype before communicating or using them
        for compute, while keeping master weights and optimizer updates in FP32.
        """
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.compute_dtype = compute_dtype
        self.shapes = {}
        self.shards = {}
        self.original_dtypes = {}
        self.shardable_handles = []
        self.nonshardable_handles = []

        # only shard Linear layers inside transformer blocks
        self.shardable_mods = [
            m for m in module.modules()
            if isinstance(m, Linear) and not isinstance(m, LMHead)
        ]
        self.shardable_params = {id(m.weight) for m in self.shardable_mods}

        self.replicated_compute_mods = [
            m for m in module.modules()
            if isinstance(m, (Embedding, LMHead))
        ]
        self.replicated_compute_params = {id(m.weight) for m in self.replicated_compute_mods}
        self.replicated_masters = {}

        # given a layer, what's the NEXT shardable layer?
        self.next_mod = {
            self.shardable_mods[i]: self.shardable_mods[i + 1]
            for i in range(len(self.shardable_mods) - 1)
        }

        self.backward_mods = [m for m in self.shardable_mods if isinstance(m, Linear)]
        # given a layer in backward order, what's the PREVIOUS layer we should prefetch next?
        self.prev_backward_mod = {
            self.backward_mods[i + 1]: self.backward_mods[i]
            for i in range(len(self.backward_mods) - 1)
        }
        self.prefetch_distance = prefetch_distance
        # prefetch: maps module -> (handle, gathered_tensor, source_buffer) for forward prefetches.
        # backward_prefetch: same idea but for backward pass prefetches.
        self.prefetch = {}
        self.backward_prefetch = {}
        self._backward_active = set()

        # separate streams for forward and backward communication for prefetch
        self.forward_comm_stream = None
        self.backward_comm_stream = None
        if torch.cuda.is_available() and any(param.is_cuda for param in module.parameters()):
            self.forward_comm_stream = torch.cuda.Stream()
            self.backward_comm_stream = torch.cuda.Stream()

        for mod in self.shardable_mods:
            mod.register_forward_pre_hook(self._pre_forward)
            mod.register_forward_hook(self._post_forward)
            if isinstance(mod, Linear):
                mod.register_full_backward_pre_hook(self._pre_backward)
        for mod in self.replicated_compute_mods:
            mod.register_forward_pre_hook(self._replicated_pre_forward)

        for param in module.parameters():
            # broadcast from rank 0 so all ranks start with identical weights, then shard them.
            dist.broadcast(param.data, src=0)
            self.original_dtypes[id(param)] = param.dtype
            if id(param) in self.shardable_params:
                self._shard_param(param)
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._grad_hook)

    def _shard_param(self, param):
        param_id = id(param)
        self.shapes[param_id] = param.data.shape

        if param.data.ndim >= 2 and param.data.shape[0] % self.world_size == 0:
            rows_per_rank = param.data.shape[0] // self.world_size
            start = self.rank * rows_per_rank
            shard = param.data[start:start + rows_per_rank].clone()
        else:
            flat = param.data.view(-1)
            chunk = flat.numel() // self.world_size
            start = self.rank * chunk
            shard = flat[start:start + chunk].clone()

        self.shards[param_id] = shard
        param.data = shard

    def _do_gather(self, param):
        param_id = id(param)
        shard = self.shards[param_id]
        source = shard if (self.compute_dtype is None or shard.dtype == self.compute_dtype) else shard.to(self.compute_dtype)
        source = source.reshape(-1)
        gathered = torch.empty(self.world_size * source.numel(), device=source.device, dtype=source.dtype)
        dist.all_gather_into_tensor(gathered, source)
        return gathered.view(self.shapes[param_id])

    def _async_gather(self, module, stream):
        # non-blocking all-gather used by the prefetcher.
        # return (handle, gathered, source) so the caller can:
        #   - wait on handle when the weight is actually needed,
        #   - use gathered as the full weight tensor,
        #   - keep source alive (if we cast to compute_dtype, the cast buffer
        #     must not be garbage-collected before NCCL finishes reading it).
        param = module.weight
        param_id = id(param)
        shard = self.shards[param_id]
        source = shard if (self.compute_dtype is None or shard.dtype == self.compute_dtype) else shard.to(self.compute_dtype)
        source = source.reshape(-1)
        gathered = torch.empty(self.world_size * source.numel(), device=source.device, dtype=source.dtype)
        if stream is None:
            handle = dist.all_gather_into_tensor(gathered, source, async_op=True)
        else:
            # make the comm stream wait until the compute stream has written the shard,
            # then launch the collective on the comm stream
            stream.wait_stream(torch.cuda.current_stream(source.device))
            with torch.cuda.stream(stream):
                handle = dist.all_gather_into_tensor(gathered, source, async_op=True)
        return handle, gathered, source

    def _async_reduce_scatter(self, output, flat):
        if self.backward_comm_stream is None:
            return dist.reduce_scatter_tensor(output, flat, op=dist.ReduceOp.AVG, async_op=True)
        self.backward_comm_stream.wait_stream(torch.cuda.current_stream(flat.device))
        with torch.cuda.stream(self.backward_comm_stream):
            return dist.reduce_scatter_tensor(output, flat, op=dist.ReduceOp.AVG, async_op=True)

    def _async_all_reduce(self, grad):
        if self.backward_comm_stream is None:
            return dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True)
        self.backward_comm_stream.wait_stream(torch.cuda.current_stream(grad.device))
        with torch.cuda.stream(self.backward_comm_stream):
            return dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True)

    def _replicated_pre_forward(self, module, *args):
        if self.compute_dtype is None:
            return

        param = module.weight
        param_id = id(param)
        if param.data.dtype == self.compute_dtype:
            return
        self.replicated_masters[param_id] = param.data
        param.data = param.data.to(self.compute_dtype)

    def _pre_forward(self, module, *args):
        param = module.weight
        # already gathered
        if param.data.shape == self.shapes[id(param)]:
            return
        if module in self.prefetch:
            # an earlier layer's hook already launched the all-gather for this layer.
            handle, gathered, _source = self.prefetch.pop(module)
            handle.wait()
            param.data = gathered.view(self.shapes[id(param)])
        elif module in self.backward_prefetch:
            # backward prefetch may already hold this module's full weight.
            # reuse it to avoid allocating a second copy.
            handle, gathered, _source = self.backward_prefetch.pop(module)
            handle.wait()
            param.data = gathered.view(self.shapes[id(param)])
        else:
            param.data = self._do_gather(param)

        # kick off async gathers for the next `prefetch_distance` layers so their weights arrive in time.
        active_layer_range = get_active_layer_range()
        nxt = module
        for _ in range(self.prefetch_distance):
            nxt = self.next_mod.get(nxt)
            if nxt is None:
                break
            # only prefetch layers that will be recomputed
            if active_layer_range is not None:
                layer_idx = getattr(nxt, "transformer_layer_idx", None)
                if layer_idx is not None:
                    start, end = active_layer_range
                    if layer_idx < start or layer_idx >= end:
                        break
            if nxt not in self.prefetch:
                self.prefetch[nxt] = self._async_gather(nxt, self.forward_comm_stream)

    def _post_forward(self, module, input, output):
        param = module.weight
        if id(param) in self._backward_active:
            return
        param.data = self.shards[id(param)]

    def _pre_backward(self, module, *args):
        param = module.weight
        # mark this param as "actively in backward" so _post_forward won't free the weight
        self._backward_active.add(id(param))
        if module in self.backward_prefetch:
            # a later backward hook already prefetched this layer's weight.
            handle, gathered, _source = self.backward_prefetch.pop(module)
            handle.wait()
            param.data = gathered.view(self.shapes[id(param)])
        else:
            param.data = self._do_gather(param)

        # launch async gathers for the next `prefetch_distance` layers in backward order
        nxt = module
        for _ in range(self.prefetch_distance):
            nxt = self.prev_backward_mod.get(nxt)
            if nxt is None:
                break
            if nxt not in self.backward_prefetch:
                self.backward_prefetch[nxt] = self._async_gather(nxt, self.backward_comm_stream)

    def _grad_hook(self, param):
        # called by autograd after it accumulates a gradient
        param_id = id(param)
        self._backward_active.discard(param_id)
        comm_dtype = self.compute_dtype if self.compute_dtype is not None else param.grad.dtype

        if param_id in self.shardable_params:
            grad = param.grad if param.grad.dtype == comm_dtype else param.grad.to(comm_dtype)
            flat = grad.reshape(-1)
            chunk = flat.numel() // self.world_size
            output = torch.empty(chunk, device=flat.device, dtype=flat.dtype)
            res = self._async_reduce_scatter(output, flat)

            param.data = self.shards[param_id]
            param.grad = None 
            self.shardable_handles.append((res, param, output, flat))
        else:
            # full gradient must be averaged across all ranks via all-reduce
            if param.grad.dtype == comm_dtype:
                res = self._async_all_reduce(param.grad)
                cast_grad = param.grad if param_id in self.replicated_masters else None
                self.nonshardable_handles.append((res, param, cast_grad))
            else:
                cast_grad = param.grad.to(comm_dtype)
                res = self._async_all_reduce(cast_grad)
                self.nonshardable_handles.append((res, param, cast_grad))

            if param_id in self.replicated_masters:
                param.data = self.replicated_masters.pop(param_id)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def gather_full_params(self):
        state_dict = {}
        for name, param in self.module.named_parameters():
            if id(param) in self.shardable_params:
                shard = self.shards[id(param)]
                source = shard.reshape(-1)
                gathered = torch.empty(self.world_size * source.numel(), device=source.device, dtype=source.dtype)
                dist.all_gather_into_tensor(gathered, source)
                state_dict[name] = gathered.view(self.shapes[id(param)])
            else:
                data = param.data
                if id(param) in self.replicated_compute_params:
                    data = data.to(self.original_dtypes[id(param)])
                state_dict[name] = data
        return state_dict

    def finish_gradient_synchronization(self):
        """
        When called, wait for asynchronous communication calls to finish on the GPU.
        """
        for res, param, output, _flat in self.shardable_handles:
            res.wait()
            grad = output.to(param.dtype) if output.dtype != param.dtype else output
            param.grad = grad.view_as(param.data)
        for res, param, cast_grad in self.nonshardable_handles:
            res.wait()
            if cast_grad is not None:
                param.grad = cast_grad.to(param.dtype) if cast_grad.dtype != param.dtype else cast_grad
        self.shardable_handles.clear()
        self.nonshardable_handles.clear()
