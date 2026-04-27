"""
Deliverable: Implement a container class to handle distributed data parallel training. This class
should overlap gradient communication and the computation of the backward pass. To test your
DDP class, first implement the adapters [adapters.get_ddp_individual_parameters] and
[adapters.ddp_individual_parameters_on_after_backward] (the latter is optional,
depending on your implementation you may not need it)
"""

import torch
from torch.nn import Module
import torch.multiprocessing as mp
import torch.distributed as dist

"""
Backward pass starts
PyTorch computes gradient for parameter A
Your hook fires → you call dist.all_reduce(A.grad, async_op=True) → you get back a handle → you append it to self.async_handles
PyTorch computes gradient for parameter B
Your hook fires again → same thing for B
... (for all parameters)
Backward pass ends
finish_gradient_synchronization is called → you wait on all handles → you average the gradients
"""

class DDPOverlap(Module):
    def __init__(self, module: Module):
        super().__init__()
        self.module = module
        
        '''
        Ensure all ranks start with the same weights (broadcast rank 0's parameters to everyone)
        For each parameter that requires grad, register a hook that fires when that parameter's gradient is ready — 
        the hook should launch an async all-reduce and store the returned handle somewhere on self
        '''
        self.async_handles = []   

        for param in module.parameters():
            dist.broadcast(param.data, src=0)
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._hook)

    def _hook(self, param):
        res = dist.all_reduce(param.grad, async_op=True)
        self.async_handles.append(res)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.async_handles:
            handle.wait()
        
        # dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
        for param in self.module.parameters():
            if param.grad is not None:
                param.grad.div_(world_size)
        
        self.async_handles = []

        

