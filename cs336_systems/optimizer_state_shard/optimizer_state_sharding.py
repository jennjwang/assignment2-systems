import torch
import torch.distributed as dist
from typing import Any, Type
from torch.optim import Optimizer

"""
Implement a Python class to handle optimizer state sharding. The class should wrap an arbitrary
input PyTorch optim.Optimizer and take care of synchronizing updated parameters after each
optimizer step. We recommend the following public interface:
"""

class OptimizerStateShard(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        """
        Initializes the sharded state optimizer. params is a collection of parameters to be optimized (or parameter
        groups, in case the user wants to use different hyperparameters, such as learning rates, for
        different parts of the model); these parameters will be sharded across all the ranks. The
        optimizer_cls parameter specifies the type of optimizer to be wrapped (e.g., optim.AdamW).
        Finally, any remaining keyword arguments are forwarded to the constructor of the
        optimizer_cls. Make sure to call the torch.optim.Optimizer super-class constructor in this
        method.
        """
        self._optimizer_cls = optimizer_cls
        self._optimizer_kwargs = kwargs
        self.optimizer = None
        self.all_params = []
        self.param_index = 0
        super().__init__(params, kwargs)

    def step(self, closure=None, **kwargs):
        """
        Calls the wrapped optimizer’s step() method with the provided closure and keyword arguments.
        After updating the parameters, synchronize with the other ranks.
        """
        if self.optimizer is not None:
            self.optimizer.step(closure, **kwargs)
        for (param, rank) in self.all_params:
            dist.broadcast(param.data, src=rank)


    def add_param_group(self, param_group: dict[str, Any]):
        """
        This method should add a parameter
        group to the sharded optimizer. This is called during construction of the sharded optimizer by
        the super-class constructor and may also be called during training (e.g., for gradually
        unfreezing layers in a model). As a result, this method should handle assigning the model’s
        parameters among the ranks.
        """
        super().add_param_group(param_group)
        world_size = dist.get_world_size()
        for p in param_group['params']:
            rank = self.param_index % world_size
            if rank == dist.get_rank():
                shard_group = {**param_group, 'params': [p]}
                if self.optimizer is None:
                    self.optimizer = self._optimizer_cls([shard_group], **self._optimizer_kwargs)
                else:
                    self.optimizer.add_param_group(shard_group)
            self.all_params.append((p, rank))
            self.param_index += 1
    
