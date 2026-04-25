'''
Write a pure PyTorch (no Triton) autograd.Function that implements the FlashAttention-2
forward pass. This will be a lot slower than the regular PyTorch implementation, but will
help you debug your Triton kernel.
Your implementation should take input 𝑸, 𝑲, and 𝑽 as well as a flag is_causal and produce
the output 𝑶 and the logsumexp value 𝐿. You can ignore the is_causal flag for this task.
The autograd.Function forward should then save 𝐿, 𝑄, 𝐾, 𝑉 , 𝑂 for the backward pass and
return 𝑂. Remember that the implementation of the forward method of autograd.Function
always takes the context as its first parameter. Any autograd.Function class needs to
implement a backward method, but for now you can make it just raise NotImplementedError. If
you need something to compare against, you can implement Equation 4 to Equation 6 and
Equation 12 in PyTorch and compare your outputs.
The interface is then def forward(ctx, Q, K, V, is_causal=False). Determine your own tile
sizes, but make sure they are at least of size 16 × 16. We will always test your code with
dimensions that are powers of 2 and at least 16, so you don’t need to worry about out-of-
bounds accesses.
Deliverable: A torch.autograd.Function subclass that implements FlashAttention-2 in the
forward pass. To test your code, implement
[adapters.get_flashattention_autograd_function_pytorch] . Then, run the test with
uv run pytest -k test_flash_forward_pass_pytorch and make sure your implementation
passes it.
'''

import torch
from torch.autograd import Function
from einops import einsum 

class FlashAttention2Function(Function):
    
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B_q = 16
        B_k = 16
        T_q = Q.shape[-2] // B_q
        T_k = K.shape[-2] // B_k
        d = Q.shape[-1]
        O = torch.zeros(Q.shape, device=Q.device, dtype=Q.dtype)
        L = torch.zeros((*Q.shape[:-1],), device=Q.device, dtype=Q.dtype)
        for i in range(T_q):
            # load q
            # split Q into tiles
            Q_tile = Q[..., i*B_q: (i+1)*B_q, :]
            # initialize o
            O_i = torch.zeros(*Q.shape[:-2], B_q, d, device=Q.device, dtype=Q.dtype)
            l_i = torch.zeros((*Q.shape[:-2], B_q), device=Q.device, dtype=Q.dtype)
            m_i = torch.full((*Q.shape[:-2], B_q), -torch.inf, device=Q.device, dtype=Q.dtype)
            for j in range(T_k):
                # load k and v
                # split K, V into tiles
                K_tile = K[..., j*B_k: (j+1)*B_k, :]
                V_tile = V[..., j*B_k: (j+1)*B_k, :]
                # compute tile of pre-softmax attn scores
                S = einsum(Q_tile, K_tile, "batch B_q d, batch B_k d -> batch B_q B_k")
                S = S / (d ** 0.5)
                # compute m
                m_ij = S.amax(dim=-1)
                m_i_new = torch.maximum(m_i, m_ij)
                # compute p
                P = torch.exp(S - m_i_new.unsqueeze(-1))
                # compute l
                l_i = torch.exp(m_i - m_i_new) * l_i + P.sum(dim=-1)
                # compute o
                O_attn = einsum(P, V_tile, "batch B_q B_k, batch B_k d -> batch B_q d")
                O_i = torch.exp(m_i - m_i_new).unsqueeze(-1) * O_i + O_attn
                m_i = m_i_new
            # compute o
            O_i = O_i / l_i.unsqueeze(-1)
            # compute l
            l_i = m_i_new + torch.log(l_i)
            # write O
            O[..., i*B_q: (i+1)*B_q, :] = O_i 
            # write l
            L[..., i*B_q: (i+1)*B_q] = l_i
        ctx.save_for_backward(L, Q, K, V, O)
        return O

    @staticmethod
    def backward(ctx, dO):
        raise NotImplementedError