"""
Write a Triton kernel for the forward pass of FlashAttention-2 following Algorithm 1. Then,
write another subclass of torch.autograd.Function that calls this (fused) kernel in the
forward pass, instead of computing the result in PyTorch. A few problem-specific tips:
• To debug, we suggest comparing the results of each Triton operation you perform with the
tiled PyTorch implementation you wrote in part (a).
• Your launch grid should be set as (𝑇𝑞, batch_size), meaning each Triton program instance
will load only elements from a single batch index, and only read/write to a single query
tile of 𝑸, 𝑶, and 𝐿.
• The kernel should only have a single loop, which will iterate key tiles 1 ≤ 𝑗 ≤ 𝑇𝑘.
• Advance block pointers at the end of the loop.
• Use the function declaration below (using the block pointer we give you, you should be
able to infer the setup of the rest of the pointers):
"""

import triton
import triton.language as tl
from torch.autograd import Function
import torch
from cs336_systems.flash_forward import _flash_backward_compiled

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
    input_dtype: tl.constexpr,
    ):

    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
    Q_ptr + batch_index * stride_qb,
    shape=(N_QUERIES, D),
    strides=(stride_qq, stride_qd),
    offsets=(query_tile_index * Q_TILE_SIZE, 0),
    block_shape=(Q_TILE_SIZE, D),
    order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
    K_ptr + batch_index * stride_kb,
    shape=(N_KEYS, D),
    strides=(stride_kk, stride_kd),
    offsets=(0, 0),
    block_shape=(K_TILE_SIZE, D),
    order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
    V_ptr + batch_index * stride_vb,
    shape=(N_KEYS, D),
    strides=(stride_vk, stride_vd),
    offsets=(0, 0),
    block_shape=(K_TILE_SIZE, D),
    order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
    O_ptr + batch_index * stride_ob,
    shape=(N_QUERIES, D),
    strides=(stride_oq, stride_od),
    offsets=(query_tile_index * Q_TILE_SIZE, 0),
    block_shape=(Q_TILE_SIZE, D),
    order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
    L_ptr + batch_index * stride_lb,
    shape=(N_QUERIES,),
    strides=(stride_lq,),
    offsets=(query_tile_index * Q_TILE_SIZE,),
    block_shape=(Q_TILE_SIZE,),
    order=(0,),
    )

    """
    Load Q tile: Q = tl.load(Q_block_ptr) — shape (Q_TILE_SIZE, D)
    Initialize O_i, l_i, m_i — in Triton you use tl.zeros and tl.full
    Inner loop over T_k = N_KEYS // K_TILE_SIZE key tiles
    Advance K and V block pointers at the end of each iteration with tl.advance
    After loop: normalize, compute L, store O and L
    """

    Q = tl.load(Q_block_ptr).to(tl.float32)
    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K = tl.load(K_block_ptr).to(tl.float32)
        V = tl.load(V_block_ptr).to(tl.float32)

        S = tl.dot(Q, tl.trans(K)) * scale
        if is_causal:
            Q_tile_index = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            K_tile_index = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            mask = Q_tile_index[:, None] >= K_tile_index[None, :]
            S = tl.where(mask, S, S + -1e6)

        m_ij = tl.max(S, axis=-1)
        m_i_new = tl.maximum(m_i, m_ij)

        P = tl.exp(S - m_i_new[:, None])
        
        l_i = tl.exp(m_i - m_i_new) * l_i + tl.sum(P, axis=1)
        O_i = tl.exp(m_i - m_i_new)[:, None] * O_i + tl.dot(P, V)
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
        m_i = m_i_new
    
    O_i = O_i / l_i[:, None]
    L_i = m_i + tl.log(l_i)
    tl.store(O_block_ptr, O_i.to(input_dtype))
    tl.store(L_block_ptr, L_i.to(input_dtype))

class TritonFlashAttentionAutograd(Function):

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False, B_q=16, B_k=16):
        N_q = Q.shape[-2]
        N_k = K.shape[-2]
        T_q = N_q // B_q
        T_k = N_k // B_k
        d = Q.shape[-1]
        O = torch.zeros_like(Q)
        L = torch.zeros(*Q.shape[:-1], device=Q.device, dtype=Q.dtype)
        grid = (T_q, Q.shape[0], B_q, B_k)
        dtype_map = {torch.float32: tl.float32, torch.bfloat16: tl.bfloat16}
        input_dtype = dtype_map[Q.dtype]

    # Q_ptr, K_ptr, V_ptr,
    # O_ptr, L_ptr,
    # stride_qb, stride_qq, stride_qd,
    # stride_kb, stride_kk, stride_kd,
    # stride_vb, stride_vk, stride_vd,
    # stride_ob, stride_oq, stride_od,
    # stride_lb, stride_lq,
    # N_QUERIES, N_KEYS,
    # scale,
    # D: tl.constexpr,
    # Q_TILE_SIZE: tl.constexpr,
    # K_TILE_SIZE: tl.constexpr,
        flash_fwd_kernel[grid](Q, K, V, O, L,
                            Q.stride(0), Q.stride(1), Q.stride(2),
                            K.stride(0), K.stride(1), K.stride(2),
                            V.stride(0), V.stride(1), V.stride(2),
                            O.stride(0), O.stride(1), O.stride(2),
                            L.stride(0), L.stride(1), 
                            N_q, N_k,
                            1 / (d ** 0.5),
                            d,
                            B_q, B_k,
                            is_causal,
                            input_dtype)

        ctx.is_causal = is_causal
        ctx.save_for_backward(L, Q, K, V, O)
        return O
    
    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        dQ, dK, dV = _flash_backward_compiled(Q, K, V, O, dO, L)
        return dQ, dK, dV, None, None, None


