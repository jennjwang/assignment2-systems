'''
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
able to infer the setup of the rest of the pointers)
'''

import torch
import triton
import triton.language as tl
from torch.autograd import Function

dtype_mapping = {torch.float32: tl.float32, torch.bfloat16: tl.bfloat16}
_bwd_streams = None


def _get_bwd_streams(device):
    global _bwd_streams
    if _bwd_streams is None or _bwd_streams[0].device != device:
        _bwd_streams = (torch.cuda.Stream(device=device), torch.cuda.Stream(device=device))
    return _bwd_streams


# @triton.autotune(
#     configs=[
#         triton.Config({'Q_TILE_SIZE': 64,  'K_TILE_SIZE': 64},  num_warps=4, num_stages=1),
#         triton.Config({'Q_TILE_SIZE': 64,  'K_TILE_SIZE': 128}, num_warps=4, num_stages=1),
#         triton.Config({'Q_TILE_SIZE': 128, 'K_TILE_SIZE': 64},  num_warps=8, num_stages=1),
#         triton.Config({'Q_TILE_SIZE': 128, 'K_TILE_SIZE': 128}, num_warps=8, num_stages=1),
#         triton.Config({'Q_TILE_SIZE': 64,  'K_TILE_SIZE': 64},  num_warps=4, num_stages=2),
#         triton.Config({'Q_TILE_SIZE': 64,  'K_TILE_SIZE': 128}, num_warps=4, num_stages=2),
#         triton.Config({'Q_TILE_SIZE': 128, 'K_TILE_SIZE': 64},  num_warps=8, num_stages=2),
#         triton.Config({'Q_TILE_SIZE': 128, 'K_TILE_SIZE': 128}, num_warps=8, num_stages=2),
#     ],
#     key=['N_QUERIES', 'N_KEYS', 'D', 'is_causal'],
# )
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
    # program_id(0) = which Q tile we own, program_id(1) = which batch element.
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

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

    # for i = 1, ..., T_q (parallelized — each program owns one Q tile i)
    # load Q_i from global memory
    Q = tl.load(Q_block_ptr)
    # initialize O_i^(0) = 0, l_i^(0) = 0, m_i^(0) = -inf
    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)

    # three regions:
    #   [0, T_k_full): all keys in this tile are past (query_idx >= key_idx always), no masking needed
    #   [T_k_full, T_k_causal): diagonal tiles where some keys are future, need masking
    #   [T_k_causal, T_k): all keys are future — skip entirely
    T_k = tl.cdiv(N_KEYS, K_TILE_SIZE)
    T_k_full = T_k
    T_k_causal = T_k
    # For non-causal attention, T_k_full = T_k_causal = T_k
    if is_causal:
        query_start = query_tile_index * Q_TILE_SIZE
        query_end = tl.minimum(N_QUERIES, query_start + Q_TILE_SIZE)
        T_k_causal = tl.cdiv(tl.minimum(N_KEYS, query_end), K_TILE_SIZE)
        T_k_full = tl.minimum(query_start // K_TILE_SIZE, T_k_causal)

    # for j = 1, ..., T_k do
    for j in range(T_k_full):
        # load K^(j), V^(j) from global memory
        K = tl.load(K_block_ptr)
        V = tl.load(V_block_ptr)
        # compute tile of pre-softmax attention scores S_i^(j) = Q_i (K^(j))^T / sqrt(d)
        S = (tl.dot(Q, tl.trans(K)) * scale).to(tl.float32)
        # compute m_i^(j) = max(m_i^(j-1), rowmax(S_i^(j)))
        m_i_new = tl.maximum(m_i, tl.max(S, axis=-1))
        # compute P̃_i^(j) = exp(S_i^(j) - m_i^(j))
        P = tl.exp(S - m_i_new[:, None])
        # compute l_i^(j) = exp(m_i^(j-1) - m_i^(j)) * l_i^(j-1) + rowsum(P̃_i^(j))
        l_i = tl.exp(m_i - m_i_new) * l_i + tl.sum(P, axis=1)
        # compute O_i^(j) = diag(exp(m_i^(j-1) - m_i^(j))) * O_i^(j-1) + P̃_i^(j) V^(j)
        O_i = tl.exp(m_i - m_i_new)[:, None] * O_i + tl.dot(P.to(input_dtype), V.to(input_dtype))
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
        m_i = m_i_new

    # diagonal tiles — some key positions are in the future and must be masked out
    for j in range(T_k_full, T_k_causal):
        # load K^(j), V^(j) from global memory
        K = tl.load(K_block_ptr)
        V = tl.load(V_block_ptr)
        # compute tile of pre-softmax attention scores S_i^(j) = Q_i (K^(j))^T / sqrt(d)
        S = (tl.dot(Q, tl.trans(K)) * scale).to(tl.float32)
        Q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        K_idx = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
        S = tl.where(Q_idx[:, None] >= K_idx[None, :], S, S + -1e6)
        # compute m_i^(j) = max(m_i^(j-1), rowmax(S_i^(j)))
        m_i_new = tl.maximum(m_i, tl.max(S, axis=-1))
        # compute P̃_i^(j) = exp(S_i^(j) - m_i^(j))
        P = tl.exp(S - m_i_new[:, None])
        # compute l_i^(j) = exp(m_i^(j-1) - m_i^(j)) * l_i^(j-1) + rowsum(P̃_i^(j))
        l_i = tl.exp(m_i - m_i_new) * l_i + tl.sum(P, axis=1)
        # compute O_i^(j) = diag(exp(m_i^(j-1) - m_i^(j))) * O_i^(j-1) + P̃_i^(j) V^(j)
        O_i = tl.exp(m_i - m_i_new)[:, None] * O_i + tl.dot(P.to(input_dtype), V.to(input_dtype))
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
        m_i = m_i_new

    # compute O_i = diag(l_i)^{-1} * O_i
    O_i = O_i / l_i[:, None]
    # compute L_i = m_i + log(l_i)
    L_i = m_i + tl.log(l_i)
    # write O_i to global memory as the i-th tile of O
    tl.store(O_block_ptr, O_i.to(input_dtype))
    # write L_i to global memory as the i-th tile of L
    tl.store(L_block_ptr, L_i)


@triton.jit
def flash_bwd_preprocess_kernel(
    O_ptr, dO_ptr, D_ptr,
    stride_ob, stride_oq, stride_od,
    stride_dob, stride_doq, stride_dod,
    stride_db, stride_dq,
    N_QUERIES,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    O = tl.load(O_block_ptr).to(tl.float32)
    dO = tl.load(dO_block_ptr).to(tl.float32)
    # compute D = rowsum(dO ∘ O)
    D_i = tl.sum(dO * O, axis=1)
    tl.store(D_block_ptr, D_i)


@triton.jit
def flash_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, D_ptr, dO_ptr, L_ptr, dQ_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_db, stride_dq,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_dqb, stride_dqq, stride_dqd,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
    input_dtype: tl.constexpr,
    ):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
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
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_dqb,
        shape=(N_QUERIES, D),
        strides=(stride_dqq, stride_dqd),
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

    # for i = 1, ..., T_q (parallelized — each program owns one Q tile i)
    # load Q_i, dO_i, L_i, D_i from global memory
    Q = tl.load(Q_block_ptr)
    dO = tl.load(dO_block_ptr)
    L = tl.load(L_block_ptr).to(tl.float32)
    D_i = tl.load(D_block_ptr).to(tl.float32)
    # initialize dQ_i = 0
    dQ = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    T_k = tl.cdiv(N_KEYS, K_TILE_SIZE)
    loop_end = T_k
    full_end = T_k
    if is_causal:
        query_start = query_tile_index * Q_TILE_SIZE
        query_end = tl.minimum(N_QUERIES, (query_tile_index + 1) * Q_TILE_SIZE)
        loop_end = tl.cdiv(tl.minimum(N_KEYS, query_end), K_TILE_SIZE)
        full_end = tl.minimum(query_start // K_TILE_SIZE, loop_end)

    # for j = 1, ..., T_k do — fully visible K tiles, no causal mask needed
    for j in range(full_end):
        # load K^(j), V^(j) from global memory
        K = tl.load(K_block_ptr)
        V = tl.load(V_block_ptr)
        # compute tile of attention scores S_i^(j) = Q_i (K^(j))^T / sqrt(d), reusing saved L to get P
        S = (tl.dot(Q, tl.trans(K)) * scale).to(tl.float32)
        P = tl.exp(S - L[:, None])
        # compute dP_i^(j) = dO_i (V^(j))^T
        dP = tl.dot(dO.to(input_dtype), tl.trans(V.to(input_dtype)))
        # compute dS_i^(j) = P_i^(j) ∘ (dP_i^(j) - D_i) / sqrt(d)
        dS = P * (dP - D_i[:, None])
        # update dQ_i += dS_i^(j) K^(j)
        dQ += (tl.dot(dS.to(input_dtype), K.to(input_dtype)) * scale).to(tl.float32)

        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    # diagonal tiles — mask future positions before computing P
    for j in range(full_end, loop_end):
        # load K^(j), V^(j) from global memory
        K = tl.load(K_block_ptr)
        V = tl.load(V_block_ptr)
        # compute tile of attention scores S_i^(j) = Q_i (K^(j))^T / sqrt(d), reusing saved L to get P
        S = (tl.dot(Q, tl.trans(K)) * scale).to(tl.float32)
        Q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        K_idx = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
        S = tl.where(Q_idx[:, None] >= K_idx[None, :], S, S + -1e6)
        P = tl.exp(S - L[:, None])
        # compute dP_i^(j) = dO_i (V^(j))^T
        dP = tl.dot(dO.to(input_dtype), tl.trans(V.to(input_dtype)))
        # compute dS_i^(j) = P_i^(j) ∘ (dP_i^(j) - D_i) / sqrt(d)
        dS = P * (dP - D_i[:, None])
        # update dQ_i += dS_i^(j) K^(j)
        dQ += (tl.dot(dS.to(input_dtype), K.to(input_dtype)) * scale).to(tl.float32)

        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    # write dQ_i to global memory as the i-th tile of dQ
    tl.store(dQ_block_ptr, dQ.to(input_dtype))


@triton.jit
def flash_bwd_dkdv_kernel(
    Q_ptr, K_ptr, V_ptr, D_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_db, stride_dq,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
    input_dtype: tl.constexpr,
):
    # the "transpose" of flash_bwd_dq_kernel:
    # instead of owning a Q tile and looping over K tiles,
    # we own a K/V tile and loop over all Q tiles that can attend to it
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dkb,
        shape=(N_KEYS, D),
        strides=(stride_dkk, stride_dkd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dvb,
        shape=(N_KEYS, D),
        strides=(stride_dvk, stride_dvd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # Causal masking from the K tile's point of view: queries can only attend to
    # keys with index <= query index. So for this key tile:
    #   [0, q_tile_start): these Q tiles have all query indices < key_start → skip (future)
    #   [q_tile_start, q_tile_full_start): diagonal tiles where some queries can't see these keys
    #   [q_tile_full_start, T_q): these Q tiles all have query >= key, no masking needed
    T_q = tl.cdiv(N_QUERIES, Q_TILE_SIZE)
    q_tile_start = 0
    q_tile_full_start = 0
    if is_causal:
        key_start = key_tile_index * K_TILE_SIZE
        key_end = tl.minimum(N_KEYS, key_start + K_TILE_SIZE)
        q_tile_start = tl.minimum(key_start // Q_TILE_SIZE, T_q)
        q_tile_full_start = tl.minimum(tl.cdiv(key_end, Q_TILE_SIZE), T_q)
    q_start = q_tile_start * Q_TILE_SIZE

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(q_start, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(q_start, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(q_start,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(q_start,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # for j = 1, ..., T_k (parallelized — each program owns one K/V tile j)
    # load K^(j), V^(j) from global memory
    K = tl.load(K_block_ptr)
    V = tl.load(V_block_ptr)
    # initialize dK^(j) = dV^(j) = 0
    dK = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    dV = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)

    # for i = 1, ..., T_q do — diagonal Q tiles (partial causal mask)
    for i in range(q_tile_start, q_tile_full_start):
        # load Q_i, dO_i, L_i, D_i from global memory
        Q = tl.load(Q_block_ptr)
        dO = tl.load(dO_block_ptr)
        L = tl.load(L_block_ptr).to(tl.float32)
        D_i = tl.load(D_block_ptr).to(tl.float32)
        # compute tile of attention scores S_i^(j) = Q_i (K^(j))^T / sqrt(d), reusing saved L to get P
        S = (tl.dot(Q, tl.trans(K)) * scale).to(tl.float32)
        Q_idx = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        K_idx = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
        S = tl.where(Q_idx[:, None] >= K_idx[None, :], S, S + -1e6)
        P = tl.exp(S - L[:, None])
        # compute dV^(j) += (P_i^(j))^T dO_i
        dV += tl.dot(tl.trans(P.to(input_dtype)), dO.to(input_dtype))
        # compute dP_i^(j) = dO_i (V^(j))^T, then dS_i^(j) = P_i^(j) ∘ (dP_i^(j) - D_i) / sqrt(d)
        dP = tl.dot(dO.to(input_dtype), tl.trans(V.to(input_dtype)))
        dS = P * (dP - D_i[:, None])
        # compute dK^(j) += (dS_i^(j))^T Q_i
        dK += (tl.dot(tl.trans(dS.to(input_dtype)), Q.to(input_dtype)) * scale).to(tl.float32)

        Q_block_ptr = tl.advance(Q_block_ptr, (Q_TILE_SIZE, 0))
        dO_block_ptr = tl.advance(dO_block_ptr, (Q_TILE_SIZE, 0))
        L_block_ptr = tl.advance(L_block_ptr, (Q_TILE_SIZE,))
        D_block_ptr = tl.advance(D_block_ptr, (Q_TILE_SIZE,))

    # Q tiles where all queries can fully see this key tile — no masking needed
    for i in range(q_tile_full_start, T_q):
        # load Q_i, dO_i, L_i, D_i from global memory
        Q = tl.load(Q_block_ptr)
        dO = tl.load(dO_block_ptr)
        L = tl.load(L_block_ptr).to(tl.float32)
        D_i = tl.load(D_block_ptr).to(tl.float32)
        # compute tile of attention scores S_i^(j) = Q_i (K^(j))^T / sqrt(d), reusing saved L to get P
        S = (tl.dot(Q, tl.trans(K)) * scale).to(tl.float32)
        P = tl.exp(S - L[:, None])
        # compute dV^(j) += (P_i^(j))^T dO_i
        dV += tl.dot(tl.trans(P.to(input_dtype)), dO.to(input_dtype))
        # compute dP_i^(j) = dO_i (V^(j))^T, then dS_i^(j) = P_i^(j) ∘ (dP_i^(j) - D_i) / sqrt(d)
        dP = tl.dot(dO.to(input_dtype), tl.trans(V.to(input_dtype)))
        dS = P * (dP - D_i[:, None])
        # compute dK^(j) += (dS_i^(j))^T Q_i
        dK += (tl.dot(tl.trans(dS.to(input_dtype)), Q.to(input_dtype)) * scale).to(tl.float32)

        Q_block_ptr = tl.advance(Q_block_ptr, (Q_TILE_SIZE, 0))
        dO_block_ptr = tl.advance(dO_block_ptr, (Q_TILE_SIZE, 0))
        L_block_ptr = tl.advance(L_block_ptr, (Q_TILE_SIZE,))
        D_block_ptr = tl.advance(D_block_ptr, (Q_TILE_SIZE,))

    # write dK^(j) and dV^(j) to global memory as the j-th tiles of dK and dV
    tl.store(dK_block_ptr, dK.to(input_dtype))
    tl.store(dV_block_ptr, dV.to(input_dtype))


flash_fwd_kernel.best_config = "Q_TILE_SIZE: 128, K_TILE_SIZE: 128, num_warps: 8, num_ctas: 1, num_stages: 2, maxnreg: None"
flash_bwd_dq_kernel.best_config = "Q_TILE_SIZE: 128, K_TILE_SIZE: 128, num_warps: 8, num_ctas: 1, num_stages: 2, maxnreg: None"
flash_bwd_dkdv_kernel.best_config = "Q_TILE_SIZE: 64, K_TILE_SIZE: 64, num_warps: 4, num_ctas: 1, num_stages: 2, maxnreg: None"

class TritonFlashAttentionAutograd(Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        N_q = Q.shape[-2]
        N_k = K.shape[-2]
        d = Q.shape[-1]
        O = torch.empty_like(Q)
        L = torch.empty(*Q.shape[:-1], device=Q.device, dtype=torch.float32)
        input_dtype = dtype_mapping[Q.dtype]

        flash_fwd_kernel[(triton.cdiv(N_q, 128), Q.shape[0])](
            Q, K, V, O, L,
            *Q.stride(), *K.stride(), *V.stride(),
            *O.stride(), *L.stride(),
            N_q, N_k,
            1 / (d ** 0.5),
            D=d,
            Q_TILE_SIZE=128,
            K_TILE_SIZE=128,
            is_causal=is_causal,
            input_dtype=input_dtype,
            num_warps=8,
            num_stages=2,
        )

        ctx.is_causal = is_causal
        ctx.save_for_backward(L, Q, K, V, O)
        return O

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        n_queries = Q.shape[-2]
        n_keys = K.shape[-2]
        d = Q.shape[-1]
        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)
        D_values = torch.empty_like(L)
        input_dtype = dtype_mapping[Q.dtype]
        scale = d ** -0.5

        # preprocess kernel computes D_i = rowsum(dO * O) for every query position
        preprocess_q_tile = 128
        flash_bwd_preprocess_kernel[(n_queries // preprocess_q_tile, Q.shape[0])](
            O, dO, D_values,
            *O.stride(), *dO.stride(), *D_values.stride(),
            n_queries, d, preprocess_q_tile,
            num_warps=4, num_stages=2,
        )

        # Step 2: dQ and dK/dV write completely disjoint output buffers, so they can
        # run in parallel on two separate CUDA streams. We make both streams wait for
        # the current stream first (so D_values is ready), then launch concurrently,
        # then the current stream waits for both to finish before returning gradients.
        current_stream = torch.cuda.current_stream(device=Q.device)
        s_dq, s_dkdv = _get_bwd_streams(Q.device)
        s_dq.wait_stream(current_stream)
        s_dkdv.wait_stream(current_stream)

        with torch.cuda.stream(s_dq):
            flash_bwd_dq_kernel[(triton.cdiv(n_queries, 128), Q.shape[0])](
                Q, K, V, D_values, dO, L, dQ,
                *Q.stride(), *K.stride(), *V.stride(),
                *D_values.stride(), *dO.stride(), *L.stride(),
                *dQ.stride(),
                n_queries, n_keys, scale,
                D=d,
                Q_TILE_SIZE=128,
                K_TILE_SIZE=128,
                is_causal=ctx.is_causal,
                input_dtype=input_dtype,
                num_warps=8,
                num_stages=2,
            )

        with torch.cuda.stream(s_dkdv):
            flash_bwd_dkdv_kernel[(triton.cdiv(n_keys, 64), Q.shape[0])](
                Q, K, V, D_values, dO, L, dK, dV,
                *Q.stride(), *K.stride(), *V.stride(),
                *D_values.stride(), *dO.stride(), *L.stride(),
                *dK.stride(), *dV.stride(),
                n_queries, n_keys, scale,
                D=d,
                Q_TILE_SIZE=64,
                K_TILE_SIZE=64,
                is_causal=ctx.is_causal,
                input_dtype=input_dtype,
                num_warps=4,
                num_stages=2,
            )

        current_stream.wait_stream(s_dq)
        current_stream.wait_stream(s_dkdv)
        return dQ, dK, dV, None
