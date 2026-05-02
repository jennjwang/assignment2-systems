import torch


default_chunk_size = 8192


class FusedLinearCrossEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, hidden, weight, targets, chunk_size):
        chunk_size = int(chunk_size)
        hidden_shape = hidden.shape
        hidden = hidden.reshape(-1, hidden_shape[-1])
        targets = targets.reshape(-1)

        N = hidden.shape[0]

        # Stream over token chunks.
        # Each chunk does one normal LM-head matmul, reduces CE immediately,
        # then drops the logits before moving on to the next token chunk.
        loss_sum = torch.zeros((), dtype=torch.float32, device=hidden.device)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            hidden_chunk = hidden[start:end]
            target_chunk = targets[start:end]

            logits = torch.einsum('bd,vd->bv', hidden_chunk, weight)
            logits_f = logits.float()

            log_sum_exp = torch.logsumexp(logits_f, dim=-1)
            target_logit = logits_f.gather(1, target_chunk[:, None]).squeeze(1)
            loss_sum = loss_sum + (log_sum_exp - target_logit).sum()

        loss = loss_sum / N

        ctx.save_for_backward(hidden, weight, targets)
        ctx.chunk_size = chunk_size
        ctx.hidden_shape = hidden_shape
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        hidden, weight, targets = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        N = hidden.shape[0]

        dL_dhidden = torch.zeros_like(hidden, dtype=torch.float32)
        dL_dweight = torch.zeros_like(weight, dtype=torch.float32)
        scale = grad_output.float() / N

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            hidden_chunk = hidden[start:end]
            target_chunk = targets[start:end]

            logits = torch.einsum('bd,vd->bv', hidden_chunk, weight)  # (token_chunk, vocab)
            logits_f = logits.float()

            # d CE / d logit_v = softmax_v - 1[v == target]
            probs = torch.softmax(logits_f, dim=-1)
            probs.scatter_add_(
                1,
                target_chunk[:, None],
                -torch.ones((end - start, 1), device=probs.device, dtype=probs.dtype),
            )
            probs.mul_(scale)
            dL_dhidden[start:end] = torch.einsum('bv,vd->bd', probs.to(weight.dtype), weight).float()
            dL_dweight.add_(torch.einsum('bv,bd->vd', probs.to(hidden.dtype), hidden_chunk).float())

        return dL_dhidden.reshape(ctx.hidden_shape).to(hidden.dtype), dL_dweight.to(weight.dtype), None, None


def fused_linear_cross_entropy(hidden, weight, targets, chunk_size=default_chunk_size):
    return FusedLinearCrossEntropy.apply(hidden, weight, targets, chunk_size)
