import math
from typing import Optional
from typing import Tuple
import torch
from torch import inference_mode
import torch.nn.functional as F
import torch.utils.benchmark as benchmark
from torch.nn.functional import scaled_dot_product_attention

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


def meta_forward(
    x: torch.Tensor,
    start_pos: int,
    freqs_cis: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    wo: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    n_local_heads: int,
    n_local_kv_heads: int,
    n_rep: int,
    head_dim: int,
    mask: Optional[torch.Tensor] = None,
):
    bsz, seqlen, _ = x.shape
    xq, xk, xv = wq(x), wk(x), wv(x)

    xq = xq.view(bsz, seqlen, n_local_heads, head_dim)
    xk = xk.view(bsz, seqlen, n_local_kv_heads, head_dim)
    xv = xv.view(bsz, seqlen, n_local_kv_heads, head_dim)

    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

    cache_k = cache_k.to(xq)
    cache_v = cache_v.to(xq)

    cache_k[:bsz, start_pos : start_pos + seqlen] = xk
    cache_v[:bsz, start_pos : start_pos + seqlen] = xv

    keys = cache_k[:bsz, : start_pos + seqlen]
    values = cache_v[:bsz, : start_pos + seqlen]

    # repeat k/v heads if n_kv_heads < n_heads
    keys = repeat_kv(
        keys, n_rep
    )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
    values = repeat_kv(
        values, n_rep
    )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

    xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
    keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
    values = values.transpose(
        1, 2
    )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
    scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(head_dim)
    if mask is not None:
        scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
    scores = F.softmax(scores.float(), dim=-1).type_as(xq)
    output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
    
    output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
    return wo(output)


from model.llama3 import Attention
from model.llama3 import precompute_complex_positional_embeddings

with torch.no_grad():
    attention = Attention(
        model_dimension=4096,
        number_of_heads=32,
        number_of_kv_heads=8,
        batch_size_limit=32,
        sequence_lenght_limit=2048,
    )

    freqs_cis = precompute_complex_positional_embeddings(4096, 2048, 500000)
    freqs_cis = freqs_cis[:, :64]
    
    x = torch.randn(2, 2048, 4096).cuda()
    attention.k_cache.sequence_cache.random_()
    attention.v_cache.sequence_cache.random_()
    attention.q_projector.weight.random_()
    attention.k_projector.weight.random_()
    attention.v_projector.weight.random_()
    attention.output_projector.weight.random_()

    freqs_cis = freqs_cis.cuda()
    attention.cuda()

    # Define parameters
    wq, wk, wv, wo = attention.q_projector, attention.k_projector, attention.v_projector, attention.output_projector
    cache_k = attention.k_cache.sequence_cache.transpose(1, 2).detach().clone()
    cache_v = attention.v_cache.sequence_cache.transpose(1, 2).detach().clone()

    n_local_heads = attention.number_of_heads
    n_local_kv_heads = attention.number_of_kv_heads
    n_rep = attention.number_of_heads // attention.number_of_kv_heads
    head_dim = attention.model_dimension // attention.number_of_heads
    start_pos = 0
    mask = None

    # Benchmark meta_forward
    meta_forward_timer = benchmark.Timer(
        stmt='meta_forward(x, start_pos, freqs_cis, wq, wk, wv, wo, cache_k, cache_v, n_local_heads, n_local_kv_heads, n_rep, head_dim, mask)',
        setup='from __main__ import meta_forward, x, start_pos, freqs_cis, wq, wk, wv, wo, cache_k, cache_v, n_local_heads, n_local_kv_heads, n_rep, head_dim, mask',
        globals=globals(),
        num_threads=1
    )
    
    # Run and print results
    meta_forward_time = meta_forward_timer.timeit(10)
    print("Meta's implementation benchmark:")
    print(meta_forward_time)

    # Benchmark Attention
    attention_forward_timer = benchmark.Timer(
        stmt='attention(x,freqs_cis,start_pos)',
        setup='from __main__ import attention, x',
        globals=globals(),
        num_threads=1
    )
    
    attention_time = attention_forward_timer.timeit(10)
    print("Own implementation benchmark:")
    print(attention_time)

    print(f"Meta's: {meta_forward_time.mean} ms")
    print(f"Mine: {attention_time.mean} ms")
