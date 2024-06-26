import math

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

# Load the CUDA kernel as a python module
minimal_attn = load(
    name='minimal_attn',
    sources=['main.cpp', 'flash_attention_1.cu', 'flash_attention_2.cu'],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
)

# Use small model params, otherwise slower than manual attention. See caveats in README.
batch_size = 8
n_head = 12
seq_len = 1024
head_embd = 64

q = torch.randn(batch_size, n_head, seq_len, head_embd, requires_grad=True).cuda()
k = torch.randn(batch_size, n_head, seq_len, head_embd, requires_grad=True).cuda()
v = torch.randn(batch_size, n_head, seq_len, head_embd, requires_grad=True).cuda()

print('====== profiling forward pass ======')

print('=== profiling manual attention (forward pass) ===')

# Our minimal flash attention aims to be faster than this by avoiding HBM read/writes of N^2 matrices.
def manual_attn(q, k, v):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    # add casual mask
    mask = torch.tril(torch.ones(att.size(-2), att.size(-1)), diagonal=0).cuda()
    att = att.masked_fill(mask == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    manual_result = manual_attn(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print("\n\n")

print('=== profiling minimal flash attention (forward pass) === ')

with (
    torch.autograd.profiler.profile(use_cuda=True) as prof,
    torch.no_grad(),
):
    minimal_result, l, m = minimal_attn.flash_attention_1_forward(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print(
    'attn values sanity check:',
    torch.allclose(minimal_result, manual_result, rtol=0, atol=1e-02),
)

print('=== profiling minimal flash attention 2 (forward pass) === ')

with (
    torch.autograd.profiler.profile(use_cuda=True) as prof,
    torch.no_grad(),
):
    minimal_result_2, L = minimal_attn.flash_attention_2_forward(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print(
    'attn values sanity check:',
    torch.allclose(minimal_result_2, manual_result, rtol=0, atol=1e-02),
)

print("\n\n\n\n")

print('====== profiling backward pass ======')

print('=== profiling manual attention (backward pass) ===')

y_grad = torch.ones_like(minimal_result)

def manual_attn_backward(q, k, v, y, y_grad):
    return torch.autograd.grad([y], [q, k, v], grad_outputs=[y_grad])

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    manual_grad_q, manual_grad_k, manual_grad_v = manual_attn_backward(
        q, k, v, manual_result, y_grad
    )
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('=== profiling minimal flash attention (backward pass) === ')

with (
    torch.autograd.profiler.profile(use_cuda=True) as prof,
    torch.no_grad(),
):
    (
        minimal_grad_q,
        minimal_grad_k,
        minimal_grad_v,
    ) = minimal_attn.flash_attention_1_backward(q, k, v, minimal_result, y_grad, l, m)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print(
    'q grad sanity check:',
    torch.allclose(manual_grad_q, minimal_grad_q, rtol=0, atol=1e-02),
)
print(
    'k grad sanity check:',
    torch.allclose(manual_grad_k, minimal_grad_k, rtol=0, atol=1e-02),
)
print(
    'v grad sanity check:',
    torch.allclose(manual_grad_v, minimal_grad_v, rtol=0, atol=1e-02),
)

print("\n\n")

print('=== profiling minimal flash attention 2 (backward pass) === ')

with (
    torch.autograd.profiler.profile(use_cuda=True) as prof,
    torch.no_grad(),
):
    (
        minimal_grad_q,
        minimal_grad_k,
        minimal_grad_v,
    ) = minimal_attn.flash_attention_2_backward(q, k, v, minimal_result_2, y_grad, L)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print(
    'q grad sanity check:',
    torch.allclose(manual_grad_q, minimal_grad_q, rtol=0, atol=1e-02),
)
print(
    'k grad sanity check:',
    torch.allclose(manual_grad_k, minimal_grad_k, rtol=0, atol=1e-02),
)
print(
    'v grad sanity check:',
    torch.allclose(manual_grad_v, minimal_grad_v, rtol=0, atol=1e-02),
)
