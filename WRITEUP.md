2.1.3
(b)
small

- forward-only: 0.012483552504272665 seconds ± 6.916700928639048e-06 seconds
- forward-backward: 0.03757758270367049 seconds ± 2.8421599683508137e-05 seconds
- full: 0.05123383380123414 seconds ± 0.0005569040607991754 seconds

medium

- forward-only: 0.03765818640822545 seconds ± 1.4549029341365301e-05 seconds
- forward-backward: 0.11230256620037835 seconds ± 8.591247866268128e-05 seconds
- full: 0.14262789020431227 seconds ± 0.0030507038222256453 seconds

large

- forward-only: 0.08885995779710357 seconds ± 7.998553277053665e-05 seconds
- forward-backward: 0.2650387536996277 seconds ± 7.163240894850236e-05 seconds
- full: 0.30413982169702647 seconds ± 0.0007879235790531475 seconds

xl

- forward-only: 0.2710546380956657 seconds ± 0.0005837254801516107 seconds
- forward-backward: 0.793097680201754 seconds ± 0.00042054144871165304 seconds
- full: 0.8885817858000025 seconds ± 0.0018546173603436122 seconds

10b

- forward-only: 0.8979648717009695 seconds ± 0.0004916406544460476 seconds
- forward-backward: 2.671391514595598 seconds ± 0.00021758402306584234 seconds
- full: OOM

nsys_profile

small - context length: 512
(a) 13.639 ms. This matches what we measured.
(b) The cutlass matmul kernels take the most cumulative GPU time during forward pass (75.0%). These kernels are invoked 88 times during a single forward pass. They also take the most runtime for both forward and backward passes.
(c) The elementwise kernel for division takes a nontrivial amount of time at 2.9%, and the elementwise kernels for attention masking and addition take up 2.7% and 2.4% respectively.
(d) The fraction of time spent changes from 75.0% to 65.5%. The elementwise kernel for division goes from 2.9% to 3.6%, the elementwise kernel for masking drops from 2.7% to 1.5%, and the elementwise kernel for addition decreases from 2.4% to 1.7%.
(e) The matmul operations take around 8.8ms and the softmax operation takes up 0.78ms in runtime. Matmuls takes around 0.39T FLOPs, while softmax take 0.76B FLOPs. This means matmuls have 512x more FLOPs but take 11x longer in runtime.

small - context length:2048
(a) 18.388ms. This is slightly longer than what we measured.
(b) The cutlass matmul kernels take the most cumulative GPU time during forward pass (58.7%). These kernels are invoked 98 times during a single forward pass. They also take the most runtime for both forward and backward passes.
(c) The elementwise kernel for division takes a nontrivial amount of time at 8.7%, and the elementwise kernels for attention masking and addition take up 7.6% and 7.5% respectively.
(d) The fraction of time spent changes from 58.7% to 56.7%. The elementwise kernel for division goes from 9.0% to 11.0%, the elementwise kernel for masking drops from 7.6% to 4.9%, and the elementwise kernel for addition decreases from 7.5% to 5.0%.
(e) The matmul operations take around 9.8ms and the softmax operation takes up 3.0ms in runtime. Matmuls takes around 2T FLOPs, while softmax take 12B FLOPs. This means matmuls have 167x more FLOPs but take 3.3x longer in runtime.

small - context length:4096
(a) 30.497 ms. This seems much longer than what we had measured before with the Python library.
(b) The cutlass matmul kernels (4 variants) take the most cumulative GPU time during forward pass (46.7%). These kernels are invoked 13 times in total during a single forward pass. They also take the most runtime for both forward and backward passes.
(c) The elementwise kernel for division takes a nontrivial amount of time at 8.0%, and the elementwise kernels for attention masking and addition take up 15.1% and 10.4% respectively.
(d) The fraction of time spent changes from 46.7% to 47.4%. The elementwise kernel for division goes from 8.0% to 15.1%, the elementwise kernel for masking drops from 15.1% to 7.2%, and the elementwise kernel for addition decreases from 10.4% to 6.8%.
(e) The matmul operations take around 14ms and the softmax operation takes up 5.4ms in runtime. Matmuls take around 5.3T FLOPs, while softmax takes around 48B FLOPs. This means matmuls have ~110x more FLOPs but take only ~2.6x longer in runtime.

large - context length: 512
(a) 50.7 ms. This faster than what we measured.
(b) The cutlass matmul kernels take the most cumulative GPU time during forward pass (82.5%). These kernels are invoked 156 times during a single forward pass. They also take the most runtime for both forward and backward passes.
(c) The elementwise kernel for division takes a nontrivial amount of time at 2.2%, and the elementwise kernels for attention masking and addition take up 2.1% and 1.9% respectively.
(d) The fraction of time spent changes from 82.5% to 73.7%. The elementwise kernel for division goes from 2.2% to 2.8%, the elementwise kernel for masking drops from 2.1% to 1.3%, and the elementwise kernel for addition decreases from 1.9% to 1.2%.
(e) The matmul operations take around 40.0ms and the softmax operation takes up 2.5ms in runtime. Matmuls takes around 3.1T FLOPs, while softmax take 3.8B FLOPs. This means matmuls have 820x more FLOPs but take 16x longer in runtime.

large - context length:1024
(a) 107.335 ms. This is slightly slower than what we measured.
(b) The cutlass matmul kernels (4 variants) take the most cumulative GPU time during forward pass (76.8%). These kernels are invoked 153 times in total during a single forward pass. They also take the most runtime for both forward and backward passes.
(c) The elementwise kernel for division takes a nontrivial amount of time at 4.1%, and the elementwise kernels for attention masking and addition take up 3.6% and 3.5% respectively.
(d) The fraction of time spent changes from 76.8% to 72.4%. The elementwise kernel for division increases from 4.1% to 5.4%, the elementwise kernel for masking drops from 3.6% to 2.4%, and the elementwise kernel for addition decreases from 3.5% to 2.4%.
(e) The matmul operations take around 81ms and the softmax operation takes up 8.9ms in runtime. Matmuls take around 6.6T FLOPs, while softmax takes around 15.1B FLOPs. This means matmuls have ~435x more FLOPs but take only ~9.1x longer in runtime.

large - context length:2048
(a) 240.47 ms. This is much slower than what we measured.
(b) The cutlass matmul kernels take the most cumulative GPU time during forward pass (65.6%). These kernels are invoked 151 times during a single forward pass. They also take the most runtime for both forward and backward passes.
(c) The elementwise kernel for division takes a nontrivial amount of time at 7.3%, and the elementwise kernels for attention masking and addition take up 6.3% and 6.3% respectively.
(d) The fraction of time spent changes from 65.6% to 64.7%. The elementwise kernel for division increases from 7.3% to 9.2%, the elementwise kernel for masking drops from 6.3% to 4.1%, and the elementwise kernel for addition decreases from 6.3% to 4.1%.
(e) The matmul operations take around 156.7ms and the softmax operation takes up 33.4ms in runtime. Matmuls takes around 14.7T FLOPs, while softmax take 60.4B FLOPs. This means matmuls have 243x more FLOPs but take 4.7x longer in runtime.

mixed_precision_accumulation
We see the highest accuracy with fp32 and the lowest accuracy with fp16 for initialization and accumulation. We see that an initialization with fp32 and accumulation with fp16 leads to slightly worse accuracy than just fp32. Finally, we see that when we downcast fp32 to fp16 and accumulate with fp32, we see the same accuracy as the case before. The fact that two cases are equivalent seems to indicates that the accumulation loop automatically upcasts fp16 to fp32 before adding, giving you more precision despite that the initial value has lower precision.

benchmarking_mixed_precision

a
dtype is fp32
fc1 is fp16
ln is fp32
fc2 is fp16
loss is fp32
gradients is fp32

b
The parts that are sensitive to mixed precision include (1) squaring the activations during variance computation, since we can easily have overflow, and (2) summing for mean calculation, since we saw precision loss when we accumulated too many values in fp16. If we use BF16, we don't need to treat normalization differently because BF16 has a much larger max value and therefore is not at risk of overflow.

c
For the small model, the forward pass is around the same time (13.639ms without mixed precision and 14.644ms with). The backward pass is slightly slower at 27.321ms with mixed precision than without (23.874ms). With the large model, the backward passes are much faster (54.448ms with mixed precision than 444.480ms without). The forward passes also are slightly faster (72.661ms vs 107.335 ms). As model size increases, the speedups from mixed precision are much clearer.

memory_profiling
a
You can see a zigzag pattern in the memory timelines for the full training, while the timelines for just the forward pass look like triangles, with sudden increases and drops. You can tell that the increase is during the forward run, and the flat line with tiny spikes at the top is during the backward pass, while the sudden high peak is during optimizer step.

b
The peak memory usage for a context length of 128 during a forward pass is ~20.5GB, while the peak usage for a context length of 1024 is ~70.2GB. During a full training step, the peak usage is ~60.2GB for a context length of 128, and ~107.4GB for a context length of 1024.
[Note: I could only use 1024 for the context length since 2048 ran into OOM without mixed-precision.]

c
The peak memory usage for a forward pass with mixed-precision is now ~20.5 GB for a context length of 128 and ~48GB for a context length of 1024. During a full training step, it is ~57.2GB for a context length of 128 and ~85.8 GB for a context length of 1024. Mixed-precision helped decrease memory usage when we ran a long context length.

d
For a context length of 128, the size of the residual stream tensor is batch_size x context_length x d_model, which gives us 4 \* 128 \* 2560 \* 4 / (1024 \*\* 2) = 5 MB. For a context length of 1024, the size is 4 \* 1024 \* 2560 \* 4 / (1024 \*\* 2) = 40 MB.

e
The largest allocation is 50MB for the xl model with 128 context length during the forward pass with mixed precision. The allocation is from the SwiGLU forward pass. For the same model but with 1024 context length, the largest allocation is 256MB. The allocation here is from the attention masking.

f

gradient_checkpointing

a
Recursive checkpointing can help minimize peak activation memory. We start by storing only the model input, and as we go through each block during the backward pass, we recompute its input by running a forward through the blocks before it. We then run the current block with gradients to get the activations needed. Since we only need to store one active residual at a time, the peak memory would be O(1), but the compute would be O(N^2) since we need to recompute every block from scratch.

assume that in forward we only save the input
in backward pass, we now do recursive checkpointing

x0 = saved_checkpoint # original input

for i in reversed block indices: # iterate blocks in reverse
with torch.no_grad(): # recompute xi
x = x0
for b in blocks[:i]:
x = b(x)
xi = x.detach()

    with torch.enable_grad():
        xj = blocks[i](xi)

    xj.backward(grad)
    grad = xi.grad

return grad

b
The best checkpointing strategy requires us to balance the group size with the number of groups, since we need store each group's residual, and while we backprop through a group, we also need to hold all the activations within that group in memory. If we have N blocks split into g groups, our peak memory should be g \* a + (N/g) \* r, where a is the activation memory for each block and r is the residual size per. To minimize this, we can take the derivative over the formula and set it to zero. We end up with g = sqrt(N \* r/a). For the xl model, we have 32 layers, a residual size of 80MB (4 \* 2048 \* 2560 \* 4), and an activation memory of ~2GB ( from attention = 4 \* 32 \* 2048 \* 2048). This gives us ~1 (= sqrt(32 \* 25)).

group_size= 1 num_groups=32: peak = 68.29 GB
group_size= 2 num_groups=16: peak = 74.70 GB
group_size= 4 num_groups= 8: peak = 87.49 GB
group_size= 6 num_groups= 5: peak = 100.26 GB

pytorch_attention

a

| d_model | seq    | fwd (ms)      | bwd (ms)      | mem_before_bwd |
| ------- | ------ | ------------- | ------------- | -------------- |
| 16      | 256    | 0.15 ± 0.02   | 1.37 ± 4.47   | 12.6 MB        |
| 16      | 1024   | 0.16 ± 0.01   | 1.38 ± 7.04   | 74.2 MB        |
| 16      | 4096   | 1.80 ± 0.69   | 4.13 ± 0.06   | 1048.6 MB      |
| 16      | 8192   | 7.27 ± 2.89   | 15.25 ± 2.41  | 4129.0 MB      |
| 16      | 16384  | 27.80 ± 10.78 | 57.41 ± 0.83  | 16433.8 MB     |
| 16      | 32768  | OOM           | —             | —              |
| 16      | 65536  | OOM           | —             | —              |
| 16      | 131072 | OOM           | —             | —              |
| 32      | 256    | 0.14 ± 0.03   | 0.66 ± 0.62   | 21.3 MB        |
| 32      | 1024   | 0.17 ± 0.01   | 0.68 ± 0.30   | 84.3 MB        |
| 32      | 4096   | 1.86 ± 0.67   | 4.10 ± 1.88   | 1048.5 MB      |
| 32      | 8192   | 7.73 ± 3.39   | 15.46 ± 0.46  | 4145.0 MB      |
| 32      | 16384  | 29.27 ± 13.60 | 59.76 ± 16.86 | 16465.8 MB     |
| 32      | 32768  | OOM           | —             | —              |
| 32      | 65536  | OOM           | —             | —              |
| 32      | 131072 | OOM           | —             | —              |
| 64      | 256    | 0.13 ± 0.02   | 0.64 ± 0.18   | 22.3 MB        |
| 64      | 1024   | 0.19 ± 0.01   | 0.71 ± 0.16   | 88.3 MB        |
| 64      | 4096   | 2.23 ± 0.76   | 4.92 ± 0.70   | 1072.6 MB      |
| 64      | 8192   | 9.02 ± 4.08   | 17.73 ± 0.55  | 4177.0 MB      |
| 64      | 16384  | 34.84 ± 16.14 | 68.77 ± 1.20  | 16529.8 MB     |
| 64      | 32768  | OOM           | —             | —              |
| 64      | 65536  | OOM           | —             | —              |
| 64      | 131072 | OOM           | —             | —              |
| 128     | 256    | 0.14 ± 0.01   | 1.06 ± 1.51   | 24.3 MB        |
| 128     | 1024   | 0.23 ± 0.02   | 0.92 ± 0.15   | 96.3 MB        |
| 128     | 4096   | 2.80 ± 1.07   | 5.94 ± 0.09   | 1104.6 MB      |
| 128     | 8192   | 11.36 ± 5.20  | 22.39 ± 0.06  | 4241.0 MB      |
| 128     | 16384  | 43.57 ± 20.27 | 84.82 ± 0.06  | 16657.8 MB     |
| 128     | 32768  | OOM           | —             | —              |
| 128     | 65536  | OOM           | —             | —              |
| 128     | 131072 | OOM           | —             | —              |

OOM errors occur at seq=32768 for all d_model values.

For d_model=16, the memory breakdown is:

- Q, K, V tensors: 3 × batch × seq × d_model × 4 bytes = 3 × 8 × 32768 × 16 × 4 = 50 MB
- Attention weight matrix A: batch × seq × seq × 4 bytes = 8 × 32768² × 4 = 32,768 MB
- Output: batch × seq × d_model × 4 bytes ≈ 16 MB

We see that the attention weight matrix dominates the memory usage. The attention matrix A has size batch × seq² × bytes, so memory grows quadratically (O(N²)) with sequence length. Doubling seq from 8192→16384 grows mem_before_bwd from 4129 MB to 16433 MB, roughly a 4× increase. We can elimitate this memory cost by recomputing attention during the backward pass rather than storing it.

torch_compile

| d_model | seq   | Uncompiled fwd (ms) | Compiled fwd (ms) | Uncompiled bwd (ms) | Compiled bwd (ms) | Uncompiled mem (MB) | Compiled mem (MB) |
| ------- | ----- | ------------------- | ----------------- | ------------------- | ----------------- | ------------------- | ----------------- |
| 16      | 256   | 0.15 ± 0.02         | 0.26 ± 0.02       | 1.37 ± 4.47         | 7.01 ± 62.09      | 12.6                | 12.7              |
| 16      | 1024  | 0.16 ± 0.01         | 0.31 ± 0.02       | 1.38 ± 7.04         | 1.45 ± 5.77       | 74.2                | 82.4              |
| 16      | 4096  | 1.80 ± 0.69         | 1.13 ± 0.29       | 4.13 ± 0.06         | 2.17 ± 0.11       | 1048.6              | 1048.8            |
| 16      | 8192  | 7.27 ± 2.89         | 4.17 ± 1.70       | 15.25 ± 2.41        | 7.56 ± 0.10       | 4129.0              | 4129.2            |
| 16      | 16384 | 27.80 ± 10.78       | 15.86 ± 6.55      | 57.41 ± 0.83        | 74.66 ± 482.39    | 16433.8             | 16434.2           |
| 32      | 256   | 0.14 ± 0.03         | 0.14 ± 0.01       | 0.66 ± 0.62         | 4.62 ± 43.63      | 21.3                | 13.2              |
| 32      | 1024  | 0.17 ± 0.01         | 0.13 ± 0.01       | 0.68 ± 0.30         | 4.42 ± 41.49      | 84.3                | 76.2              |
| 32      | 4096  | 1.86 ± 0.67         | 1.26 ± 0.37       | 4.10 ± 1.88         | 5.15 ± 31.46      | 1048.5              | 1056.8            |
| 32      | 8192  | 7.73 ± 3.39         | 4.47 ± 1.94       | 15.46 ± 0.46        | 7.21 ± 0.04       | 4145.0              | 4145.2            |
| 32      | 16384 | 29.27 ± 13.60       | 17.31 ± 6.76      | 59.76 ± 16.86       | 59.81 ± 320.62    | 16465.8             | 16466.2           |
| 64      | 256   | 0.13 ± 0.02         | 0.32 ± 0.02       | 0.64 ± 0.18         | 0.68 ± 0.25       | 22.3                | 22.3              |
| 64      | 1024  | 0.19 ± 0.01         | 0.21 ± 0.00       | 0.71 ± 0.16         | 0.33 ± 0.04       | 88.3                | 88.4              |
| 64      | 4096  | 2.23 ± 0.76         | 1.70 ± 0.48       | 4.92 ± 0.70         | 2.90 ± 0.10       | 1072.6              | 1072.8            |
| 64      | 8192  | 9.02 ± 4.08         | 5.72 ± 2.21       | 17.73 ± 0.55        | 9.42 ± 0.06       | 4177.0              | 4177.2            |
| 64      | 16384 | 34.84 ± 16.14       | 22.80 ± 10.57     | 68.77 ± 1.20        | 37.51 ± 0.12      | 16529.8             | 16530.2           |
| 128     | 256   | 0.14 ± 0.01         | 0.23 ± 0.04       | 1.06 ± 1.51         | 0.25 ± 0.06       | 24.3                | 24.3              |
| 128     | 1024  | 0.23 ± 0.02         | 0.23 ± 0.00       | 0.92 ± 0.15         | 0.40 ± 0.01       | 96.3                | 96.4              |
| 128     | 4096  | 2.80 ± 1.07         | 2.17 ± 0.92       | 5.94 ± 0.09         | 3.68 ± 0.01       | 1104.6              | 1104.8            |
| 128     | 8192  | 11.36 ± 5.20        | 8.15 ± 3.65       | 22.39 ± 0.06        | 14.23 ± 0.03      | 4241.0              | 4241.2            |
| 128     | 16384 | 43.57 ± 20.27       | 31.58 ± 13.35     | 84.82 ± 0.06        | 86.27 ± 321.82    | 16657.8             | 16658.2           |

The memory is roughly the same between the compiled and uncompiled versions. The compute time is faster for the compiled version during a forward pass than the uncompiled version.

b
data_size num_gpu mean_time std_time
0 262144 2 0.000028 0.000005
1 262144 4 0.000040 0.000006
2 262144 6 0.000057 0.000172
3 2621440 2 0.000092 0.000295
4 2621440 4 0.000106 0.000177
5 2621440 6 0.000097 0.000032
6 26214400 2 0.000240 0.000042
7 26214400 4 0.000347 0.000293
8 26214400 6 0.000332 0.000173
9 268435456 2 0.001923 0.000260
10 268435456 4 0.002564 0.000017
11 268435456 6 0.002732 0.000016

The time doesn't scale linearly with data size. As data size increases, latency increases but by a smaller factor. This seems to suggest that a fixed latency overhead dominates at the smaller sizes, but bandwidth dominates at larger sizes. The increase in GPU numbers also doesn't always lead to a faster communication, since there's also a growing communication cost.

naive_ddp_benchmarking

num_gpu mean_training_time std_training_time mean_communication_time std_communication_time communication_pct
0 2 1.844413 0.002226 0.039807 0.003796 2.158235

I ran 5 warmup steps with 10 measurement steps, and each rank had a batch size of 4. After each backward pass, I synchronized the gradients by all-reducing each model parameter. I averaged the timings across both ranks with an all-gather.

minimal_ddp_flat_benchmarking

world_size mean_training_time std_training_time mean_communication_time std_communication_time communication_pct
0 2 1.85573 0.001566 0.037168 0.001187 2.002902
There doesn't seem much difference other than in the standard deviation.

optimizer_state_sharding_accounting

a
With sharding:
[Rank 0] After model init: total=13125.0 MB params=12995.9 MB grads=0.0 MB opt_states=0.0 MB
[Rank 0] Before optimizer step: total=39484.7 MB params=12995.9 MB grads=12995.9 MB opt_states=13191.2 MB
[Rank 0] After optimizer step: total=39484.7 MB params=12995.9 MB grads=12995.9 MB opt_states=13191.2 MB

Without sharding:
[Rank 0] After model init: total=13125.0 MB params=12995.9 MB grads=0.0 MB opt_states=0.0 MB
[Rank 0] Before optimizer step: total=52285.4 MB params=12995.9 MB grads=12995.9 MB opt_states=25991.9 MB
[Rank 0] After optimizer step: total=52285.4 MB params=12995.9 MB grads=12995.9 MB opt_states=25991.9 MB

b
use_sharding num_gpu mean_training_time std_training_time mean_communication_time std_communication_time communication_pct
0 True 2 1.842472 0.001536 0.036310 0.000603 1.97070
1 False 2 1.854803 0.000878 0.036118 0.000444 1.94728
