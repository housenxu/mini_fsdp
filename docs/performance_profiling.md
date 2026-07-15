# MiniFSDP performance measurement: MFU, Profiler, and Nsight

## The four numbers are not interchangeable

MiniFSDP now reports four deliberately separate views of performance. The clean benchmark prints end-to-end throughput and a formula-based model TFLOP/s per GPU. PyTorch Profiler writes an operator-formula FLOP estimate for its measured window. Nsight Systems shows where time is spent and whether collectives overlap compute. Nsight Compute reports per-kernel hardware behavior such as Tensor Core use, roofline position, memory traffic, occupancy, and warp stalls. Only the first view with the correct vendor denominator is the conventional end-to-end MFU.

## Standard formula MFU

For the dense causal Transformer workload, the benchmark uses:

```text
FLOPs/token = 6P + 12 * L * H * S
model TFLOP/s/GPU = local tokens/s * FLOPs/token / 1e12
MFU = model TFLOP/s/GPU / vendor peak TFLOP/s/GPU
```

`P` is the unique global parameter count, so tied token/output embeddings are not counted twice. `L`, `H`, and `S` are the number of Transformer blocks, hidden size, and sequence length. The `6P` term approximates forward and backward parameterized operations, while the second term restores sequence-length-dependent attention work. This is a model-work estimate; it is not read from GPU performance counters.

The denominator must match the exact GPU SKU, dense precision used by the GEMMs, sparsity mode, and clock convention. MiniFSDP intentionally has no default vendor peak. Omitting `--peak-tflops` prints `N/A` instead of silently using an H100 value for another GPU.

Run a clean FSDP2 benchmark with a known vendor peak as follows:

```bash
torchrun --standalone --nproc_per_node=2 benchmark.py \
  --model transformer --strategy fsdp2 \
  --fsdp2-param-dtype bf16 --fsdp2-reduce-dtype fp32 \
  --peak-tflops <dense_bf16_peak_for_one_gpu>
```

Rank zero writes `benchmark_results/formula_mfu_fsdp2.json`. The printed global tokens/s includes all data-parallel ranks, whereas the MFU calculation uses local tokens/s and a per-GPU peak.

## Practical GEMM ceiling

A data-sheet peak is useful for standard MFU, but a same-machine cuBLAS benchmark is a useful tuning ceiling. Generate it with model-like shapes and the same compute dtype:

```bash
python benchmarks/gemm_peak.py \
  --dtype bfloat16 \
  --sizes 2048 4096 8192 \
  --shape 1024,2048,512 \
  --shape 1024,512,2048 \
  --output benchmark_results/gemm_peak.json
```

The benchmark then reads that file automatically. `efficiency_vs_practical_gemm_peak` is not standard MFU; it answers how much of the locally observed GEMM ceiling is converted into end-to-end model work. It may expose poor shapes or software overhead even when the vendor denominator is not known.

## PyTorch Profiler

Run:

```bash
PEAK_TFLOPS=<optional_vendor_peak> bash tools/run_torch_profiler.sh
```

Every rank writes a TensorBoard trace and `rankN/profiler_metrics.json`. The profiler is enabled with `with_flops=True`, but these FLOPs are PyTorch formulas for supported operators. Fused scaled-dot-product attention and custom kernels can be missing, and tracing changes wall time. Therefore `profiler_estimated_mfu_vs_vendor_percent` is a diagnostic estimate, not a replacement for the clean end-to-end formula MFU.

Useful ranges include `fsdp2::forward_backward`, `transformer::block_N`, `minifsdp::all_gather_params`, `minifsdp::reduce_scatter_grads`, and `fsdp2::optimizer_step`. Compare every rank when looking for stragglers.

## Nsight Systems

Run a bounded two-step capture:

```bash
bash tools/run_nsight_systems.sh
```

Set `GPU_METRICS=1` only when the machine permits GPU metric sampling:

```bash
GPU_METRICS=1 bash tools/run_nsight_systems.sh
```

The script uses the CUDA Profiler API so warmup and shutdown do not dominate the report, and MiniFSDP emits matching NVTX ranges. Inspect CPU launch gaps, CUDA synchronizations, NCCL AllGather/ReduceScatter duration, communication-computation overlap, kernel gaps, memcpy traffic, and rank imbalance. GPU Metrics can add SM Active, Tensor Active, and DRAM bandwidth samples. Tensor Active is a cycle-activity signal, not MFU: a job may have high Tensor Active inside short GEMMs but still have low end-to-end MFU because of idle gaps or communication waits.

## Nsight Compute and Tensor Core use

Run the intentionally small replay:

```bash
bash tools/run_nsight_compute.sh
```

Open the generated `.ncu-rep` and first identify expensive GEMM and scaled-dot-product-attention kernels. In Speed Of Light and Roofline, inspect achieved compute throughput, arithmetic intensity, DRAM/L2 pressure, and whether the kernel is compute-bound or memory-bound. In Launch Statistics and Occupancy, inspect grid size, waves per SM, registers, shared memory, achieved occupancy, and whether shapes leave Tensor Core tiles underfilled. The exact tensor-pipe metric name changes across GPU architectures and Nsight versions; use the Tensor Core or Tensor Pipe subsection generated for that device instead of hard-coding one metric name.

Do not profile the whole distributed iteration with every Nsight Compute section. Metric replay serializes and reruns kernels, greatly perturbing NCCL and potentially causing timeouts. Use Nsight Systems to find the expensive kernel, then adjust `LAUNCH_SKIP` and `LAUNCH_COUNT` to inspect a small stable window.

## Interview-safe interpretation

A precise answer is: “I use clean step time and a dense Transformer FLOPs model for standard end-to-end MFU, with a vendor peak matching the exact GPU and dtype. I also run a same-machine GEMM benchmark as a practical ceiling. PyTorch Profiler gives operator-formula FLOPs and a timeline, not hardware-measured MFU. Nsight Systems diagnoses idle time and communication overlap; Nsight Compute verifies per-kernel Tensor Core and roofline behavior. I keep these values separate in both console output and JSON.”

## References

PyTorch Profiler: https://docs.pytorch.org/docs/stable/profiler.html

Nsight Systems User Guide: https://docs.nvidia.com/nsight-systems/UserGuide/

Nsight Compute Profiling Guide: https://docs.nvidia.com/nsight-compute/ProfilingGuide/