# gpt-oss-120b: CUDA RMSNorm kernels instead of Inductor-generated ones (serving flags only)

    vllm serve openai/gpt-oss-120b ... \
      --ir-op-priority.fused_add_rms_norm vllm_c --ir-op-priority.rms_norm vllm_c

Why: at decode batch sizes (1-8 tokens) the default `native` lowering becomes single-CTA Triton reductions
(`triton_red_fused_fused_add_rms_norm_*`, grid 1) taking 3.3-4.1 us each, two per layer (~0.26 ms of a ~3.25 ms
step on B200).
