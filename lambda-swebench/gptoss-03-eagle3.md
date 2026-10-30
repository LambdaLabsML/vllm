# gpt-oss-120b: EAGLE3 speculative decoding (serving flags only)

Lossless speculative decoding with NVIDIA's long-context EAGLE3 head (the SWE-bench agent prompts are 48K-96K
tokens, p50/p90), draft length 3 as in the model card:

    vllm serve openai/gpt-oss-120b ... \
      --speculative-config.method eagle3 \
      --speculative-config.model nvidia/gpt-oss-120b-Eagle3-long-context \
      --speculative-config.num_speculative_tokens 3

Why: decode on 8x B200 at 2-4 running requests per GPU is a ~3.2 ms chain of ~630 small, latency-bound kernels per
token; verifying several draft tokens per target forward amortizes it. Rejection sampling keeps the output
distribution of the target model.
