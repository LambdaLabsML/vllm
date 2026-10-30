# gpt-oss-120b: FlashAttention (FA4 on Blackwell) instead of FlashInfer trtllm-gen on the final config (serving flag only)

    vllm serve openai/gpt-oss-120b ... <final config flags: see gptoss-stack README> --attention-backend FLASH_ATTN

Exact attention with a different kernel. Long-context prefill attention (4K-16K new tokens over 40K-64K cached
tokens) is a large share of TTFT; FA4 (CuTe DSL) supports attention sinks on SM100.
