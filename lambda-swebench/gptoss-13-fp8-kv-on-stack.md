# gpt-oss-120b: FP8 KV cache on the final config (serving flags only; output-changing -> accuracy check)

    vllm serve openai/gpt-oss-120b ... <final config flags: see gptoss-stack README> --kv-cache-dtype fp8

Why: long-context prefill attention (full-attention layers over 40K-64K cached tokens) is a large part of TTFT for
the 4K-16K-token uncached prefills; FP8 KV runs FP8 trtllm-gen attention and halves KV bytes for decode verification.
08 measured this on the EAGLE3 flags without the stack; this re-measures it on the final config, for TTFT.
