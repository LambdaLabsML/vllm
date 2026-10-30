# gpt-oss-120b: FP8 KV cache on top of EAGLE3 (serving flags only; output-changing -> accuracy check)

    vllm serve openai/gpt-oss-120b ... <07 flags> --kv-cache-dtype fp8

Why: with speculative decoding the verification attention (3-4 query tokens per request over 48K-96K cached
tokens) is ~27% of decode GPU time on B200 and bound by KV-cache bandwidth; FP8 halves the bytes read.
