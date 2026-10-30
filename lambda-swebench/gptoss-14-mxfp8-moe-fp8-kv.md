# gpt-oss-120b: MXFP8 MoE activations + FP8 KV cache on the final config (serving flags only; output-changing)

    vllm serve openai/gpt-oss-120b ... <final config flags: see gptoss-stack README> \
        --quantization-config.moe.activation mxfp8 --kv-cache-dtype fp8

Combination of 12 and 13 (fixed-load replay vs final: 12 TTFT -6% / TPOT -5%, 13 TTFT -3% / TPOT -2.4%).
