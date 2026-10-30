# gpt-oss-120b: MXFP8 MoE activations on the final config (serving flags only; output-changing -> accuracy check)

    vllm serve openai/gpt-oss-120b ... <final config flags: see gptoss-stack README> --quantization-config.moe.activation mxfp8

Why: TTFT on the agent trace is now dominated by 4K-16K-token uncached prefills over 40K-64K contexts (48% of summed
TTFT, ~28 ms per 1K uncached tokens). The MXFP4 MoE runs W4A16; MXFP8 activations use the W4A8 trtllm-gen path
(offline screen: uncached prefill -15..22%, decode ~neutral).
