# Qwen3.8-27B: online FP8 weight quantization on the Qwen stack (serving flag only; output-changing)

    vllm serve Qwen/Qwen3.8-27B ... <qwen-03 stack flags> --quantization fp8

Dense 27B BF16: decode reads ~54 GB of weights per step and prefill is GEMM-bound; FP8 W8A8 (per-tensor weight
scales computed at load, dynamic activation scales) halves weight bytes and doubles tensor-core throughput.
Changes numerics -> sub50 accuracy run required (baseline sub50 74%).
