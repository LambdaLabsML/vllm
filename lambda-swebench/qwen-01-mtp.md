# Qwen3.8-27B: native MTP speculative decoding (serving flags only)

    vllm serve Qwen/Qwen3.8-27B ... --speculative-config.method mtp \
        --speculative-config.num_speculative_tokens <k> --speculative-config.draft_sample_method probabilistic

Qwen3.8-27B (dense 27B BF16, Gated-DeltaNet hybrid) ships a 1-layer MTP head. Decode on the agent trace is
bandwidth-bound small-batch (2-4 requests per GPU; TPOT 13.9 ms ~ 54 GB of weights per step), so verified drafts
amortize the weight reads. Exact speculative sampling: target distribution unchanged.
