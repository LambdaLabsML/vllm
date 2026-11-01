# Qwen3.8-27B: MTP without the trailing prefix-cache block drop (serving flags only)

    vllm serve Qwen/Qwen3.8-27B ... <qwen-01 MTP flags> --speculative-config.disable_eagle_block_drop true

With EAGLE/MTP, vLLM drops the last matched prefix-cache block on every hit so the drafter's entry at the hit
boundary is recomputed. Qwen3.8-27B's hybrid (Mamba-align) cache uses 800-token blocks, so every agent turn
re-prefills ~800 already-cached tokens (replay: +43% uncached tokens, TTFT median +22% vs no spec decode).
Without the drop, at most one drafter KV entry per hit is stale; the target verifies every token, so outputs keep
the target distribution and only acceptance can change.
