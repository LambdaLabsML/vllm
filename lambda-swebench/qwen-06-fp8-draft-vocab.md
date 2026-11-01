# Qwen3.8-27B: stack recipe (qwen-02 MTP + qwen-03 token cache + qwen-05 draft vocab + qwen-04 FP8)

    vllm serve Qwen/Qwen3.8-27B --tool-call-parser qwen3_coder --reasoning-parser qwen3 --max-model-len 262144 \
        --speculative-config.method mtp --speculative-config.num_speculative_tokens 6 \
        --speculative-config.draft_sample_method probabilistic \
        --speculative-config.disable_eagle_block_drop true \
        --speculative-config.draft_vocab_path <draft_vocab_32k.json> \
        --quantization fp8

draft_vocab_32k.json: the 32,768 most frequent output tokens (plus all added tokens) of held-out tasks
(bench/scripts/draft_vocab_counts.py --hf-tokenizer on qwen_full500 minus sub20/sub50; 99.99% coverage).
FP8 passed the sub50 accuracy gate (38/50 vs 37/50 baseline).
