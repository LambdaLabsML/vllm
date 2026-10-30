# gpt-oss-120b: EAGLE3 speculative decoding with NVIDIA's v3 head (serving flags only)

    vllm serve openai/gpt-oss-120b ... \
      --speculative-config.method eagle3 \
      --speculative-config.model nvidia/gpt-oss-120b-Eagle3-v3 \
      --speculative-config.num_speculative_tokens 3

Follow-up to gptoss-03-eagle3: the long-context head accepted only 14% of drafts on the SWE-bench agent replay
(mean acceptance length 1.42, TPOT +15%). v3 is trained on a newer data blend and uses YaRN RoPE to 131K.
