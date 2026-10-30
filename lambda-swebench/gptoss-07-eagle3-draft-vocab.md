# gpt-oss-120b: EAGLE3 v3 head with a 32K draft vocabulary (serving flags + derived draft checkpoint)

The v3 draft head computes logits over the full 201,088-token vocabulary (a 1.16 GB lm_head read per draft
step, ~0.17 ms on B200). Held-out gpt-oss outputs on SWE-bench agent tasks (425 tasks of the full-500 run that are
not in sub20/sub50/smoke10; 6.8M tokens) use only 26,751 distinct tokens, so the derived checkpoint keeps
those + all special tokens, filled to 32,768 rows (vLLM `draft_vocab_size` + `d2t` mapping; built by
bench/scripts/make_draft_vocab_head.py). Output distribution is unchanged: the target verifies every token.

    vllm serve openai/gpt-oss-120b ... \
      --speculative-config.method eagle3 \
      --speculative-config.model /data/dj-mat-mfu-v3/models/gpt-oss-120b-Eagle3-v3-dv32k \
      --speculative-config.num_speculative_tokens 3 \
      --speculative-config.draft_sample_method probabilistic
