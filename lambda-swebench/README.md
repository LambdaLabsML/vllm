# gpt-oss-120b serving stack for the OpenCode SWE-bench agent workload (8x B200, 1 replica per GPU)

Cumulative branch: every change here was validated on a fixed-load replay of a recorded agent run (864 requests,
original arrival times and output lengths) against v0.32.0. Experiment branches and numbers:
`lambda-swebench/gptoss-*` and bench/EXPERIMENTS.md.

Code changes (on v0.32.0):
1. Reuse zeroed trtllm-gen multi-CTA KV counter buffers (one per kernel kind) - TPOT -2.8%.
2. Memoize gpt-oss legacy tool-call header rewriting (openai_harmony rebuilt a special-token regex per encode()) -
   TTFT -25%.
3. Render gpt-oss prompts message by message with a token cache - TTFT -42% (with 2), prompt render 70 -> 4 ms.

Serving flags (speculative decoding; exact sampling, the target verifies every token):

    vllm serve openai/gpt-oss-120b --max-model-len 131072 --enable-auto-tool-choice \
      --tool-call-parser openai --reasoning-parser openai_gptoss \
      --speculative-config.method eagle3 \
      --speculative-config.model <gpt-oss-120b-Eagle3-v3 with a 32K draft vocabulary> \
      --speculative-config.num_speculative_tokens 3 \
      --speculative-config.draft_sample_method probabilistic

The draft head is nvidia/gpt-oss-120b-Eagle3-v3 with its lm_head cut to 32,768 rows (tokens seen in held-out
gpt-oss outputs + all special tokens; bench/scripts/make_draft_vocab_head.py). Replay vs v0.32.0: TPOT -42.5%,
end-to-end request time -26% (spec decode alone).

Live OpenCode SWE-bench runs (8 replicas, 32 concurrent tasks), stack + EAGLE3 vs v0.32.0:
- sub50: 30/50 resolved (baseline 32/50, within noise); TTFT p50/p95 384/1059 -> 162/693 ms; TPOT p50/p95
  3.60/7.08 -> 2.05/4.96 ms; tasks/hour 65 -> 98; GPU-hours per resolved task 0.192 -> 0.136.
- sub20: TTFT p50/p95 255/846 -> 134/625 ms; TPOT p50/p95 3.23/5.29 -> 1.88/3.55 ms; tasks/hour 42 -> 58.
