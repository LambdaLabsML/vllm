# gpt-oss-120b: chunked-prefill token budget on the final config (serving flag only; scheduling)

    vllm serve openai/gpt-oss-120b ... <final config flags: see gptoss-stack README> --max-num-batched-tokens {8192,32768}

Default 16384. Uncached prefills on the agent trace reach ~16.7K tokens; a 16K-token chunk over a 64K context takes
~450 ms, stalling every co-batched decode (TPOT tail) while it runs. Smaller budget: shorter decode stalls, more
prefill steps (TTFT); larger: single-step prefills for the longest requests.
