# gpt-oss-120b: Inductor graph partitioning (serving flag only)

    vllm serve openai/gpt-oss-120b ... --compilation-config.use_inductor_graph_partition true

Inductor compiles the whole model graph and partitions it at the attention/KV-cache splitting ops itself, instead of
vLLM splitting the FX graph into separately compiled pieces first. On the fixed-load replay of the SWE-bench agent
trace (v0.32.0, no spec decode): TPOT median -2.3%, TPOT p95 5.58 -> 4.34 ms, TTFT -7%, e2e -3.9%.

RESULT: rejected. With this flag gpt-oss-120b (v0.32.0) produces NaN logits: greedy requests fail with
"Out of range float values are not JSON compliant: nan" and EAGLE3 draft acceptance drops from 48% to 0%. The
apparent TPOT gain on the no-spec replay came from broken computation (forced-length replays still time it).
