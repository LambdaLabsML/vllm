# gpt-oss-120b: Inductor graph partitioning (serving flag only)

    vllm serve openai/gpt-oss-120b ... --compilation-config.use_inductor_graph_partition true

Inductor compiles the whole model graph and partitions it at the attention/KV-cache splitting ops itself, instead of
vLLM splitting the FX graph into separately compiled pieces first. On the fixed-load replay of the SWE-bench agent
trace (v0.32.0, no spec decode): TPOT median -2.3%, TPOT p95 5.58 -> 4.34 ms, TTFT -7%, e2e -3.9%.
