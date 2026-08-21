# Context management strategy comparison

10 long-context transcripts (28-45 tool-noise turns each), one critical fact planted early, re-asked at the final turn.

| Strategy | Exact fact string retained | Avg. input tokens/run | Avg. output tokens/run | Avg. latency |
| --- | --- | --- | --- | --- |
| sliding_window | 0/10 | 270.8 | 0.0 | 0.001ms |
| tool_output_masking | 10/10 | 1197.5 | 0.0 | 0.022ms |
| recursive_summarization | 0/10 | 365.8 | 296.1 | 2202.531ms |
| zone_based_pruning | 0/10 | 253.1 | 0.0 | 0.011ms |
