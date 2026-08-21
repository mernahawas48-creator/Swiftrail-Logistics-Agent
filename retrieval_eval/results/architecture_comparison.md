# Swiftrail Retrieval Architecture Comparison

Model: `mistral-small-latest`
Fixed test cases: 10

| Architecture | Correct / Total | Accuracy | Avg. input tokens/query | Avg. output tokens/query | Avg. total tokens/query | Avg. latency/query | Avg. retrieval attempts | Safe abstentions | Transient API retries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Naive RAG | 5/10 | 50.0% | 291.9 | 25.8 | 317.7 | 0.642s | 1.00 | 7 | 0 |
| Hybrid RAG | 6/10 | 60.0% | 272.6 | 29.7 | 302.3 | 0.597s | 1.00 | 5 | 0 |
| Agentic RAG | 7/10 | 70.0% | 647.7 | 28.0 | 675.7 | 0.682s | 1.40 | 3 | 0 |

## Per-case results

| Architecture | Case | Category | Correct | Sources | Verification | Attempts | API retries | Latency | Reason |
|---|---|---|---|---|---|---:|---:|---:|---|
| Naive RAG | semantic-severe-release | naive_friendly | yes | CH-3, CH-1, CH-2 | pass | 1 | 0 | 1.605s | Passed the fixed answer rubric. |
| Naive RAG | semantic-hold-thresholds | naive_friendly | yes | CH-1, CH-2, PR-2 | pass | 1 | 0 | 0.851s | Passed the fixed answer rubric. |
| Naive RAG | semantic-invoice-followup | naive_friendly | yes | IC-2, IC-1, IC-3 | pass | 1 | 0 | 0.717s | Passed the fixed answer rubric. |
| Naive RAG | exact-re2 | hybrid_friendly | no | SP-2, RE-3, RE-1 | fail | 1 | 0 | 0.087s | The pipeline abstained on an answerable case. |
| Naive RAG | exact-ac4 | hybrid_friendly | no | AC-2, SP-4, AC-4 | fail | 1 | 0 | 0.758s | The pipeline abstained on an answerable case. |
| Naive RAG | exact-sp3 | hybrid_friendly | no | SP-3, CH-3, SP-4 | pass | 1 | 0 | 0.554s | The pipeline abstained on an answerable case. |
| Naive RAG | multi-part-discount-and-hold | agentic_friendly | no | RE-2, RE-4 | fail | 1 | 0 | 0.952s | The pipeline abstained on an answerable case. |
| Naive RAG | multi-part-pricing-and-hold | agentic_friendly | no | SP-4, RE-4 | fail | 1 | 0 | 0.671s | The pipeline abstained on an answerable case. |
| Naive RAG | unauthorized-pr2 | authorization | yes | CH-2, SP-2, SP-3 | fail | 1 | 0 | 0.100s | Safe abstention expected and returned. |
| Naive RAG | unsupported-storage-fee | safe_abstention | yes | SP-3, SP-2, SP-4 | fail | 1 | 0 | 0.121s | Safe abstention expected and returned. |
| Hybrid RAG | semantic-severe-release | naive_friendly | no | CH-3, CH-1, CH-2 | fail | 1 | 0 | 0.733s | The pipeline abstained on an answerable case. |
| Hybrid RAG | semantic-hold-thresholds | naive_friendly | yes | CH-1, CH-2, CH-4 | pass | 1 | 0 | 0.778s | Passed the fixed answer rubric. |
| Hybrid RAG | semantic-invoice-followup | naive_friendly | yes | IC-2, IC-1, IC-3 | pass | 1 | 0 | 0.894s | Passed the fixed answer rubric. |
| Hybrid RAG | exact-re2 | hybrid_friendly | yes | RE-2 | pass | 1 | 0 | 0.704s | Passed the fixed answer rubric. |
| Hybrid RAG | exact-ac4 | hybrid_friendly | no | AC-4 | fail | 1 | 0 | 0.505s | The pipeline abstained on an answerable case. |
| Hybrid RAG | exact-sp3 | hybrid_friendly | no | SP-3 | pass | 1 | 0 | 0.487s | Missing required answer evidence: shipment |
| Hybrid RAG | multi-part-discount-and-hold | agentic_friendly | no | CH-1, RE-4 | fail | 1 | 0 | 0.967s | The pipeline abstained on an answerable case. |
| Hybrid RAG | multi-part-pricing-and-hold | agentic_friendly | yes | SP-4, CH-3 | pass | 1 | 0 | 0.723s | Passed the fixed answer rubric. |
| Hybrid RAG | unauthorized-pr2 | authorization | yes | - | fail | 1 | 0 | 0.077s | Safe abstention expected and returned. |
| Hybrid RAG | unsupported-storage-fee | safe_abstention | yes | SP-1, CH-1, IC-2 | fail | 1 | 0 | 0.105s | Safe abstention expected and returned. |
| Agentic RAG | semantic-severe-release | naive_friendly | no | CH-3, CH-1, CH-2 | fail | 1 | 0 | 1.142s | The pipeline abstained on an answerable case. |
| Agentic RAG | semantic-hold-thresholds | naive_friendly | yes | CH-1, CH-2, CH-4 | pass | 1 | 0 | 0.899s | Passed the fixed answer rubric. |
| Agentic RAG | semantic-invoice-followup | naive_friendly | yes | IC-2, IC-1, IC-3 | pass | 1 | 0 | 0.669s | Passed the fixed answer rubric. |
| Agentic RAG | exact-re2 | hybrid_friendly | yes | RE-2 | pass | 1 | 0 | 0.520s | Passed the fixed answer rubric. |
| Agentic RAG | exact-ac4 | hybrid_friendly | no | AC-4 | pass | 1 | 0 | 0.532s | Missing required answer evidence: session |
| Agentic RAG | exact-sp3 | hybrid_friendly | no | SP-3 | pass | 1 | 0 | 0.500s | Missing required answer evidence: shipment |
| Agentic RAG | multi-part-discount-and-hold | agentic_friendly | yes | CH-1, RE-4, RE-2, CH-3, AC-3, RE-1 | pass | 2 | 0 | 1.172s | Passed the fixed answer rubric. |
| Agentic RAG | multi-part-pricing-and-hold | agentic_friendly | yes | SP-4, CH-3, SP-1, SP-3, RE-4 | pass | 2 | 0 | 1.006s | Passed the fixed answer rubric. |
| Agentic RAG | unauthorized-pr2 | authorization | yes | - | fail | 2 | 0 | 0.175s | Safe abstention expected and returned. |
| Agentic RAG | unsupported-storage-fee | safe_abstention | yes | SP-1, CH-1, IC-2, SP-4, SP-3, SP-2 | fail | 2 | 0 | 0.206s | Safe abstention expected and returned. |
