# Retrieval Architecture Evaluation

This folder compares the complete Naive, Hybrid, and Agentic RAG answer
pipelines on the same fixed questions. It records answer accuracy,
provider-reported token usage, end-to-end latency, verification, and retrieval
attempts.

`rag/evaluation/` is different: it measures retrieval ranking quality such as
Hit@K, Recall@K, MRR, and access safety.

## Fixed test set

`questions.json` covers semantic lookup, exact policy IDs, multi-section
questions, authorization-sensitive retrieval, and safe abstention. Do not
change it after seeing results or between architecture runs.

## Run

Set `MISTRAL_API_KEY` and `MISTRAL_MODEL` in the root `.env`, then start Qdrant
and rebuild the collection:

```powershell
docker compose -f rag/vector_store/docker-compose.yml up -d
python -m rag.ingestion.pipeline --recreate
python -m retrieval_eval.evaluate_architectures
```

Outputs:

```text
retrieval_eval/results/architecture_comparison.json
retrieval_eval/results/architecture_comparison.md
```

Token counts come from Mistral/LangChain `usage_metadata`; they are not
estimated from character counts. Transient 429 and provider 5xx responses are
retried with exponential backoff. Successful cases are checkpointed, allowing
an interrupted run to resume without changing the fixed dataset.

The old checked-in report was generated with Gemini and is historical evidence,
not a valid post-migration comparison. Regenerate both result files with
Mistral before presenting or merging the provider-migration pull request.
