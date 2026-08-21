# Context-window management

Swiftrail tests four strategies against 10 fixed long conversations. Each
conversation plants an operational fact early, adds 28-45 noisy tool turns,
then asks for that fact at the end.

## Strategies

- `sliding_window`: keeps only the newest messages.
- `tool_output_masking`: preserves dialogue but masks old tool payloads.
- `recursive_summarization`: asks Mistral to compact older messages and keeps
  the latest messages verbatim.
- `zone_based_pruning`: keeps system messages and a recent-message zone.

## Unit tests

The strategy tests inject a fake generator, so they require no API key:

```powershell
python -m pytest context_eval/strategies -q
```

## Live evaluation

Copy `.env.example` to `.env`, set `MISTRAL_API_KEY`, then run:

```powershell
python -m context_eval.evaluate_strategies
```

The command writes:

```text
context_eval/results/context_comparison.json
context_eval/results/context_comparison.md
```

"Exact fact string retained" is a strict string-retention metric: the planted
fact must remain verbatim in the messages returned by a strategy. A summarized
or paraphrased fact does not pass this check, so a `0/10` result must not be
presented as proof that all semantic meaning was lost.

Input-token figures are a clearly labelled four-characters-per-token prompt
proxy. Recursive summarization reports Mistral's provider-reported output
tokens and includes its network/model time in latency. Deterministic strategies
make no model call and therefore report zero output tokens.

The checked-in comparison must be regenerated after changing model provider,
prompt, dataset, or strategy configuration. Never edit measured numbers by
hand.

## Selected default

`tool_output_masking` remains the default for the live agent because the main
source of Swiftrail context growth is repeated tool JSON. It reduces that noise
without adding a model call to every turn. Recursive summarization is kept as a
real Mistral-backed alternative and is measured honestly as the slower,
token-consuming option.
