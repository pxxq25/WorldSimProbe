# Public Examples

These videos are deterministic synthetic format examples. They are not derived
from the hidden leaderboard set and are not intended to demonstrate model
quality.

Rebuild them with:

```bash
python scripts/build_examples.py
```

Validate the resulting package with:

```bash
worldsimprobe validate-submission \
  --manifest examples/example_submission/submission.jsonl \
  --root examples/example_submission \
  --decode
```

## Real Generated-Rollout Samples

Five ManiSkill LingBotVA examples per task are available in
[`maniskill_lingbot_samples/`](maniskill_lingbot_samples/). These samples
demonstrate the generated-video organization without exposing hidden benchmark
references.
