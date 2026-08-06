# ManiSkill LingBotVA Sample Rollouts

This directory contains five example evaluation instances for each
WorldSimProbe task, sampled from ManiSkill LingBotVA generations. Task 1
retains the original, small-perturbation, and large-perturbation videos for
every instance; Tasks 2--5 contain one candidate rollout per instance.

The examples are organized by the public Task 1--5 names. `manifest.jsonl`
records relative video paths and non-sensitive descriptive metadata. Task 4
examples cover distractor, fake-contact, and spatial-proximity stress tests;
Task 5 examples span five interaction primitives.

These files illustrate benchmark inputs and generated-rollout structure. They
are a small sample, are not representative of aggregate benchmark performance,
and do not include hidden reference videos, action streams, or private
evaluator annotations.

`scores.csv` and the `evaluation` field in `manifest.jsonl` record the cached
official evaluator score for each copied rollout. These per-instance scores
are provided for inspection only; their five-example means are not benchmark
estimates.
