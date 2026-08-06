# Submission Format

## Manifest

`submission.jsonl` contains one JSON object per assigned sample:

```json
{
  "schema_version": "1.0",
  "sample_id": "opaque-sample-id",
  "task_id": "task2",
  "model_id": "my-world-model",
  "videos": {
    "candidate": "videos/opaque-sample-id.mp4"
  }
}
```

Each manifest row represents exactly one prediction for one sample.
Frame rate is evaluator-owned and must not appear in the manifest. The packaged
`worldsimprobe.configs/video.yaml` sets the official default to 10 FPS. The
validator decodes each video and rejects a frame rate that differs from the
configured value.

Task 1 instead uses:

```json
{
  "videos": {
    "original": "videos/id__original.mp4",
    "small": "videos/id__small.mp4",
    "large": "videos/id__large.mp4"
  }
}
```

Paths must be relative, remain inside the submission directory, and identify
decodable video files. `sample_id` values must exactly match the IDs supplied
by the leaderboard.

## Leaderboard Workflow

1. Obtain an inference-input package through the leaderboard.
2. Generate one complete-horizon video for each assigned sample and action
   variant.
3. Write `submission.jsonl`.
4. Run `worldsimprobe validate-submission --decode`.
5. Upload the archive produced by `worldsimprobe package-submission`.

The inference-input package contains only the model inputs needed for
prediction. Hidden simulator references, object tracks, success metadata, and
metric annotations are never distributed.
