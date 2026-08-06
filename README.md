# WorldSimProbe

WorldSimProbe evaluates whether action-conditioned world models follow the
supplied control stream and ground environment responses in the realized robot
motion.

This public repository contains:

- the Task 1-5 evaluation implementation;
- the video submission contract and validator;
- small, synthetic examples that demonstrate the file format;
- documentation for preparing a leaderboard submission.

It does not contain the leaderboard test set, hidden simulator references,
training code, model checkpoints, or baseline-specific inference code.

## Tasks

| Task | Name | Primary diagnostic |
| --- | --- | --- |
| 1 | Local Action Calibration | Local response to graded action changes |
| 2 | Global Trajectory Coverage | Realization of cross-task donor actions |
| 3 | Action-Source Behavior Preservation | Preservation of heterogeneous action sources |
| 4 | Interaction Grounding | Target and distractor response under object binding |
| 5 | Interaction Dynamics | Contact-conditioned primitive recognition |

Task 4 evaluates distractor-object, fake-contact, and spatial-proximity
settings. Task 3 includes an interactive RoboTwin operator console for
collecting real human control traces; generated human-like action profiles are
not part of that interface or the public Task 3 collection path.

## Simulator Resources

WorldSimProbe currently evaluates videos derived from three widely used
simulators:

- [RoboTwin](https://github.com/robotwin-Platform/RoboTwin)
- [ManiSkill](https://github.com/mani-skill/ManiSkill)
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)

These links are provided for users who want compatible training data or local
simulator development. WorldSimProbe submissions contain generated videos, not
simulator installations.

## Installation

```bash
python -m pip install -e .
```

Install optional evaluator dependencies only on the leaderboard worker that
runs the corresponding task:

```bash
python -m pip install -e ".[flow,task4,task5]"
```

Install the operator-console dependencies only when collecting Task 3
teleoperation traces:

```bash
python -m pip install -e ".[teleoperation]"
```

## Submission

The leaderboard distributes model inputs separately from this repository.
Hidden simulator references and annotations remain server-side. Run your model
on the assigned inputs and submit:

```text
submission/
├── submission.jsonl
└── videos/
    ├── <sample_id>.mp4
    └── ...
```

Each sample contains exactly one prediction. Task 1 requires `original`,
`small`, and `large` videos, while Tasks 2-5 require one `candidate` video.
Every candidate must cover the exact requested time horizon, subject only to
one-frame timestamp rounding. Submitted videos use the evaluator-owned timing
configuration, which defaults to 10 FPS; manifests cannot override timing.

Validate before upload:

```bash
worldsimprobe validate-submission \
  --manifest submission/submission.jsonl \
  --root submission \
  --decode
```

Create an archive:

```bash
worldsimprobe package-submission \
  --manifest submission/submission.jsonl \
  --root submission \
  --output worldsimprobe_submission.zip
```

See [submission.md](docs/submission.md) for the exact JSONL and video contract.
The files under `examples/` are format examples only and are not leaderboard
test samples.

## Evaluation

Leaderboard workers join a submission with a private reference manifest, then
run the task-specific evaluators in `worldsimprobe.evaluation`. The public
implementation exposes all metric logic, while hidden rows provide only the
reference videos, actions, object metadata, and opaque sample identifiers.

Task 5 uses the frozen VLM prompt and the final 12-frame physical-time sampling
protocol. No additional GT-oracle filtering is applied to model scores.

The Task 3 operator console, its RoboTwin adapter, and the trace-integrity gate
are documented in [task3_teleoperation.md](docs/task3_teleoperation.md).

## Data Privacy

The repository intentionally excludes:

- benchmark test videos and initial frames;
- hidden action streams and simulator trajectories;
- private sample IDs and annotations;
- absolute paths from internal machines;
- model checkpoints and generated benchmark predictions.

`scripts/check_public_release.py` enforces these constraints before release.
