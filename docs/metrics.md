# Metrics

## Task 1

Let `Ds` and `Dl` be the time-aligned video MSE induced by the small and large
actions. The hidden simulator provides `Ds_sim` and `Dl_sim`. The evaluator
compares `Ds / Dl` with `Ds_sim / Dl_sim` using the triangular oracle-ratio
score implemented in `oracle_ratio.py`. The reported value is the mean score
multiplied by 100.

## Tasks 2 and 3

RobotSeg supplies robot masks and the configured flow estimator supplies dense
flow. Per-window error is normalized by the larger of reference robot-flow RMS
and the motion floor:

```text
100 * max(0, 1 - flow_error / max(reference_flow_rms, motion_floor))
```

The reported score is the mean over active windows and samples.

## Task 4

Task 4 evaluates `distractor_hallucination`, `fake_contact_hallucination`, and
`proximity_hallucination`. TAPNext++ tracks the distractor object in the first
subset and the target object in the latter two. The evaluated object's stored
initial location defines a 3-by-3 query grid inscribed within a true 10-pixel
radius, and its tracked centroid is measured in the canonical 256-by-256
coordinate system. This keeps every query point within the stated radius;
placing each grid axis at plus or minus 10 would put corner queries 14.14
pixels from the center.

A rollout passes when the evaluated object's maximum displacement is at most
10 pixels and the RobotSeg centroid motion gate confirms at least 20 pixels of
robot-arm displacement across its three gate frames. Motion-gate or tracking
failure receives zero.

## Task 5

The VLM performs an eight-way forced choice and separately verifies agent
motion, object motion, interaction visibility, visual integrity, and physical
plausibility. The principal score is primitive accuracy after the two motion
checks. Per-primitive scores and the confusion matrix are also reported.
