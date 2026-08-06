# Tasks

## Task 1: Local Action Calibration

For one initial scene, the model receives original, small-perturbation, and
large-perturbation action streams. The evaluator measures the video MSE between
the original output and each perturbed output, then compares the model response
ratio with the hidden simulator response ratio.

## Task 2: Global Trajectory Coverage

A donor action stream from a different task is executed in a receiver scene.
RobotSeg-masked optical flow compares generated robot motion with the hidden
counterfactual simulator reference.

## Task 3: Action-Source Behavior Preservation

Task 3 uses the same masked-flow principle with heterogeneous control sources,
including independently trained policies and real human teleoperation traces.
The optional operator console under
`task3_action_coverage/teleoperation/` drives RoboTwin directly, records the
resulting simulator video and action trace, and checks that commands changed
the requested arm while preserving the inactive arm. Synthetic human-like
profiles are not accepted as human teleoperation.

## Task 4: Interaction Grounding

The official setting evaluates distractor-object, fake-contact, and spatial-
proximity hallucination conditions. TAPNext++ tracks the evaluated object in
the generated rollout, while RobotSeg verifies that the agent moved. The Task 4
score measures whether unsupported environment interaction remains absent.

## Task 5: Interaction Dynamics

A VLM classifies the realized interaction as one of eight primitives:
`push`, `rotate`, `slide_drag`, `pull`, `tap`, `shake`, `drop`, or
`knock_over`. Candidate clips use 12 samples on a shared physical-time grid.
The primitive decision is gated by agent-motion and object-motion agreement.
Leaderboard model scores do not use an additional GT-oracle filter.
