# Task 3 Teleoperation

Task 3 can use control traces recorded from a real operator through the
included browser console. The console drives an official RoboTwin environment,
shows the live simulator view beside the original episode, and records the
operator's commands, realized joint actions, simulator telemetry, and output
video.

Procedurally generated control profiles are not human teleoperation and are
intentionally excluded from this collection path.

## Requirements

Install WorldSimProbe with the optional console dependencies:

```bash
python -m pip install -e ".[teleoperation]"
```

Prepare:

- an official [RoboTwin](https://github.com/robotwin-Platform/RoboTwin)
  checkout with its environment already working;
- RoboTwin episode data with `joint_action/vector`, instructions, and the
  original episode video;
- a writable annotation directory. Missing annotation rows are generated from
  the selected official episode when a session starts.

## Start The Console

```bash
python scripts/run_task3_operator.py \
  --robotwin-root /path/to/RoboTwin \
  --annotation-root /path/to/task3_annotations \
  --output-root outputs/task3_operator
```

The equivalent environment variables are `ROBOTWIN_ROOT` and
`WORLDSIMPROBE_TASK3_ANNOTATION_ROOT`. Open `http://127.0.0.1:8791/`.
For a remote worker, forward the port over SSH:

```bash
ssh -L 8791:127.0.0.1:8791 user@worker
```

The interface supports both arms, translation, rotation, and gripper commands.
Each command is executed in the simulator before the next live frame is
displayed. Exporting a session writes:

```text
<output-root>/<episode-and-time>/
├── teleoperation.mp4
├── teleoperation_raw.mp4
├── first_frame.png
├── teleoperation_trace.json
├── task3_trace_validation.json
└── dataset_row.json
```

`physical_drive_target` is the default execution mode. `force_qpos` is
available only for debugging because it can introduce non-physical impulses.

## Validation Boundary

The trace validator checks command order, screen or world direction, rotation,
gripper direction, inactive-arm stability, and whether the commanded action
changed. These checks are collection-time integrity gates.

The official Task 3 benchmark score remains RobotSeg-masked optical-flow
agreement between a submitted candidate video and the hidden source-reference
video. The operator UI does not expose private benchmark rows or hidden
references.
