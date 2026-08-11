# Video Contract

- Container: MP4
- Pixel format: RGB-compatible 8-bit video
- Orientation: upright, with no vertical flip
- Timeline: full requested horizon
- Ordering: chronological
- Frame rate: 10 FPS by default, controlled by the evaluator-owned
  `worldsimprobe.configs/video.yaml`
- Resolution: model-native output is accepted and resized only where the
  metric requires alignment
- Overlays: no labels, borders, contact sheets, or debug UI

Models may generate native chunks internally, but the submitted candidate must
cover the complete timeline assigned to the sample. Early termination, repeated
short chunks, or a single frame held for the missing horizon fail validation on
the official evaluator.

Submission manifests cannot override frame rate or timestamps. Validation
decodes the media and compares its frame rate with the evaluator configuration.
