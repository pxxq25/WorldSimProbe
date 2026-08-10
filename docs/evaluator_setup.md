# Evaluator Worker Setup

## Requirements

- Linux with an NVIDIA GPU and a working CUDA toolchain;
- Python 3.11, matching the upstream RobotSeg environment;
- enough local storage for RobotSeg, TAPNext++, DPFlow, and the Task 5 VLM;
- access to the private WorldSimProbe reference manifests.

Install the PyTorch build matching the worker's CUDA runtime first, following
the [official PyTorch selector](https://pytorch.org/get-started/locally/). Then
install the public evaluator package:

```bash
python -m pip install -e ".[flow,task4,task5]"
mkdir -p checkpoints
```

The command installs PTLFlow 0.4.1, Hydra 1.3.2, the video dependencies, and
the Task 5 Transformers stack. Model source trees and weights are intentionally
kept out of the package installation.

## Task 2 and Task 3: RobotSeg and DPFlow

WorldSimProbe's RobotSeg adapter expects the public Show Lab package and its
`robotseg.build_robotseg` API. Install the pinned compatible source revision:

```bash
git clone https://github.com/showlab/RobotSeg.git checkpoints/RobotSeg
git -C checkpoints/RobotSeg checkout dafb8c0d507276e2f96d2b07ac3661a7b3a41a5f
python -m pip install -e checkpoints/RobotSeg
(cd checkpoints/RobotSeg && python setup.py build_ext --inplace)
```

Download `robotseg.pt` from the links in the
[official RobotSeg release instructions](https://github.com/showlab/RobotSeg#72-download)
and save it as:

```text
checkpoints/robotseg.pt
```

Download the official PTLFlow `things` checkpoint used by the frozen protocol:

```bash
curl -L \
  https://github.com/hmorimitsu/ptlflow/releases/download/weights1/dpflow-things-2012b5d6.ckpt \
  -o checkpoints/dpflow-things-2012b5d6.ckpt
```

Pass these paths to `scripts/evaluate_task2.py` and
`scripts/evaluate_task3.py` through `--robotseg-root`,
`--robotseg-checkpoint`, and `--dpflow-checkpoint`.

## Task 4: TAPNext++

Install the compatible Google DeepMind TAPNet source. WorldSimProbe uses the
PyTorch implementation at `tapnet.tapnext.tapnext_torch`:

```bash
git clone https://github.com/google-deepmind/tapnet.git checkpoints/tapnet
git -C checkpoints/tapnet checkout c2cbab81cc06092b5f05bfe2da7bfec54e2079c9
python -m pip install einops torchvision
python -m pip install --no-deps -e checkpoints/tapnet
curl -L \
  https://storage.googleapis.com/dm-tapnet/tapnextpp/tapnextpp_ckpt.pt \
  -o checkpoints/tapnextpp_ckpt.pt
```

The TAPNext++ checkpoint is approximately 2.5 GB. Pass its path, plus the same
RobotSeg paths used above, to `scripts/evaluate_task4.py`.

## Task 5: Frozen VLM

The frozen Task 5 protocol defaults to the public
[`Qwen/Qwen3-VL-8B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
model. Transformers downloads it into the Hugging Face cache on first use, or
the worker may pre-download it and pass the local model directory. Keep the
model identifier and revision fixed in the worker deployment record.

## Expected Asset Layout

```text
checkpoints/
├── RobotSeg/
├── tapnet/
├── robotseg.pt
├── dpflow-things-2012b5d6.ckpt
└── tapnextpp_ckpt.pt
```

Do not commit checkpoints or private reference manifests to this repository.

## Smoke Test

After installing the assets, verify that all public evaluator imports resolve:

```bash
python - <<'PY'
import hydra
import ptlflow
import torch
from robotseg.build_robotseg import build_robotseg_video_predictor
from tapnet.tapnext.tapnext_torch import TAPNext
from transformers import AutoModelForImageTextToText, AutoProcessor

assert torch.cuda.is_available(), "A CUDA device is required on evaluator workers"
print("WorldSimProbe evaluator dependencies are available")
PY
```

Successful imports confirm the environment, not the model weights. The task
scripts validate checkpoint paths when evaluation starts.
