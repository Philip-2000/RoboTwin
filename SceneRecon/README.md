# SceneRecon

SceneRecon is a standalone scene-reconstruction workspace for WorldArena_RoboTwin2.0.
It intentionally lives outside the RoboTwin task/environment modules so reconstruction
experiments do not leak into simulator code.

## Design

- `core.py`: shared dataclasses for episodes, priors, candidates, and reports.
- `dataset.py`: WorldArena file indexing and HDF5/instruction loading.
- `task_mapping.py`: task-level top1 mapping from `yl_outputs/search_gt`.
- `registry.py`: strategy registry.
- `strategies/`: task-specific reconstruction priors.

The intended dependency direction is:

```text
CLI / runners
  -> dataset + task_mapping + registry
  -> one task strategy
  -> shared core types
```

Task strategies should not import each other. Shared utilities should go into common
modules only when at least two strategies need them.

## First Tasks

Start with low-dimensional scenes:

1. `click_bell`
2. `place_empty_cup`
3. `stack_blocks_three`
4. `beat_block_hammer`
5. `place_object_basket`

Harder scenes such as multi-bottle/bin tasks can be added after the first-frame fitting
loop is stable.

## Dry Run

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin

python -m SceneRecon.cli inspect \
  --worldarena-root /home/users/liang01.yue/D/WorldArena_Robotwin2.0 \
  --worldarena-split val_dataset \
  --episode 1
```

Write a reconstruction report:

```bash
python -m SceneRecon.cli reconstruct \
  --worldarena-root /home/users/liang01.yue/D/WorldArena_Robotwin2.0 \
  --worldarena-split val_dataset \
  --episode 1 \
  --output-root /home/users/liang01.yue/D/WorldArena_Robotwin2.0/scene_recon
```

## Detection Input

SceneRecon can already consume external detections without depending on a specific
model. Create a template:

```bash
python -m SceneRecon.cli make-detection-template \
  --worldarena-root /home/users/liang01.yue/D/WorldArena_Robotwin2.0 \
  --worldarena-split val_dataset \
  --episode 37 \
  --out /tmp/episode37_detections.json
```

Edit the JSON:

```json
{
  "image_path": ".../episode37.png",
  "source": "manual",
  "detections": [
    {"label": "cup", "bbox_xyxy": [120, 80, 160, 135], "score": 0.95},
    {"label": "coaster", "bbox_xyxy": [180, 120, 225, 150], "score": 0.90}
  ]
}
```

Then run:

```bash
python -m SceneRecon.cli reconstruct \
  --worldarena-root /home/users/liang01.yue/D/WorldArena_Robotwin2.0 \
  --worldarena-split val_dataset \
  --episode 37 \
  --detections-json /tmp/episode37_detections.json \
  --output-root /home/users/liang01.yue/D/WorldArena_Robotwin2.0/scene_recon
```

## Model Adapters

The model-specific files are placeholders by design:

- `detectors/yolo_world.py`
- `vlm/qwen_vl.py`

After weights are downloaded, those adapters should translate model outputs into
the shared `DetectionSet` type. Task strategies should not import YOLO/Qwen code
directly; they should only consume `DetectionSet`.
