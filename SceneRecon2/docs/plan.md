# SceneRecon2 Plan

## Why Restart

The first SceneRecon pass proved that the idea is plausible:

- first-frame object hypotheses can be projected onto the table plane;
- SAPIEN render-search can improve placement;
- different tasks need different object logic;
- visualization is the main bottleneck for judgment and debugging.

The weak point is product shape. Reports and scripts are too deep in the file
tree, while the user needs a direct visual workbench: target image on one side,
current render on the other, controls in reach, and automatic search available
as a button rather than a hidden batch process.

SceneRecon2 should therefore start from the interface and state model.

## Core Product Shape

The central object is an editable reconstruction:

```text
target image + task guess + camera + robot state + object hypotheses
```

Every object hypothesis should contain:

- semantic role, for example `cup`, `coaster`, `bell`, `hammer`;
- selected asset/model id;
- candidate asset/model ids;
- pose: `x`, `y`, `z`, `roll`, `pitch`, `yaw`;
- optional scale, if the asset pipeline supports it safely;
- material overrides, especially color;
- lock flags for fields the user trusts;
- source information, for example detector, VLM, manual, or optimizer;
- score/debug fields.

Manual edits and automatic edits should both produce patches to this state.

## Architecture

### 1. State Layer

Responsible for serializing and validating reconstruction state.

Candidate files:

```text
scenerecon2/state.py
```

Useful operations:

- create empty state from dataset episode;
- add object hypothesis;
- update transform;
- update selected model;
- lock or unlock fields;
- save and load JSON.

The state should not import SAPIEN. It should be plain Python data.

### 2. Task Strategy Layer

Responsible for task-specific assumptions.

Candidate files:

```text
scenerecon2/strategies/base.py
scenerecon2/strategies/place_empty_cup.py
scenerecon2/strategies/click_bell.py
```

Each strategy should answer:

- what object roles may appear;
- what RoboTwin assets/models are valid for each role;
- which roles are movable, fixed, stacked, or constrained;
- how to initialize from detections;
- which fields are coupled, if any;
- what search ranges are allowed.

This is where task-specific logic belongs. The render service and editor should
not know that `place_empty_cup` has a cup and coaster, or that `click_bell`
needs a particular bell family.

### 3. Asset Layer

Responsible for listing RoboTwin assets and mapping semantic roles to candidate
models.

Candidate files:

```text
scenerecon2/assets.py
```

Initial asset selection can be simple:

1. task strategy provides legal model ids;
2. vision/detection provides rough class and shape hints;
3. render loop tests the top candidates;
4. user can override model id in the web UI.

Model selection should be a first-class action, not hidden inside pose search.

### 4. Geometry Layer

Responsible for camera math and coarse placement.

Candidate files:

```text
scenerecon2/camera.py
scenerecon2/geometry.py
```

Initial placement should use:

- known RoboTwin camera intrinsics/extrinsics when available;
- table plane intersection;
- bbox bottom-center for upright objects;
- bbox center for flat objects;
- task-specific height priors.

This gives a good first guess before any render-search.

### 5. Persistent Render Service

Responsible for keeping a SAPIEN scene alive.

Candidate files:

```text
scenerecon2/render_service.py
```

The important rule:

```text
frequent loop = set object transforms -> render -> score
```

Avoid rebuilding the whole scene for every candidate.

Expected API shape:

```text
load_episode(state)
apply_patch(patch)
render_rgb()
render_masks()
score_against_target()
snapshot()
```

Changing pose should be cheap. Changing model or scale may require actor
recreation, so those actions should be less frequent and explicit.

### 6. Scoring Layer

Responsible for comparing current render with target image.

Candidate files:

```text
scenerecon2/scoring.py
```

Scores should be composable:

- object bbox overlap;
- object mask overlap;
- edge alignment;
- color consistency;
- whole-image similarity;
- task-specific penalties.

The UI should show score components, not only one final number, because a single
number is hard to debug.

### 7. Web Editor

Responsible for making reconstruction visible and editable.

Candidate files:

```text
scenerecon2/web/server.py
scenerecon2/web/static/
```

Required first screen:

- left: target first frame;
- right: current render;
- overlay mode: target/render/difference/mask/bbox;
- object list with selected model and score;
- transform controls for selected object;
- buttons for auto-place, pose-search, model-search, save, reload.

The editor should not contain task logic. It should call strategy/render APIs.

## Manual And Automatic Editing

The key interface is a patch:

```json
{
  "object_id": "cup",
  "updates": {
    "pose.x": 0.12,
    "pose.yaw": 1.57
  },
  "source": "manual"
}
```

Automatic search sends the same kind of patch:

```json
{
  "object_id": "cup",
  "updates": {
    "pose.x": 0.118,
    "pose.y": -0.044
  },
  "source": "pose_search"
}
```

This keeps the system simple:

- manual edits are not special;
- automatic edits are inspectable;
- undo/history becomes possible;
- saved states can train or evaluate later automation.

## Model Selection

Model selection should happen before fine pose search, but remain editable.

Suggested flow:

1. For each object role, strategy lists legal candidate models.
2. Detection/VLM filters candidates by visual hints.
3. Render service tests a small top-k set.
4. Best model is selected.
5. User may manually switch model.
6. Pose search runs with selected model fixed.

If scale is needed, treat it like model selection:

- coarse discrete scale candidates first;
- then pose optimization;
- avoid continuous scale updates until we know SAPIEN supports it cleanly.

## Task Strategy Examples

### place_empty_cup

Objects:

- cup;
- coaster or target receptacle.

Likely initialization:

- detect cup body and coaster;
- use bbox bottom-center for cup table position;
- use bbox center for flat coaster;
- search cup and coaster independently unless evidence says they are coupled.

### click_bell

Objects:

- bell body/button.

Likely initialization:

- model choice is important;
- start with known bell model candidates from RoboTwin;
- use bbox center and table-plane projection;
- allow yaw search and small xy refinement.

### beat_block_hammer

Objects:

- hammer;
- block;
- possibly target area.

Likely initialization:

- hammer model choice matters;
- pose includes yaw and maybe roll/pitch if lying down;
- bbox shape and handle/head color can help model selection.

## What To Build First

Build the smallest useful loop:

1. JSON state schema.
2. `place_empty_cup` strategy.
3. persistent render service for that task.
4. web editor with target/render side by side.
5. object transform controls.
6. save/load reconstruction JSON.
7. one automatic `refine selected object` button.

After this works, add:

- model selection UI;
- model-search button;
- more tasks;
- detector and VLM integrations.

## Explicit Non-Goals For The First Version

- no full 50-task reconstruction immediately;
- no differentiable renderer;
- no perfect automation before the editor exists;
- no hidden batch-only workflow;
- no changes to old `SceneRecon` until SceneRecon2 has replaced a clear part of it.

