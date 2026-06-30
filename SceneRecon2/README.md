# SceneRecon2

SceneRecon2 is a fresh reconstruction workspace for turning WorldArena /
RoboTwin first-frame observations into editable SAPIEN scenes.

The design goal is not full automation on day one. The core loop should support
both:

- automatic algorithms that propose or refine object placement;
- a human operator who can move, rotate, scale, swap, hide, or lock objects.

Both paths should call the same scene editing API, so manual corrections can
become initialization, supervision, or debugging data for later automation.

## First Principle

Keep three things separate:

1. **Scene state**: the current task, camera, robot, table, and object
   hypotheses.
2. **Render service**: a long-lived SAPIEN process that can quickly update
   object transforms and return images, masks, and scores.
3. **Strategies**: task-specific logic that decides which models may appear and
   where they should be initialized.

The old SceneRecon mixed exploration scripts, reports, and rendering loops
together. SceneRecon2 should make the interactive and automatic paths feel like
two frontends over the same machinery.

## Target Workflow

For one episode:

1. Load the target first frame and metadata.
2. Infer or select the top-1 RoboTwin task.
3. Ask that task strategy for candidate objects and candidate models.
4. Detect visible objects in 2D using task-specific prompts, YOLO, masks, or
   vision-language models.
5. Pick initial model candidates and approximate 3D positions.
6. Start a persistent render scene.
7. Let automatic search refine poses and model choices.
8. Let the user manually adjust anything that is wrong.
9. Save a reconstruction state file that can be reopened, rendered, optimized,
   or used as pseudo-ground-truth.

## Near-Term Milestones

1. Define a reconstruction state schema.
2. Build a minimal render service API with persistent scenes.
3. Build a local web editor for object transforms and model selection.
4. Implement one simple task end-to-end.
5. Add automatic pose search as a client of the same edit API.
6. Add model selection loops.
7. Expand task strategies one by one.

## Directory Plan

```text
SceneRecon2/
  README.md
  docs/
    plan.md
  scenerecon2/
    state.py
    assets.py
    camera.py
    render_service.py
    scoring.py
    strategies/
      base.py
      place_empty_cup.py
      click_bell.py
    web/
      server.py
      static/
  outputs/
    ...
```

This tree is a plan, not a requirement that every file exists immediately.
Implementation should stay incremental.

## Current Vertical MVP

The first runnable slice is a hard-coded `place_empty_cup` editor:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export TORCH_CUDA_ARCH_LIST=12.0

python -m SceneRecon2.place_empty_cup_editor --host 127.0.0.1 --port 8787
```

Then open:

```text
http://127.0.0.1:8787
```

Optional target image override:

```bash
python -m SceneRecon2.place_empty_cup_editor \
  --host 127.0.0.1 \
  --port 8787 \
  --first-frame /path/to/episode.png
```

Interactive rendering defaults to a lightweight transport path:

- the editor keeps RoboTwin's ray-tracing shader path;
- target/render images are served as in-memory JPEG instead of disk PNG;
- displayed images keep the original size by default, `--display-scale 1.0`.

Useful knobs:

```bash
python -m SceneRecon2.place_empty_cup_editor \
  --host 127.0.0.1 \
  --port 8787 \
  --display-scale 1.0 \
  --jpeg-quality 95
```

Current limitations are intentional:

- only `place_empty_cup`;
- only `cup` and `coaster`;
- only table `x/y` editing;
- editor rendering is a static no-physics posing mode;
- no save yet;
- no automatic refine yet;
- no module cleanup yet.

Current interaction contract:

- hold Space to compare the right panel against the target first frame;
- use arrow keys to move the active object on the table plane;
- use `1` / `2` or Tab to switch active object;
- use `C` to switch to camera editing;
- in camera mode, arrow left/right edits camera `y`, arrow up/down edits
  camera `z`, and `Q` / `E` edits pitch;
- hold `[` for coarse movement;
- hold `]` for fine movement;
- press `-` to save the current reconstruction;
- `+` and `0` are reserved for next/previous episode;
- rotation and scale controls are reserved but not implemented yet.
