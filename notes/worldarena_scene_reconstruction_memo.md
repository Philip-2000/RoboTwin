# WorldArena / RoboTwin2.0 Scene Reconstruction Memo

## Background

WorldArena Track 1 provides, for each episode:

- first frame: `first_frame/fixed_scene_task/episodeN.png`
- instruction variants: `instructions/`, `instructions_1/`, `instructions_2/`
- action/end-effector trajectory: `data/fixed_scene_task/episodeN.hdf5`

The public WorldArena test HDF5 files do not contain object poses, task names, random seeds, camera intrinsics/extrinsics, or full RGB observation sequences. They mainly contain:

- `joint_action/vector`
- `joint_action/{left_arm,left_gripper,right_arm,right_gripper}`
- `endpose/{left_endpose,left_gripper,right_endpose,right_gripper}`
- empty `pointcloud`

By contrast, the original RoboTwin2.0 Clean-50 HDF5 files include RGB observations and camera matrices, but still do not directly expose every object pose in a simple public metadata field.

The practical conclusion is that the official WorldArena test scene state is not directly recoverable from released files. Scene reconstruction must be treated as an inverse problem using first-frame vision, task priors, and RoboTwin initialization code.

## Key Observation

RoboTwin task initialization is highly structured. Each task has a `load_actors()` function in `envs/<task>.py`. Most tasks instantiate objects with patterns like:

```python
pose = rand_pose(
    xlim=[...],
    ylim=[...],
    zlim=[...],
    qpos=[...],
    rotate_rand=True,
    rotate_lim=[...],
)

actor = create_actor(
    scene=self,
    pose=pose,
    modelname="...",
    model_id=...,
    convex=True,
    is_static=True,
)
```

This means scene reconstruction is not unrestricted 3D reconstruction. It is closer to parameter inversion inside a known low-dimensional task template:

```text
task_name
object model_id
object x/y/z/yaw
some task-specific discrete choices
camera/table alignment
```

## Asset Priors

RoboTwin assets live under:

```text
assets/objects/
```

There are roughly:

- 120 numbered object categories
- about 580 model variants, counted from `base{id}` / `model_data{id}.json`

Each `model_data*.json` may include:

- `scale`
- `extents`
- `center`
- `contact_points_pose`
- `functional_matrix`
- target/functional/contact point descriptions

Important detail: `create_actor(..., scale=...)` accepts a runtime `scale`, but when `model_data*.json` exists, `create_actor()` loads `model_data["scale"]` and overwrites the passed scale. Therefore most object sizes should initially be treated as fixed by model variant, not freely optimized.

Colors are usually only runtime-set for programmatic geometry such as `create_box(...)`, pads, colored blocks, and target regions. GLB/OBJ asset colors are generally from the mesh/material itself.

## Representative Task Priors

### `click_bell`

Initialization:

- object: `050_bell`
- `model_id in {0, 1}`
- `x in [-0.25, 0.25]`
- `y in [-0.2, 0.0]`
- constraint: `abs(x) >= 0.05`
- orientation fixed by `qpos=[0.5, 0.5, 0.5, 0.5]`

This is one of the easiest reconstruction targets: one object, small discrete model set, clear instruction, clear visual target.

### `place_empty_cup`

Initialization:

- cup: `021_cup/base0`
- coaster: `019_coaster/base0`
- cup appears on one side:
  - right: `x in [0.15, 0.3]`
  - left: `x in [-0.3, -0.15]`
- coaster appears near center on the same side family:
  - `x in [-0.05, 0.1]` or `[-0.1, 0.05]`
- both use `y in [-0.2, 0.05]`
- cup/coaster cannot be too close initially

This is a good early target because objects are visually distinct and task logic is simple.

### `place_object_basket`

Initialization:

- basket: `110_basket`, usually `model_id in {0, 1}` for this task
- object: `081_playingcards` or `057_toycar`
- if using left arm:
  - basket near `x = 0.02`, `y in [-0.08, -0.05]`
  - object on left: `x in [-0.25, -0.2]`, `y in [-0.1, 0.1]`
- if using right arm:
  - basket near `x = -0.02`, `y in [-0.08, -0.05]`
  - object on right: `x in [0.2, 0.25]`, `y in [-0.1, 0.1]`

The layout prior is very strong. First-frame detection can likely infer both arm side and object side.

### `put_bottles_dustbin`

Initialization:

- three bottles: `114_bottle/base1`, `base2`, `base3`
- bottle placement:
  - `x in [-0.25, 0.3]`
  - `y in [0.03, 0.23]`
  - `abs(x) >= 0.05`
  - pairwise distance at least about `0.13m`
- dustbin fixed:
  - `011_dustbin`
  - pose around `[-0.45, 0, 0]`

This is useful but harder: multiple similar objects, possible occlusion, object identity matching matters.

### `stack_blocks_three`

Initialization:

- three programmatic colored boxes:
  - red
  - green
  - blue
- each block:
  - `x in [-0.28, 0.28]`
  - `y in [-0.08, 0.05]`
  - `z = 0.741 + block_half_size`
  - random yaw within `rotate_lim=[0, 0, 0.75]`
- constraints:
  - `abs(x) >= 0.05`
  - blocks separated by about `0.1m`
  - avoid target stacking point near `[0, -0.1]`
- target stack location fixed near `[0, -0.13]`

This is good for first-frame reconstruction because colored blocks are easy to detect.

## Proposed Reconstruction Pipeline

1. Classify task from instruction

   Use lexical rules first, then optionally an LLM/VLM classifier. Map the instruction to RoboTwin task names such as `click_bell`, `place_empty_cup`, `stack_blocks_three`, etc.

2. Load task template priors

   For each task, extract or hand-code a compact schema:

   ```json
   {
     "task": "click_bell",
     "objects": [
       {
         "modelname": "050_bell",
         "model_ids": [0, 1],
         "xlim": [-0.25, 0.25],
         "ylim": [-0.2, 0.0],
         "constraints": ["abs(x) >= 0.05"]
       }
     ]
   }
   ```

3. Detect/segment objects in the first frame

   Prefer open-vocabulary tools such as GroundingDINO + SAM/SAM2 or YOLO-World, because RoboTwin has many object categories and visual variants.

4. Estimate table-plane coordinates

   Because the camera/workspace is nearly fixed, use a manually calibrated homography:

   ```text
   image pixel (u, v) -> table coordinate (x, y)
   ```

   Start with a small number of known table landmarks or manually chosen correspondences.

5. Fit scene parameters

   Search or optimize over:

   - object `model_id`
   - object `(x, y)`
   - yaw, where relevant
   - discrete task choices such as left/right side
   - small camera/table alignment offsets

   Render candidate first frames and compare against the WorldArena first frame with image-level and segmentation-level losses.

6. Replay WorldArena action trajectory

   Once an approximate scene is reconstructed, use the released `joint_action/vector` to drive the RoboTwin robot and render the video. The existing `script/render_joint_video.py` already proves the action-replay path works.

7. Use reconstructed videos as pseudo-GT

   The output can support:

   - pseudo ground-truth videos for world model training
   - scene metadata for state-conditioned training
   - additional simulation rollouts
   - debugging action-conditioned generation

## Recommended Initial Tasks

Start with tasks that have few objects and strong priors:

1. `click_bell`
2. `place_empty_cup`
3. `stack_blocks_three`
4. `beat_block_hammer`
5. `place_object_basket`

Then expand to harder multi-object or articulated tasks:

1. `put_bottles_dustbin`
2. `place_cans_plasticbox`
3. `open_laptop`
4. `open_microwave`
5. `put_object_cabinet`

## Risks And Limits

- A single first frame cannot reliably recover occluded objects, hidden state, or articulated joint state.
- Some tasks require exact functional points or contact geometry; rough 2D placement may not be enough.
- Similar model variants may be visually ambiguous.
- Public WorldArena test data does not include official object poses or seeds, so this reconstruction is pseudo-GT, not official GT.
- If generated pseudo-GT is used for training, it may bias the world model toward RoboTwin's scripted/planned behavior.

## Practical Next Step

Implement a small prototype for `click_bell`:

1. Parse WorldArena instruction and select `click_bell`.
2. Detect bell in `first_frame`.
3. Estimate table `(x, y)` from pixel center with a homography.
4. Try `model_id=0` and `model_id=1`.
5. Render candidate first frames in RoboTwin.
6. Select the best candidate by image/segmentation similarity.
7. Replay the given `joint_action/vector` and render a pseudo-GT video.

If this works visually, repeat for `place_empty_cup` and `stack_blocks_three`.
