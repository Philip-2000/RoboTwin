import argparse
import importlib
import os
import sys

sys.path.append("./")

import h5py
import imageio.v2 as imageio
import numpy as np
import yaml

from envs._GLOBAL_CONFIGS import CONFIGS_PATH


def load_embodiment(embodiment):
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    robot_file = embodiment_types[embodiment]["file_path"]
    with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
        embodiment_config = yaml.load(f.read(), Loader=yaml.FullLoader)
    return robot_file, embodiment_config


def load_actions(hdf5_path, dataset):
    with h5py.File(hdf5_path, "r") as f:
        actions = np.asarray(f[dataset], dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 14:
        raise ValueError(f"Expected actions with shape (T, 14), got {actions.shape}")
    return actions


def build_task_env(task_name, embodiment, camera_name):
    env_module = importlib.import_module(f"envs.{task_name}")
    task = getattr(env_module, task_name)()
    robot_file, embodiment_config = load_embodiment(embodiment)

    kwargs = {
        "seed": 0,
        "task_name": task_name,
        "now_ep_num": 0,
        "render_freq": 0,
        "save_path": "./data/render_joint_video",
        "save_freq": 1,
        "data_type": {},
        "domain_randomization": {
            "random_background": False,
            "cluttered_table": False,
            "clean_background_rate": 1,
            "random_head_camera_dis": 0,
            "random_table_height": 0,
            "random_light": False,
            "crazy_random_light_rate": 0,
        },
        "camera": {
            "head_camera_type": "D435",
            "wrist_camera_type": "D435",
            "collect_head_camera": camera_name == "head_camera",
            "collect_wrist_camera": camera_name in ("left_camera", "right_camera"),
        },
        "left_robot_file": robot_file,
        "right_robot_file": robot_file,
        "left_embodiment_config": embodiment_config,
        "right_embodiment_config": embodiment_config,
        "dual_arm_embodied": True,
        "pcd_crop": False,
        "pcd_down_sample_num": 0,
    }
    task.setup_demo(**kwargs)
    return task


def _gripper_joint_targets(robot, gripper_val, arm_tag):
    gripper_val = np.clip(gripper_val, 0, 1)
    if arm_tag == "left":
        joints = robot.left_gripper
        gripper_scale = robot.left_gripper_scale
        robot.left_gripper_val = gripper_val
    else:
        joints = robot.right_gripper
        gripper_scale = robot.right_gripper_scale
        robot.right_gripper_val = gripper_val

    real_gripper_val = gripper_scale[0] + gripper_val * (gripper_scale[1] - gripper_scale[0])
    return [(joint[0], real_gripper_val * joint[1] + joint[2]) for joint in joints]


def set_robot_state(task, action, velocity=None):
    left_arm = action[:6]
    left_gripper = action[6]
    right_arm = action[7:13]
    right_gripper = action[13]
    if velocity is None:
        velocity = np.zeros_like(action)

    robot = task.robot
    robot.set_arm_joints(left_arm, velocity[:6], "left")
    robot.set_arm_joints(right_arm, velocity[7:13], "right")
    left_gripper_targets = _gripper_joint_targets(robot, left_gripper, "left")
    right_gripper_targets = _gripper_joint_targets(robot, right_gripper, "right")

    # set_arm_joints only sets drive targets. For frame-accurate HDF5 replay we also
    # teleport the articulation state to the saved qpos; otherwise the first frames
    # can show drive-lag or stale state from the previous episode.
    entities = []
    for entity in (robot.left_entity, robot.right_entity):
        if entity not in entities:
            entities.append(entity)

    for entity in entities:
        active_joints = entity.get_active_joints()
        qpos = entity.get_qpos()
        qvel = np.zeros_like(qpos)

        for joints, positions, velocities in (
            (robot.left_arm_joints, left_arm, velocity[:6]),
            (robot.right_arm_joints, right_arm, velocity[7:13]),
        ):
            for joint, position, joint_velocity in zip(joints, positions, velocities):
                if joint in active_joints:
                    joint_idx = active_joints.index(joint)
                    qpos[joint_idx] = position
                    qvel[joint_idx] = joint_velocity

        for joint, target in left_gripper_targets + right_gripper_targets:
            if joint in active_joints:
                joint_idx = active_joints.index(joint)
                qpos[joint_idx] = target
                qvel[joint_idx] = 0
                joint.set_drive_target(target)
                joint.set_drive_velocity_target(0)

        entity.set_qpos(qpos)
        if hasattr(entity, "set_qvel"):
            entity.set_qvel(qvel)


def apply_action(task, action, velocity, substeps):
    set_robot_state(task, action, velocity)
    for _ in range(substeps):
        task.scene.step()


def capture_rgb(task, camera_name):
    robot = task.robot
    cameras = task.cameras
    cameras.update_wrist_camera(robot.left_camera.get_pose(), robot.right_camera.get_pose())
    task.scene.update_render()
    cameras.update_picture()
    return cameras.get_rgb()[camera_name]["rgb"]


def default_output_path(hdf5_path):
    abs_path = os.path.abspath(hdf5_path)
    parts = abs_path.split(os.sep)
    if "data" not in parts:
        stem, _ = os.path.splitext(abs_path)
        return f"{stem}.mp4"

    data_idx = len(parts) - 1 - parts[::-1].index("data")
    out_parts = parts[:]
    out_parts[data_idx] = "simulation"
    out_parts[-1] = os.path.splitext(out_parts[-1])[0] + ".mp4"
    return os.sep.join(out_parts)


def load_background(background_path, shape):
    if background_path is None:
        return None
    bg = imageio.imread(background_path)
    if bg.ndim == 2:
        bg = np.repeat(bg[:, :, None], 3, axis=2)
    if bg.shape[2] == 4:
        bg = bg[:, :, :3]
    if bg.shape[:2] != shape[:2]:
        raise ValueError(f"Background shape {bg.shape[:2]} does not match frame shape {shape[:2]}")
    return bg.astype(np.float32)


def composite_green(frame, background, hard=35.0, soft=80.0):
    if background is None:
        return frame

    frame_f = frame.astype(np.float32)
    key = np.array([0, 255, 0], dtype=np.float32)
    dist = np.linalg.norm(frame_f - key, axis=2)
    alpha = np.clip((dist - hard) / soft, 0, 1)

    greenish = (
        (frame_f[:, :, 1] > frame_f[:, :, 0] * 1.25)
        & (frame_f[:, :, 1] > frame_f[:, :, 2] * 1.25)
        & (frame_f[:, :, 1] > 70)
    )
    alpha[greenish] = np.minimum(
        alpha[greenish],
        np.clip((frame_f[:, :, 1][greenish] - 70) / 160, 0, 1) * 0.15,
    )
    alpha = alpha[:, :, None]

    fg = frame_f.copy()
    fg[:, :, 1] = np.minimum(fg[:, :, 1], np.maximum(fg[:, :, 0], fg[:, :, 2]) + 35)
    comp = fg * alpha + background * (1 - alpha)
    return np.clip(comp, 0, 255).astype(np.uint8)


def background_for_hdf5(hdf5_path, data_root, background_root):
    if background_root is None:
        return None
    rel = os.path.relpath(os.path.abspath(hdf5_path), os.path.abspath(data_root))
    return os.path.join(background_root, os.path.splitext(rel)[0] + ".png")


def render_one(task, hdf5_path, args):
    actions = load_actions(hdf5_path, args.dataset)
    if args.max_frames is not None:
        actions = actions[:args.max_frames]
    velocities = np.zeros_like(actions)
    if len(actions) > 1:
        velocities[1:] = actions[1:] - actions[:-1]

    out = args.out
    if out is None:
        out = default_output_path(hdf5_path)
    elif len(args.hdf5_path) > 1:
        raise ValueError("--out can only be used with a single input hdf5")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    background_path = args.background
    if background_path is None:
        background_path = background_for_hdf5(hdf5_path, args.data_root, args.background_root)

    background = None
    writer = imageio.get_writer(out, fps=args.fps, codec="libx264", quality=8)
    try:
        for idx, action in enumerate(actions):
            apply_action(task, action, velocities[idx], args.substeps)
            frame = capture_rgb(task, args.camera)
            if background is None:
                background = load_background(background_path, frame.shape)
            frame = composite_green(frame, background)
            writer.append_data(frame)
            if (idx + 1) % 50 == 0 or idx + 1 == len(actions):
                print(f"{os.path.basename(hdf5_path)}: rendered {idx + 1}/{len(actions)}", end="\r")
    finally:
        writer.close()

    print(f"\nVideo saved to {out}, containing {len(actions)} frames at {args.fps} FPS.")


def main():
    parser = argparse.ArgumentParser(description="Render RoboTwin joint-action HDF5 files as MP4 videos.")
    parser.add_argument("hdf5_path", nargs="+")
    parser.add_argument("--dataset", default="joint_action/vector")
    parser.add_argument("--out", default=None)
    parser.add_argument("--background", default=None)
    parser.add_argument("--background-root", default=None)
    parser.add_argument(
        "--data-root",
        default="/home/users/liang01.yue/D/WorldArena_Robotwin2.0/val_dataset/data",
    )
    parser.add_argument("--camera", default="head_camera", choices=["head_camera", "left_camera", "right_camera"])
    parser.add_argument("--embodiment", default="aloha-agilex")
    parser.add_argument("--task", default="empty_green_table")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    task = build_task_env(args.task, args.embodiment, args.camera)
    try:
        for hdf5_path in args.hdf5_path:
            render_one(task, hdf5_path, args)
    finally:
        task.close_env(clear_cache=True)


if __name__ == "__main__":
    main()
