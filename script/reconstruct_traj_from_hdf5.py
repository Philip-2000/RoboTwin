import argparse
import importlib
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

sys.path.append("./")

import h5py
import numpy as np
import yaml

from envs._GLOBAL_CONFIGS import CONFIGS_PATH


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def load_embodiment_config(robot_file):
    with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def get_robot_file(embodiment_name):
    embodiment_types = load_yaml(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"))
    robot_file = embodiment_types[embodiment_name]["file_path"]
    if robot_file is None:
        raise ValueError(f"Missing embodiment file for {embodiment_name}")
    return robot_file


def task_from_dataset_dir(dataset_dir):
    path = Path(dataset_dir).resolve()
    if path.name.endswith("_clean-50") or path.name.endswith("_randomized-500"):
        return path.parent.name
    return path.name


def load_seed(dataset_dir, episode):
    seeds = [int(x) for x in (Path(dataset_dir) / "seed.txt").read_text().split()]
    return seeds[episode]


def load_joint_keyframes(dataset_dir, episode, dataset):
    hdf5_path = Path(dataset_dir) / "data" / f"episode{episode}.hdf5"
    with h5py.File(hdf5_path, "r") as f:
        qpos = np.asarray(f[dataset][:], dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != 14:
        raise ValueError(f"Expected HDF5 {dataset} to have shape (T, 14), got {qpos.shape}")
    return qpos, hdf5_path


def interpolate_keyframes(qpos, steps_per_frame):
    dense = []
    for idx in range(qpos.shape[0] - 1):
        start = qpos[idx]
        end = qpos[idx + 1]
        for step in range(1, steps_per_frame + 1):
            alpha = step / steps_per_frame
            dense.append(start * (1.0 - alpha) + end * alpha)
    return np.asarray(dense, dtype=np.float64)


def dense_to_traj(dense, control_dt=1.0):
    left_pos = dense[:, :6]
    right_pos = dense[:, 7:13]
    left_vel = np.gradient(left_pos, axis=0) / control_dt if len(left_pos) > 1 else np.zeros_like(left_pos)
    right_vel = np.gradient(right_pos, axis=0) / control_dt if len(right_pos) > 1 else np.zeros_like(right_pos)
    return {
        "left_joint_path": [
            {
                "status": "Success",
                "position": left_pos,
                "velocity": left_vel,
            }
        ],
        "right_joint_path": [
            {
                "status": "Success",
                "position": right_pos,
                "velocity": right_vel,
            }
        ],
    }


def build_args(task_name, task_config, dataset_dir, output_dir, episode):
    args = load_yaml(f"./task_config/{task_config}.yml")
    args["task_name"] = task_name
    args["task_config"] = task_config
    args["save_path"] = str(output_dir)
    args["save_data"] = True
    args["need_plan"] = False
    args["render_freq"] = 0
    args["collect_data"] = True
    args["eval_video_log"] = False
    args["now_ep_num"] = episode
    args["seed"] = load_seed(dataset_dir, episode)

    embodiment = args.get("embodiment", ["aloha-agilex"])
    if len(embodiment) != 1:
        raise ValueError("This helper currently supports single-embodiment dual-arm configs only.")

    robot_file = get_robot_file(embodiment[0])
    args["left_robot_file"] = robot_file
    args["right_robot_file"] = robot_file
    args["left_embodiment_config"] = load_embodiment_config(robot_file)
    args["right_embodiment_config"] = load_embodiment_config(robot_file)
    args["dual_arm_embodied"] = True
    args["embodiment_name"] = embodiment[0]
    return args


def set_robot_state(task, qpos):
    task.robot.set_arm_joints(qpos[:6], np.zeros(6), "left")
    task.robot.set_arm_joints(qpos[7:13], np.zeros(6), "right")
    task.robot.set_gripper(qpos[6], "left", gripper_eps=0)
    task.robot.set_gripper(qpos[13], "right", gripper_eps=0)


def replay_reconstructed(dataset_dir, output_dir, task_name, task_config, episode, steps_per_frame, dataset, keep_cache):
    qpos, source_hdf5 = load_joint_keyframes(dataset_dir, episode, dataset)
    dense = interpolate_keyframes(qpos, steps_per_frame)
    traj = dense_to_traj(dense)

    output_dir = Path(output_dir)
    traj_dir = output_dir / "_reconstructed_traj_data"
    traj_dir.mkdir(parents=True, exist_ok=True)
    traj_path = traj_dir / f"episode{episode}.pkl"
    with open(traj_path, "wb") as f:
        pickle.dump(traj, f)

    env_module = importlib.import_module(f"envs.{task_name}")
    task = getattr(env_module, task_name)()
    args = build_args(task_name, task_config, dataset_dir, output_dir, episode)

    try:
        task.setup_demo(**args)
        set_robot_state(task, qpos[0])
        task.scene.step()
        task._update_render()
        task._take_picture()

        dense_idx = 0
        for frame_idx in range(qpos.shape[0] - 1):
            for _ in range(steps_per_frame):
                now = dense[dense_idx]
                prev = qpos[frame_idx] if dense_idx == 0 else dense[dense_idx - 1]
                vel = now - prev
                task.robot.set_arm_joints(now[:6], vel[:6], "left")
                task.robot.set_arm_joints(now[7:13], vel[7:13], "right")
                task.robot.set_gripper(now[6], "left", gripper_eps=0)
                task.robot.set_gripper(now[13], "right", gripper_eps=0)
                task.scene.step()
                dense_idx += 1
            task._update_render()
            task._take_picture()

        task.merge_pkl_to_hdf5_video()

        metadata_path = output_dir / "reconstruct_from_hdf5_metadata.json"
        metadata = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
        metadata[f"episode_{episode}"] = {
            "task_name": task_name,
            "task_config": task_config,
            "source_dataset_dir": str(Path(dataset_dir).resolve()),
            "source_hdf5": str(source_hdf5),
            "source_dataset": dataset,
            "seed": args["seed"],
            "steps_per_frame": steps_per_frame,
            "keyframes": int(qpos.shape[0]),
            "dense_steps": int(dense.shape[0]),
            "reconstructed_traj": str(traj_path),
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        cache = getattr(task, "folder_path", {}).get("cache")
        task.close_env(clear_cache=True)
        if cache and (not keep_cache) and Path(cache).exists():
            shutil.rmtree(cache)


def main():
    parser = argparse.ArgumentParser(
        description="Approximate dense RoboTwin trajectory from HDF5 joint_action keyframes and replay it."
    )
    parser.add_argument("dataset_dir", help="Path like .../<task>/aloha-agilex_clean-50")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episode", type=int, action="append", required=True)
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--dataset", default="joint_action/vector")
    parser.add_argument("--steps-per-frame", type=int, default=15)
    parser.add_argument("--keep-cache", action="store_true")
    args = parser.parse_args()

    task_name = args.task_name or task_from_dataset_dir(args.dataset_dir)
    for episode in args.episode:
        print(f"Reconstructing {task_name} episode{episode} from HDF5 -> {args.output_dir}")
        replay_reconstructed(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            task_name=task_name,
            task_config=args.task_config,
            episode=episode,
            steps_per_frame=args.steps_per_frame,
            dataset=args.dataset,
            keep_cache=args.keep_cache,
        )


if __name__ == "__main__":
    main()
