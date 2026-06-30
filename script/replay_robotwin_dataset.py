import argparse
import importlib
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

sys.path.append("./")

import yaml

from envs._GLOBAL_CONFIGS import CONFIGS_PATH


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def load_embodiment_config(robot_file):
    with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def get_robot_file(embodiment_name):
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    embodiment_types = load_yaml(embodiment_config_path)
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
    seed_path = Path(dataset_dir) / "seed.txt"
    seeds = [int(x) for x in seed_path.read_text().split()]
    return seeds[episode]


def load_traj(dataset_dir, episode):
    traj_path = Path(dataset_dir) / "_traj_data" / f"episode{episode}.pkl"
    with open(traj_path, "rb") as f:
        return pickle.load(f)


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

    embodiment = args.get("embodiment", ["aloha-agilex"])
    if len(embodiment) != 1:
        raise ValueError("This replay helper currently supports single-embodiment dual-arm configs only.")

    robot_file = get_robot_file(embodiment[0])
    args["left_robot_file"] = robot_file
    args["right_robot_file"] = robot_file
    args["left_embodiment_config"] = load_embodiment_config(robot_file)
    args["right_embodiment_config"] = load_embodiment_config(robot_file)
    args["dual_arm_embodied"] = True
    args["embodiment_name"] = embodiment[0]

    args["seed"] = load_seed(dataset_dir, episode)
    args["now_ep_num"] = episode

    traj = load_traj(dataset_dir, episode)
    args["left_joint_path"] = traj["left_joint_path"]
    args["right_joint_path"] = traj["right_joint_path"]
    return args


def replay_one(dataset_dir, output_dir, task_name, task_config, episode, keep_cache=False):
    env_module = importlib.import_module(f"envs.{task_name}")
    task = getattr(env_module, task_name)()
    args = build_args(task_name, task_config, dataset_dir, output_dir, episode)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    metadata_path = Path(output_dir) / "replay_metadata.json"

    try:
        task.setup_demo(**args)
        info = task.play_once()
        if not task.plan_success:
            raise RuntimeError("Replay consumed saved paths but task.plan_success is False")
        if not task.check_success():
            raise RuntimeError("Replay finished but task.check_success() is False")

        task.merge_pkl_to_hdf5_video()

        metadata = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
        metadata[f"episode_{episode}"] = {
            "task_name": task_name,
            "task_config": task_config,
            "source_dataset_dir": str(Path(dataset_dir).resolve()),
            "seed": args["seed"],
            "info": info,
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        cache = getattr(task, "folder_path", {}).get("cache")
        task.close_env(clear_cache=True)
        if cache and (not keep_cache) and Path(cache).exists():
            shutil.rmtree(cache)


def main():
    parser = argparse.ArgumentParser(
        description="Replay RoboTwin2.0 dataset episodes from saved _traj_data in a reconstructed seeded scene."
    )
    parser.add_argument("dataset_dir", help="Path like .../<task>/aloha-agilex_clean-50")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episode", type=int, action="append", required=True)
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--keep-cache", action="store_true")
    args = parser.parse_args()

    task_name = args.task_name or task_from_dataset_dir(args.dataset_dir)
    for episode in args.episode:
        print(f"Replaying {task_name} episode{episode} -> {args.output_dir}")
        replay_one(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            task_name=task_name,
            task_config=args.task_config,
            episode=episode,
            keep_cache=args.keep_cache,
        )


if __name__ == "__main__":
    main()
