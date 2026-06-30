from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

from SceneRecon.task_mapping import TaskTop1Mapping, default_search_gt_path
from SceneRecon2.place_empty_cup_editor import OUTPUT_DIR, _state_path_for_episode
from SceneRecon2.initializers import (
    AdjustBottleInitializer,
    BeatBlockHammerInitializer,
    BlocksRankingRgbInitializer,
    BlocksRankingSizeInitializer,
    ClickAlarmClockInitializer,
    ClickBellInitializer,
    DumpBinBigbinInitializer,
    GrabRollerInitializer,
    HandoverBlockInitializer,
    HandoverMicInitializer,
    HangingMugInitializer,
    LiftPotInitializer,
    MovePillBottlePadInitializer,
    MoveStaplerPadInitializer,
    MoveCanPotInitializer,
    MovePlayingCardAwayInitializer,
    OpenLaptopInitializer,
    OpenMicrowaveInitializer,
    PlaceA2BLeftInitializer,
    PlaceA2BRightInitializer,
    PlaceBreadBasketInitializer,
    PlaceBreadSkilletInitializer,
    PlaceBurgerFriesInitializer,
    PlaceCanBasketInitializer,
    PlaceCansPlasticboxInitializer,
    PlaceContainerPlateInitializer,
    PlaceDualShoesInitializer,
    PlaceEmptyCupInitializer,
    PlaceFanInitializer,
    PlaceMousePadInitializer,
    PlaceObjectBasketInitializer,
    PlaceObjectScaleInitializer,
    PlaceObjectStandInitializer,
    PlacePhoneStandInitializer,
    PlaceShoeInitializer,
    PickDiverseBottlesInitializer,
    PickDualBottlesInitializer,
    PressStaplerInitializer,
    PutBottlesDustbinInitializer,
    PutObjectCabinetInitializer,
    RotateQrcodeInitializer,
    ScanObjectInitializer,
    ShakeBottleHorizontallyInitializer,
    ShakeBottleInitializer,
    StampSealInitializer,
    StackBowlsThreeInitializer,
    StackBowlsTwoInitializer,
    StackBlocksThreeInitializer,
    StackBlocksTwoInitializer,
    TurnSwitchInitializer,
)


INITIALIZERS = {
    "adjust_bottle": AdjustBottleInitializer,
    "beat_block_hammer": BeatBlockHammerInitializer,
    "blocks_ranking_rgb": BlocksRankingRgbInitializer,
    "blocks_ranking_size": BlocksRankingSizeInitializer,
    "click_alarmclock": ClickAlarmClockInitializer,
    "click_bell": ClickBellInitializer,
    "dump_bin_bigbin": DumpBinBigbinInitializer,
    "grab_roller": GrabRollerInitializer,
    "handover_block": HandoverBlockInitializer,
    "handover_mic": HandoverMicInitializer,
    "hanging_mug": HangingMugInitializer,
    "lift_pot": LiftPotInitializer,
    "move_pillbottle_pad": MovePillBottlePadInitializer,
    "move_stapler_pad": MoveStaplerPadInitializer,
    "move_can_pot": MoveCanPotInitializer,
    "move_playingcard_away": MovePlayingCardAwayInitializer,
    "open_laptop": OpenLaptopInitializer,
    "open_microwave": OpenMicrowaveInitializer,
    "place_a2b_left": PlaceA2BLeftInitializer,
    "place_a2b_right": PlaceA2BRightInitializer,
    "place_bread_basket": PlaceBreadBasketInitializer,
    "place_bread_skillet": PlaceBreadSkilletInitializer,
    "place_burger_fries": PlaceBurgerFriesInitializer,
    "place_can_basket": PlaceCanBasketInitializer,
    "place_cans_plasticbox": PlaceCansPlasticboxInitializer,
    "place_container_plate": PlaceContainerPlateInitializer,
    "place_dual_shoes": PlaceDualShoesInitializer,
    "place_empty_cup": PlaceEmptyCupInitializer,
    "place_fan": PlaceFanInitializer,
    "place_mouse_pad": PlaceMousePadInitializer,
    "place_object_basket": PlaceObjectBasketInitializer,
    "place_object_scale": PlaceObjectScaleInitializer,
    "place_object_stand": PlaceObjectStandInitializer,
    "place_phone_stand": PlacePhoneStandInitializer,
    "place_shoe": PlaceShoeInitializer,
    "pick_diverse_bottles": PickDiverseBottlesInitializer,
    "pick_dual_bottles": PickDualBottlesInitializer,
    "press_stapler": PressStaplerInitializer,
    "put_bottles_dustbin": PutBottlesDustbinInitializer,
    "put_object_cabinet": PutObjectCabinetInitializer,
    "rotate_qrcode": RotateQrcodeInitializer,
    "scan_object": ScanObjectInitializer,
    "shake_bottle": ShakeBottleInitializer,
    "shake_bottle_horizontally": ShakeBottleHorizontallyInitializer,
    "stamp_seal": StampSealInitializer,
    "stack_bowls_three": StackBowlsThreeInitializer,
    "stack_bowls_two": StackBowlsTwoInitializer,
    "stack_blocks_three": StackBlocksThreeInitializer,
    "stack_blocks_two": StackBlocksTwoInitializer,
    "turn_switch": TurnSwitchInitializer,
}


def _default_first_frame_dir(split: str) -> Path:
    return Path(f"/home/users/liang01.yue/D/WorldArena_Robotwin2.0/{split}/first_frame/fixed_scene_task")


def _base_state(task_name: str, first_frame: Path, objects: dict[str, dict[str, Any]]) -> dict[str, Any]:
    episode_name = first_frame.stem
    return {
        "task": task_name,
        "episode": episode_name,
        "first_frame_path": str(first_frame),
        "rendering": {
            "mode": "interactive",
            "shader_request": "rt",
            "display_scale": 1.0,
            "jpeg_quality": 95,
            "transport": "memory-jpeg",
        },
        "objects": objects,
        "camera": {
            "name": "head_camera",
            "position": [-0.032, -0.45, 1.35],
            "base_position": [-0.032, -0.45, 1.35],
            "pitch_deg": 0.0,
            "fovy_deg": 37.0,
        },
        "edit_status": "initializer",
    }


def _fallback_objects(task_name: str) -> dict[str, dict[str, Any]]:
    if task_name == "click_bell":
        return {"bell": {"name": "bell", "modelname": "050_bell", "model_id": 0, "table_xy": [0.12, -0.1], "z": 0.741}}
    if task_name == "beat_block_hammer":
        return {
            "hammer": {"name": "hammer", "modelname": "020_hammer", "model_id": 0, "table_xy": [0.0, -0.06], "z": 0.783},
            "block": {"name": "block", "kind": "box", "color": [1, 0, 0], "table_xy": [0.12, 0.05], "z": 0.76},
        }
    if task_name == "stack_blocks_three":
        return {
            "red_block": {"name": "red_block", "kind": "box", "color": [1, 0, 0], "table_xy": [-0.16, -0.02], "z": 0.766},
            "green_block": {"name": "green_block", "kind": "box", "color": [0, 1, 0], "table_xy": [0.16, -0.02], "z": 0.766},
            "blue_block": {"name": "blue_block", "kind": "box", "color": [0, 0, 1], "table_xy": [0.0, 0.03], "z": 0.766},
        }
    if task_name == "stack_blocks_two":
        return {
            "red_block": {"name": "red_block", "kind": "box", "color": [1, 0, 0], "table_xy": [-0.16, -0.02], "z": 0.766},
            "green_block": {"name": "green_block", "kind": "box", "color": [0, 1, 0], "table_xy": [0.16, -0.02], "z": 0.766},
        }
    if task_name == "blocks_ranking_rgb":
        return {
            "red_block": {"name": "red_block", "kind": "box", "color": [1, 0, 0], "table_xy": [-0.18, -0.02], "z": 0.766, "half_size": [0.025, 0.025, 0.025]},
            "green_block": {"name": "green_block", "kind": "box", "color": [0, 1, 0], "table_xy": [0.0, -0.02], "z": 0.766, "half_size": [0.025, 0.025, 0.025]},
            "blue_block": {"name": "blue_block", "kind": "box", "color": [0, 0, 1], "table_xy": [0.18, -0.02], "z": 0.766, "half_size": [0.025, 0.025, 0.025]},
        }
    if task_name == "blocks_ranking_size":
        return {
            "block1": {"name": "block1", "kind": "box", "color": "unknown", "table_xy": [-0.18, -0.02], "z": 0.766, "half_size": [0.025, 0.025, 0.025]},
            "block2": {"name": "block2", "kind": "box", "color": "unknown", "table_xy": [0.0, -0.02], "z": 0.766, "half_size": [0.025, 0.025, 0.025]},
            "block3": {"name": "block3", "kind": "box", "color": "unknown", "table_xy": [0.18, -0.02], "z": 0.766, "half_size": [0.025, 0.025, 0.025]},
        }
    if task_name == "handover_block":
        return {
            "block": {"name": "block", "kind": "box", "color": [1, 0, 0], "half_size": [0.03, 0.03, 0.1], "table_xy": [-0.15, 0.12], "z": 0.842},
            "target_box": {"name": "target_box", "kind": "box", "color": [0, 0, 1], "half_size": [0.05, 0.05, 0.005], "table_xy": [0.18, 0.17], "z": 0.741},
        }
    if task_name == "turn_switch":
        return {
            "switch": {
                "name": "switch",
                "modelname": "056_switch",
                "model_id": 0,
                "model_id_candidates": list(range(8)),
                "table_xy": [0.0, 0.05],
                "z": 0.825,
                "qpos": [0.704141, 0, 0, 0.71006],
            }
        }
    if task_name == "press_stapler":
        return {
            "stapler": {
                "name": "stapler",
                "modelname": "048_stapler",
                "model_id": 0,
                "model_id_candidates": list(range(7)),
                "table_xy": [0.0, -0.02],
                "z": 0.741,
                "qpos": [0.5, 0.5, 0.5, 0.5],
            }
        }
    if task_name == "click_alarmclock":
        return {
            "alarmclock": {
                "name": "alarmclock",
                "modelname": "046_alarm-clock",
                "model_id": 1,
                "model_id_candidates": [1, 3],
                "table_xy": [0.12, -0.1],
                "z": 0.741,
                "qpos": [0.5, 0.5, 0.5, 0.5],
            }
        }
    if task_name == "move_pillbottle_pad":
        return {
            "pillbottle": {
                "name": "pillbottle",
                "modelname": "080_pillbottle",
                "model_id": 1,
                "model_id_candidates": [1, 2, 3, 4, 5],
                "table_xy": [0.12, 0.0],
                "z": 0.741,
                "qpos": [0.5, 0.5, 0.5, 0.5],
            },
            "pad": {"name": "pad", "kind": "box", "color": [0, 0, 1], "table_xy": [0.12, -0.12], "z": 0.741},
        }
    if task_name == "move_stapler_pad":
        return {
            "stapler": {
                "name": "stapler",
                "modelname": "048_stapler",
                "model_id": 0,
                "model_id_candidates": list(range(7)),
                "table_xy": [0.12, -0.1],
                "z": 0.741,
                "qpos": [0.5, 0.5, 0.5, 0.5],
            },
            "pad": {"name": "pad", "kind": "box", "color": "unknown", "table_xy": [0.12, -0.16], "z": 0.741},
        }
    if task_name == "place_mouse_pad":
        return {
            "mouse": {"name": "mouse", "modelname": "047_mouse", "model_id": 0, "model_id_candidates": [0, 1, 2], "table_xy": [0.12, -0.1], "z": 0.741},
            "pad": {"name": "pad", "kind": "box", "color": "unknown", "table_xy": [0.12, -0.16], "z": 0.741},
        }
    if task_name == "place_fan":
        return {
            "fan": {"name": "fan", "modelname": "099_fan", "model_id": 4, "model_id_candidates": [4, 5], "table_xy": [0.0, -0.1], "z": 0.741},
            "pad": {"name": "pad", "kind": "box", "color": "unknown", "table_xy": [0.2, -0.1], "z": 0.741},
        }
    if task_name == "move_can_pot":
        return {
            "pot": {"name": "pot", "modelname": "060_kitchenpot", "model_id": 0, "model_id_candidates": list(range(7)), "table_xy": [0.0, 0.0], "z": 0.741},
            "can": {"name": "can", "modelname": "105_sauce-can", "model_id": 0, "model_id_candidates": [0, 2, 4, 5, 6], "table_xy": [0.25, 0.1], "z": 0.741},
        }
    if task_name == "place_phone_stand":
        return {
            "phone": {"name": "phone", "modelname": "077_phone", "model_id": 0, "model_id_candidates": [0, 1, 2, 4], "table_xy": [0.2, -0.1], "z": 0.741},
            "stand": {"name": "stand", "modelname": "078_phonestand", "model_id": 1, "model_id_candidates": [1, 2], "table_xy": [0.1, 0.1], "z": 0.741},
        }
    if task_name == "place_can_basket":
        return {
            "basket": {"name": "basket", "modelname": "110_basket", "model_id": 0, "model_id_candidates": [0, 1], "table_xy": [0.02, -0.065], "z": 0.741, "qpos": [0.5, 0.5, 0.5, 0.5]},
            "can": {"name": "can", "modelname": "071_can", "model_id": 0, "model_id_candidates": [0, 1, 2, 3, 5, 6], "table_xy": [-0.22, 0.05], "z": 0.741, "qpos": [0.707225, 0.706849, -0.0100455, -0.00982061]},
        }
    if task_name == "place_object_basket":
        return {
            "basket": {"name": "basket", "modelname": "110_basket", "model_id": 0, "model_id_candidates": [0, 1], "table_xy": [0.02, -0.065], "z": 0.741, "qpos": [0.5, 0.5, 0.5, 0.5]},
            "object": {"name": "object", "modelname": "057_toycar", "model_id": 0, "model_id_candidates": [0, 1, 2, 3, 4, 5], "modelname_candidates": ["057_toycar", "081_playingcards"], "table_xy": [-0.22, 0.0], "z": 0.741, "qpos": [0.707225, 0.706849, -0.0100455, -0.00982061]},
        }
    if task_name == "place_object_scale":
        return {
            "scale": {"name": "scale", "modelname": "072_electronicscale", "model_id": 0, "model_id_candidates": [0, 1, 5, 6], "table_xy": [0.12, -0.08], "z": 0.741, "qpos": [0.5, 0.5, 0.5, 0.5]},
            "object": {"name": "object", "modelname": "047_mouse", "model_id": 0, "model_id_candidates": [0, 1, 2], "modelname_candidates": ["047_mouse", "048_stapler", "050_bell"], "table_xy": [-0.12, -0.08], "z": 0.741},
        }
    if task_name == "place_object_stand":
        return {
            "stand": {"name": "stand", "modelname": "074_displaystand", "model_id": 0, "model_id_candidates": [0, 1, 2, 3, 4], "table_xy": [0.02, -0.12], "z": 0.741, "qpos": [0.707, 0.707, 0.0, 0.0]},
            "object": {"name": "object", "modelname": "047_mouse", "model_id": 0, "model_id_candidates": [0, 1, 2], "modelname_candidates": ["047_mouse", "048_stapler", "050_bell", "073_rubikscube", "057_toycar", "079_remotecontrol"], "table_xy": [0.24, 0.0], "z": 0.741, "qpos": [0.707, 0.707, 0.0, 0.0]},
        }
    if task_name in ("pick_dual_bottles", "pick_diverse_bottles"):
        return {
            "bottle1": {"name": "bottle1", "modelname": "001_bottle", "model_id": 13, "model_id_candidates": list(range(20)), "table_xy": [-0.16, 0.12], "z": 0.785, "qpos": [0.66, 0.66, -0.25, -0.25]},
            "bottle2": {"name": "bottle2", "modelname": "001_bottle", "model_id": 16, "model_id_candidates": list(range(20)), "table_xy": [0.16, 0.12], "z": 0.785, "qpos": [0.65, 0.65, 0.27, 0.27]},
        }
    if task_name == "put_bottles_dustbin":
        return {
            "bottle1": {"name": "bottle1", "modelname": "114_bottle", "model_id": 1, "model_id_candidates": [1], "table_xy": [-0.2, 0.12], "z": 0.741, "qpos": [0.707, 0.707, 0, 0]},
            "bottle2": {"name": "bottle2", "modelname": "114_bottle", "model_id": 2, "model_id_candidates": [2], "table_xy": [0.08, 0.12], "z": 0.741, "qpos": [0.707, 0.707, 0, 0]},
            "bottle3": {"name": "bottle3", "modelname": "114_bottle", "model_id": 3, "model_id_candidates": [3], "table_xy": [0.28, 0.12], "z": 0.741, "qpos": [0.707, 0.707, 0, 0]},
            "dustbin": {"name": "dustbin", "modelname": "011_dustbin", "model_id": 0, "model_id_candidates": [0], "table_xy": [-0.45, 0.0], "z": 0.0, "qpos": [0.5, 0.5, 0.5, 0.5]},
        }
    if task_name == "dump_bin_bigbin":
        return {
            "dustbin": {"name": "dustbin", "modelname": "011_dustbin", "model_id": 0, "model_id_candidates": [0], "table_xy": [-0.45, 0.0], "z": 0.0, "qpos": [0.5, 0.5, 0.5, 0.5]},
            "deskbin": {"name": "deskbin", "modelname": "063_tabletrashbin", "model_id": 0, "model_id_candidates": [0, 3, 7, 8, 9, 10], "table_xy": [0.12, -0.12], "z": 0.741, "qpos": [0.651892, 0.651428, 0.274378, 0.274584]},
        }
    if task_name in ("adjust_bottle", "shake_bottle", "shake_bottle_horizontally"):
        return {
            "bottle": {"name": "bottle", "modelname": "001_bottle", "model_id": 13, "model_id_candidates": list(range(20)), "table_xy": [-0.12, -0.1], "z": 0.785, "qpos": [0, 0, 1, 0]},
        }
    if task_name == "grab_roller":
        return {"roller": {"name": "roller", "modelname": "102_roller", "model_id": 0, "model_id_candidates": [0, 2], "table_xy": [0.0, -0.14], "z": 0.741, "qpos": [0.5, 0.5, 0.5, 0.5]}}
    if task_name == "handover_mic":
        return {"microphone": {"name": "microphone", "modelname": "018_microphone", "model_id": 0, "model_id_candidates": [0, 4, 5], "table_xy": [-0.18, -0.02], "z": 0.741, "qpos": [0.707, 0.707, 0, 0]}}
    if task_name == "lift_pot":
        return {"pot": {"name": "pot", "modelname": "060_kitchenpot", "model_id": 0, "model_id_candidates": [0, 1], "table_xy": [0.0, 0.0], "z": 0.741, "qpos": [0.704141, 0, 0, 0.71006]}}
    if task_name == "hanging_mug":
        return {
            "mug": {"name": "mug", "modelname": "039_mug", "model_id": 0, "model_id_candidates": list(range(10)), "table_xy": [-0.18, 0.0], "z": 0.741, "qpos": [0.707, 0.707, 0, 0]},
            "rack": {"name": "rack", "modelname": "040_rack", "model_id": 0, "model_id_candidates": [0], "table_xy": [0.2, 0.15], "z": 0.741, "qpos": [-0.22, -0.22, 0.67, 0.67]},
        }
    if task_name == "stamp_seal":
        return {
            "seal": {"name": "seal", "modelname": "100_seal", "model_id": 0, "model_id_candidates": [0, 2, 3, 4, 6], "table_xy": [-0.15, 0.0], "z": 0.741, "qpos": [0.5, 0.5, 0.5, 0.5]},
            "target": {"name": "target", "kind": "box", "color": "unknown", "half_size": [0.035, 0.035, 0.0005], "table_xy": [0.15, 0.0], "z": 0.741},
        }
    if task_name == "place_shoe":
        return {
            "target_pad": {"name": "target_pad", "kind": "box", "color": [0, 0, 1], "half_size": [0.13, 0.05, 0.0005], "table_xy": [0.0, -0.08], "z": 0.74},
            "shoe": {"name": "shoe", "modelname": "041_shoe", "model_id": 0, "model_id_candidates": list(range(10)), "table_xy": [0.22, 0.0], "z": 0.741, "qpos": [0.707, 0.707, 0, 0]},
        }
    if task_name == "place_dual_shoes":
        return {
            "shoe_box": {"name": "shoe_box", "modelname": "007_shoe-box", "model_id": 0, "model_id_candidates": [0], "table_xy": [0.0, -0.13], "z": 0.74, "qpos": [0.5, 0.5, -0.5, -0.5]},
            "shoe1": {"name": "shoe1", "modelname": "041_shoe", "model_id": 0, "model_id_candidates": list(range(10)), "table_xy": [-0.24, 0.0], "z": 0.741, "qpos": [0.707, 0.707, 0, 0]},
            "shoe2": {"name": "shoe2", "modelname": "041_shoe", "model_id": 0, "model_id_candidates": list(range(10)), "table_xy": [0.24, 0.0], "z": 0.741, "qpos": [0.707, 0.707, 0, 0]},
        }
    if task_name == "place_cans_plasticbox":
        return {
            "plasticbox": {"name": "plasticbox", "modelname": "062_plasticbox", "model_id": 0, "model_id_candidates": [0, 1], "table_xy": [0.0, -0.125], "z": 0.741, "qpos": [0.5, 0.5, 0.5, 0.5]},
            "can1": {"name": "can1", "modelname": "071_can", "model_id": 0, "model_id_candidates": [0, 1, 2, 3, 5, 6], "table_xy": [-0.2, -0.1], "z": 0.741, "qpos": [0.5, 0.5, 0.5, 0.5]},
            "can2": {"name": "can2", "modelname": "071_can", "model_id": 0, "model_id_candidates": [0, 1, 2, 3, 5, 6], "table_xy": [0.2, -0.1], "z": 0.741, "qpos": [0.5, 0.5, 0.5, 0.5]},
        }
    if task_name == "place_burger_fries":
        return {
            "tray": {"name": "tray", "modelname": "008_tray", "model_id": 0, "model_id_candidates": list(range(8)), "table_xy": [0.0, -0.125], "z": 0.741, "qpos": [0.706527, 0.706483, -0.0291356, -0.0291767]},
            "hamburg": {"name": "hamburg", "modelname": "006_hamburg", "model_id": 0, "model_id_candidates": [0], "table_xy": [-0.27, -0.1], "z": 0.741, "qpos": [0.5, 0.5, 0.5, 0.5]},
            "frenchfries": {"name": "frenchfries", "modelname": "005_french-fries", "model_id": 0, "model_id_candidates": [0], "table_xy": [0.25, -0.1], "z": 0.741, "qpos": [1.0, 0.0, 0.0, 0.0]},
        }
    if task_name == "place_bread_basket":
        return {
            "breadbasket": {"name": "breadbasket", "modelname": "076_breadbasket", "model_id": 0, "model_id_candidates": [0, 1, 2, 3, 4], "table_xy": [0.0, -0.2], "z": 0.741, "qpos": [0.5, 0.5, 0.5, 0.5]},
            "bread1": {"name": "bread1", "modelname": "075_bread", "model_id": 0, "model_id_candidates": [0, 1, 3, 5, 6], "table_xy": [-0.22, -0.05], "z": 0.741, "qpos": [0.707, 0.707, 0.0, 0.0]},
            "bread2": {"name": "bread2", "modelname": "075_bread", "model_id": 0, "model_id_candidates": [0, 1, 3, 5, 6], "table_xy": [0.22, -0.05], "z": 0.741, "qpos": [0.707, 0.707, 0.0, 0.0]},
        }
    if task_name == "place_bread_skillet":
        return {
            "skillet": {"name": "skillet", "modelname": "106_skillet", "model_id": 0, "model_id_candidates": list(range(4)), "table_xy": [0.18, -0.06], "z": 0.741, "qpos": [0, 0, 0.707, 0.707]},
            "bread": {"name": "bread", "modelname": "075_bread", "model_id": 0, "model_id_candidates": [0, 1, 3, 5, 6], "table_xy": [-0.18, -0.06], "z": 0.741, "qpos": [0.707, 0.707, 0.0, 0.0]},
        }
    if task_name == "place_container_plate":
        return {
            "plate": {"name": "plate", "modelname": "003_plate", "model_id": 0, "model_id_candidates": list(range(7)), "table_xy": [0.12, -0.12], "z": 0.741, "qpos": [0.5, 0.5, 0.5, 0.5]},
            "container": {"name": "container", "modelname": "004_container", "model_id": 0, "model_id_candidates": list(range(8)), "table_xy": [-0.12, -0.02], "z": 0.741, "qpos": [0.5, 0.5, 0.5, 0.5]},
        }
    if task_name == "move_playingcard_away":
        return {
            "playingcards": {"name": "playingcards", "modelname": "081_playingcards", "model_id": 0, "model_id_candidates": [0, 1, 2], "table_xy": [0.0, -0.08], "z": 0.741, "qpos": [0.5, 0.5, 0.5, 0.5]},
        }
    if task_name == "rotate_qrcode":
        return {
            "qrcode": {"name": "qrcode", "modelname": "070_paymentsign", "model_id": 0, "model_id_candidates": list(range(6)), "table_xy": [0.12, -0.12], "z": 0.741, "qpos": [0, 0, 0.707, 0.707]},
        }
    if task_name == "open_laptop":
        return {
            "laptop": {"name": "laptop", "modelname": "015_laptop", "model_id": 0, "model_id_candidates": list(range(11)), "table_xy": [0.0, -0.02], "z": 0.741, "qpos": [0.7, 0, 0, 0.7], "articulation_qpos": [0.2]},
        }
    if task_name == "open_microwave":
        return {
            "microwave": {"name": "microwave", "modelname": "044_microwave", "model_id": 0, "model_id_candidates": [0, 1], "table_xy": [-0.07, 0.17], "z": 0.8, "qpos": [0.707, 0, 0, 0.707], "articulation_qpos": [0.0]},
        }
    if task_name == "put_object_cabinet":
        return {
            "cabinet": {"name": "cabinet", "modelname": "036_cabinet", "model_id": 46653, "model_id_candidates": [46653], "table_xy": [0.0, 0.155], "z": 0.741, "qpos": [1, 0, 0, 1], "articulation_qpos": [0.0]},
            "object": {"name": "object", "modelname": "047_mouse", "model_id": 0, "model_id_candidates": [0, 1, 2], "modelname_candidates": ["047_mouse", "048_stapler", "050_bell", "057_toycar", "073_rubikscube", "075_bread", "077_phone", "081_playingcards", "086_woodenblock", "112_tea-box", "113_coffee-box", "107_soap"], "table_xy": [0.18, -0.08], "z": 0.741, "qpos": [0.5, 0.5, 0.5, 0.5]},
        }
    if task_name == "scan_object":
        return {
            "scanner": {"name": "scanner", "modelname": "024_scanner", "model_id": 0, "model_id_candidates": [0, 1, 2, 3, 4], "table_xy": [-0.18, -0.1], "z": 0.741, "qpos": [0, 0, 0.707, 0.707]},
            "object": {"name": "object", "modelname": "112_tea-box", "model_id": 0, "model_id_candidates": [0, 1, 2, 3, 4, 5], "table_xy": [0.18, -0.1], "z": 0.741, "qpos": [0.5, 0.5, 0.5, 0.5]},
        }
    if task_name in ("place_a2b_left", "place_a2b_right"):
        return {
            "object": {"name": "object", "modelname": "047_mouse", "model_id": 0, "model_id_candidates": [0, 1, 2], "modelname_candidates": ["047_mouse", "048_stapler", "050_bell", "057_toycar", "073_rubikscube", "075_bread", "077_phone", "081_playingcards", "086_woodenblock", "112_tea-box", "113_coffee-box", "107_soap"], "table_xy": [-0.16, -0.08], "z": 0.741, "qpos": [0.5, 0.5, 0.5, 0.5]},
            "target_object": {"name": "target_object", "modelname": "048_stapler", "model_id": 0, "model_id_candidates": list(range(7)), "modelname_candidates": ["047_mouse", "048_stapler", "050_bell", "057_toycar", "073_rubikscube", "075_bread", "077_phone", "081_playingcards", "086_woodenblock", "112_tea-box", "113_coffee-box", "107_soap"], "table_xy": [0.16, -0.08], "z": 0.741, "qpos": [0.5, 0.5, 0.5, 0.5]},
        }
    if task_name == "stack_bowls_two":
        return {
            "bowl1": {"name": "bowl1", "modelname": "002_bowl", "model_id": 3, "model_id_candidates": [3], "table_xy": [-0.2, 0.0], "z": 0.741},
            "bowl2": {"name": "bowl2", "modelname": "002_bowl", "model_id": 3, "model_id_candidates": [3], "table_xy": [0.2, 0.0], "z": 0.741},
        }
    if task_name == "stack_bowls_three":
        return {
            "bowl1": {"name": "bowl1", "modelname": "002_bowl", "model_id": 3, "model_id_candidates": [3], "table_xy": [-0.2, 0.0], "z": 0.741},
            "bowl2": {"name": "bowl2", "modelname": "002_bowl", "model_id": 3, "model_id_candidates": [3], "table_xy": [0.0, 0.0], "z": 0.741},
            "bowl3": {"name": "bowl3", "modelname": "002_bowl", "model_id": 3, "model_id_candidates": [3], "table_xy": [0.2, 0.0], "z": 0.741},
        }
    return {
        "cup": {"name": "cup", "modelname": "021_cup", "model_id": 0, "table_xy": [0.22, -0.08], "z": 0.741},
        "coaster": {"name": "coaster", "modelname": "019_coaster", "model_id": 0, "table_xy": [0.02, -0.08], "z": 0.741},
    }


def _ensure_object_defaults(objects: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    for obj in objects.values():
        if isinstance(obj, dict) and ("table_xy" in obj or "z" in obj):
            obj.setdefault("yaw_deg", 0.0)
    return objects


def precompute(split: str, task_names: list[str], overwrite: bool) -> None:
    mapping = TaskTop1Mapping.from_search_gt(default_search_gt_path(split))
    first_frame_dir = _default_first_frame_dir(split)
    out_dir = OUTPUT_DIR / "initializer_states" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    for task_name in task_names:
        if task_name not in INITIALIZERS:
            raise ValueError(f"Unsupported initializer task: {task_name}")
        initializer = INITIALIZERS[task_name](OUTPUT_DIR / split)
        episodes = sorted(ep for ep, match in mapping.matches.items() if match.task_name == task_name)
        print(f"{task_name}: {len(episodes)} episodes")
        for episode in episodes:
            out_path = _state_path_for_episode(out_dir, episode, task_name)
            if out_path.exists() and not overwrite:
                print(f"  {episode}: exists")
                continue
            first_frame = first_frame_dir / f"episode{episode}.png"
            objects = _ensure_object_defaults(_fallback_objects(task_name))
            try:
                result = initializer.initialize(first_frame)
                for name, obj in result.objects.items():
                    objects[name] = obj
                _ensure_object_defaults(objects)
                init_info = {
                    "enabled": True,
                    "ok": result.ok,
                    "source": result.source,
                    "notes": result.notes,
                    "debug_json_path": result.debug_json_path,
                    "debug_image_path": result.debug_image_path,
                    "offline_precomputed": True,
                }
            except Exception as exc:
                init_info = {
                    "enabled": True,
                    "ok": False,
                    "source": "fallback_default",
                    "error": repr(exc),
                    "notes": ["Offline rough initialization failed; using rough default poses."],
                    "offline_precomputed": True,
                }
            state = _base_state(task_name, first_frame, objects)
            state["initialization"] = init_info
            payload = {
                "saved_at_unix": time.time(),
                "save_kind": "auto_initializer",
                "first_frame_path": str(first_frame),
                "state": copy.deepcopy(state),
            }
            tmp = out_path.with_suffix(out_path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(out_path)
            print(f"  {episode}: {init_info['source']} ok={init_info['ok']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute SceneRecon2 initializer states.")
    parser.add_argument("--split", default="test_dataset")
    parser.add_argument("--tasks", nargs="+", default=list(INITIALIZERS))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    precompute(args.split, args.tasks, args.overwrite)


if __name__ == "__main__":
    main()
