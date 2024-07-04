# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Helper script to get task reset distribution videos.
"""
import os
import h5py
import argparse
import imageio
import numpy as np
from tqdm import tqdm

import robosuite
from robosuite.controllers import load_controller_config

import mimicgen


# base output folder
OUTPUT_FOLDER = "/tmp/mimicgen_reset_vids"

# number of resets to view per task
NUM_RESETS = 10

# store info for each dataset and environment
DATASET_INFOS = [
    # core datasets
    dict(
        name="hammer_cleanup",
        envs=["HammerCleanup_D0", "HammerCleanup_D1"],
        ds_names=["hammer_cleanup_d0.hdf5", "hammer_cleanup_d1.hdf5"],
    ),
    dict(
        name="kitchen",
        envs=["Kitchen_D0", "Kitchen_D1"],
        ds_names=["kitchen_d0.hdf5", "kitchen_d1.hdf5"],
    ),
    dict(
        name="coffee",
        envs=["Coffee_D0", "Coffee_D1", "Coffee_D2"],
        ds_names=["coffee_d0.hdf5", "coffee_d1.hdf5", "coffee_d2.hdf5"],
    ),
    dict(
        name="coffee_preparation",
        envs=["CoffeePreparation_D0", "CoffeePreparation_D1"],
        ds_names=["coffee_preparation_d0.hdf5", "coffee_preparation_d1.hdf5"],
    ),
    dict(
        name="nut_assembly",
        envs=["NutAssembly_D0"],
        robots=["Sawyer"],
        ds_names=["nut_assembly_d0.hdf5"],
    ),
    dict(
        name="mug_cleanup",
        envs=["MugCleanup_D0", "MugCleanup_D1"],
        ds_names=["mug_cleanup_d0.hdf5", "mug_cleanup_d1.hdf5"],
    ),
    dict(
        name="pick_place",
        envs=["PickPlace_D0"],
        robots=["Sawyer"],
        ds_names=["pick_place_d0.hdf5"],
    ),
    dict(
        name="square",
        envs=["Square_D0", "Square_D1", "Square_D2"],
        ds_names=["square_d0.hdf5", "square_d1.hdf5", "square_d2.hdf5"],
    ),
    dict(
        name="stack",
        envs=["Stack_D0", "Stack_D1"],
        ds_names=["stack_d0.hdf5", "stack_d1.hdf5"],
    ),
    dict(
        name="stack_three",
        envs=["StackThree_D0", "StackThree_D1"],
        ds_names=["stack_three_d0.hdf5", "stack_three_d1.hdf5"],
    ),
    dict(
        name="threading",
        envs=["Threading_D0", "Threading_D1", "Threading_D2"],
        ds_names=["threading_d0.hdf5", "threading_d1.hdf5", "threading_d2.hdf5"],
    ),
    dict(
        name="three_piece_assembly",
        envs=["ThreePieceAssembly_D0", "ThreePieceAssembly_D1", "ThreePieceAssembly_D2"],
        ds_names=["three_piece_assembly_d0.hdf5", "three_piece_assembly_d1.hdf5", "three_piece_assembly_d2.hdf5"],
    ),

    # object transfer datasets
    dict(
        name="mug_cleanup",
        envs=["MugCleanup_O1", "MugCleanup_O2"],
        ds_names=["mug_cleanup_o1.hdf5", "mug_cleanup_o2.hdf5"],
    ),

    # robot transfer datasets
    dict(
        name="square",
        envs=["Square_D0", "Square_D0", "Square_D0"],
        robots=["Sawyer", "IIWA", "UR5e"],
        grippers=["RethinkGripper", "Robotiq85Gripper", "Robotiq85Gripper"],
        ds_names=["square_d0_sawyer.hdf5", "square_d0_iiwa.hdf5", "square_d0_ur5e.hdf5"],
    ),
    dict(
        name="square",
        envs=["Square_D1", "Square_D1", "Square_D1"],
        robots=["Sawyer", "IIWA", "UR5e"],
        grippers=["RethinkGripper", "Robotiq85Gripper", "Robotiq85Gripper"],
        ds_names=["square_d1_sawyer.hdf5", "square_d1_iiwa.hdf5", "square_d1_ur5e.hdf5"],
    ),
    dict(
        name="threading",
        envs=["Threading_D0", "Threading_D0", "Threading_D0"],
        robots=["Sawyer", "IIWA", "UR5e"],
        grippers=["RethinkGripper", "Robotiq85Gripper", "Robotiq85Gripper"],
        ds_names=["threading_d0_sawyer.hdf5", "threading_d0_iiwa.hdf5", "threading_d0_ur5e.hdf5"],
    ),
    dict(
        name="threading",
        envs=["Threading_D1", "Threading_D1", "Threading_D1"],
        robots=["Sawyer", "IIWA", "UR5e"],
        grippers=["RethinkGripper", "Robotiq85Gripper", "Robotiq85Gripper"],
        ds_names=["threading_d1_sawyer.hdf5", "threading_d1_iiwa.hdf5", "threading_d1_ur5e.hdf5"],
    ),
]


def make_reset_video(
    env_name,
    robot_name,
    camera_name,
    video_path,
    num_frames,
    gripper_name=None,
):
    # initialize the task
    env_args = dict(
        env_name=env_name,
        robots=robot_name,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        controller_configs=load_controller_config(default_controller="OSC_POSE"),
    )
    if gripper_name is not None:
        env_args["gripper_types"] = gripper_name
    env = robosuite.make(**env_args)

    # write a video
    video_writer = imageio.get_writer(video_path, fps=5)
    for i in tqdm(range(num_frames)):
        env.reset()
        video_img = env.sim.render(height=512, width=512, camera_name=camera_name)[::-1]
        video_writer.append_data(video_img)
    video_writer.close()

    env.close()
    del env


if __name__ == "__main__":
    output_folder = os.path.expanduser(OUTPUT_FOLDER)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for d in DATASET_INFOS:
        assert "envs" in d
        robot_names = ["Panda"] * len(d["envs"])
        need_robot_ext = False
        if "robots" in d:
            robot_names = d["robots"]
            need_robot_ext = True
        gripper_names = [None] * len(d["envs"])
        if "grippers" in d:
            gripper_names = d["grippers"]
        for env_ind, env in enumerate(d["envs"]):
            suffix = "{}_{}.mp4".format(env, robot_names[env_ind]) if need_robot_ext else "{}.mp4".format(env)
            output_path = os.path.join(output_folder, suffix)
            print(suffix)
            make_reset_video(
                env_name=env,
                robot_name=robot_names[env_ind],
                gripper_name=gripper_names[env_ind],
                camera_name="agentview",
                video_path=output_path,
                num_frames=NUM_RESETS,
            )
