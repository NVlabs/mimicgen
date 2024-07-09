# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
We utilize robomimic's config generator class to easily generate policy training configs for the 
core set of experiments in the paper. It can be modified easily to generate other 
training configs.

See https://robomimic.github.io/docs/tutorials/hyperparam_scan.html for more info.
"""
import os
import json
import shutil
import argparse

import robomimic
from robomimic.utils.hyperparam_utils import ConfigGenerator

import mimicgen
import mimicgen.utils.config_utils as ConfigUtils
from mimicgen.utils.file_utils import config_generator_to_script_lines


# set path to folder with mimicgen generated datasets
DATASET_DIR = "/tmp/core_datasets"

# set base folder for where to generate new config files for training runs
CONFIG_DIR = "/tmp/core_train_configs"

# set base folder for training outputs (model checkpoints, videos, logs)
OUTPUT_DIR = "/tmp/core_training_results"

# path to base config
BASE_CONFIG = os.path.join(robomimic.__path__[0], "exps/templates/bc.json")


def make_generators(base_config, dataset_dir, output_dir):
    """
    An easy way to make multiple config generators by using different 
    settings for each.
    """
    all_settings = [
        # stack
        dict(
            dataset_paths=[
                os.path.join(dataset_dir, "stack", "demo_src_stack_task_D0/demo.hdf5"),
                os.path.join(dataset_dir, "stack", "demo_src_stack_task_D1/demo.hdf5"),
            ],
            dataset_names=[
                "stack_D0",
                "stack_D1",
            ],
            horizon=400,
        ),
        # stack_three
        dict(
            dataset_paths=[
                os.path.join(dataset_dir, "stack_three", "demo_src_stack_three_task_D0/demo.hdf5"),
                os.path.join(dataset_dir, "stack_three", "demo_src_stack_three_task_D1/demo.hdf5"),
            ],
            dataset_names=[
                "stack_three_D0",
                "stack_three_D1",
            ],
            horizon=400,
        ),
        # square
        dict(
            dataset_paths=[
                os.path.join(dataset_dir, "square", "demo_src_square_task_D0/demo.hdf5"),
                os.path.join(dataset_dir, "square", "demo_src_square_task_D1/demo.hdf5"),
                os.path.join(dataset_dir, "square", "demo_src_square_task_D2/demo.hdf5"),
            ],
            dataset_names=[
                "square_D0",
                "square_D1",
                "square_D2",
            ],
            horizon=400,
        ),
        # threading
        dict(
            dataset_paths=[
                os.path.join(dataset_dir, "threading", "demo_src_threading_task_D0/demo.hdf5"),
                os.path.join(dataset_dir, "threading", "demo_src_threading_task_D1/demo.hdf5"),
                os.path.join(dataset_dir, "threading", "demo_src_threading_task_D2/demo.hdf5"),
            ],
            dataset_names=[
                "threading_D0",
                "threading_D1",
                "threading_D2",
            ],
            horizon=400,
        ),
        # three_piece_assembly
        dict(
            dataset_paths=[
                os.path.join(dataset_dir, "three_piece_assembly", "demo_src_three_piece_assembly_task_D0/demo.hdf5"),
                os.path.join(dataset_dir, "three_piece_assembly", "demo_src_three_piece_assembly_task_D1/demo.hdf5"),
                os.path.join(dataset_dir, "three_piece_assembly", "demo_src_three_piece_assembly_task_D2/demo.hdf5"),
            ],
            dataset_names=[
                "three_piece_assembly_D0",
                "three_piece_assembly_D1",
                "three_piece_assembly_D2",
            ],
            horizon=500,
        ),
        # coffee
        dict(
            dataset_paths=[
                os.path.join(dataset_dir, "coffee", "demo_src_coffee_task_D0/demo.hdf5"),
                os.path.join(dataset_dir, "coffee", "demo_src_coffee_task_D1/demo.hdf5"),
                os.path.join(dataset_dir, "coffee", "demo_src_coffee_task_D2/demo.hdf5"),
            ],
            dataset_names=[
                "coffee_D0",
                "coffee_D1",
                "coffee_D2",
            ],
            horizon=400,
        ),
        # coffee_preparation
        dict(
            dataset_paths=[
                os.path.join(dataset_dir, "coffee_preparation", "demo_src_coffee_preparation_task_D0/demo.hdf5"),
                os.path.join(dataset_dir, "coffee_preparation", "demo_src_coffee_preparation_task_D1/demo.hdf5"),
            ],
            dataset_names=[
                "coffee_preparation_D0",
                "coffee_preparation_D1",
            ],
            horizon=800,
        ),
        # nut_assembly
        dict(
            dataset_paths=[
                os.path.join(dataset_dir, "nut_assembly", "demo_src_nut_assembly_task_D0/demo.hdf5"),
            ],
            dataset_names=[
                "nut_assembly_D0",
            ],
            horizon=500,
        ),
        # pick_place
        dict(
            dataset_paths=[
                os.path.join(dataset_dir, "pick_place", "demo_src_pick_place_task_D0/demo.hdf5"),
            ],
            dataset_names=[
                "pick_place_D0",
            ],
            horizon=1000,
        ),
        # mug_cleanup
        dict(
            dataset_paths=[
                os.path.join(dataset_dir, "mug_cleanup", "demo_src_mug_cleanup_task_D0/demo.hdf5"),
                os.path.join(dataset_dir, "mug_cleanup", "demo_src_mug_cleanup_task_D1/demo.hdf5"),
                os.path.join(dataset_dir, "mug_cleanup", "demo_src_mug_cleanup_task_O1/demo.hdf5"),
                os.path.join(dataset_dir, "mug_cleanup", "demo_src_mug_cleanup_task_O2/demo.hdf5"),
            ],
            dataset_names=[
                "mug_cleanup_D0",
                "mug_cleanup_D1",
                "mug_cleanup_O1",
                "mug_cleanup_O2",
            ],
            horizon=500,
        ),
        # hammer_cleanup
        dict(
            dataset_paths=[
                os.path.join(dataset_dir, "hammer_cleanup", "demo_src_hammer_cleanup_task_D0/demo.hdf5"),
                os.path.join(dataset_dir, "hammer_cleanup", "demo_src_hammer_cleanup_task_D1/demo.hdf5"),
            ],
            dataset_names=[
                "hammer_cleanup_D0",
                "hammer_cleanup_D1",
            ],
            horizon=500,
        ),
        # kitchen
        dict(
            dataset_paths=[
                os.path.join(dataset_dir, "kitchen", "demo_src_kitchen_task_D0/demo.hdf5"),
                os.path.join(dataset_dir, "kitchen", "demo_src_kitchen_task_D1/demo.hdf5"),
            ],
            dataset_names=[
                "kitchen_D0",
                "kitchen_D1",
            ],
            horizon=800,
        ),
    ]

    ret = []
    for setting in all_settings:
        for mod in ["low_dim", "image"]:
            ret.append(make_gen(os.path.expanduser(base_config), setting, output_dir, mod))
    return ret


def make_gen(base_config, settings, output_dir, mod):
    """
    Specify training configs to generate here.
    """
    generator = ConfigGenerator(
        base_config_file=base_config,
        script_file="", # will be overriden in next step
        base_exp_name="bc_rnn_{}".format(mod),
    )

    # set algo settings for bc-rnn
    modality = mod
    low_dim_keys = settings.get("low_dim_keys", None)
    image_keys = settings.get("image_keys", None)
    crop_size = settings.get("crop_size", None)
    if modality == "low_dim":
        if low_dim_keys is None:
            low_dim_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
    if modality == "image":
        if low_dim_keys is None:
            low_dim_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
        if image_keys is None:
            image_keys = ["agentview_image", "robot0_eye_in_hand_image"]
        if crop_size is None:
            crop_size = [76, 76]
        assert len(crop_size) == 2

    ConfigUtils.set_learning_settings_for_bc_rnn(
        generator=generator,
        group=-1,
        modality=modality,
        seq_length=10,
        low_dim_keys=low_dim_keys,
        image_keys=image_keys,
        crop_size=crop_size,
    )

    # set dataset
    generator.add_param(
        key="train.data", 
        name="ds", 
        group=0, 
        values=settings["dataset_paths"],
        value_names=settings["dataset_names"],
    )

    # rollout settings
    generator.add_param(
        key="experiment.rollout.horizon", 
        name="", 
        group=1, 
        values=[settings["horizon"]],
    )

    # output path
    generator.add_param(
        key="train.output_dir",
        name="", 
        group=-1, 
        values=[
            output_dir,
        ],
    )

    # num data workers 4 by default (for both low-dim and image) and cache mode "low_dim"
    generator.add_param(
        key="train.num_data_workers",
        name="",
        group=-1,
        values=[4],
    )
    generator.add_param(
        key="train.hdf5_cache_mode",
        name="",
        group=-1,
        values=["low_dim"],
    )

    # seed
    generator.add_param(
        key="train.seed",
        name="seed", 
        group=100000, 
        values=[101],
    )

    return generator


def main(args):

    # make config generators
    generators = make_generators(base_config=BASE_CONFIG, dataset_dir=args.dataset_dir, output_dir=args.output_dir)

    if os.path.exists(args.config_dir):
        ans = input("Non-empty dir at {} will be removed.\nContinue (y / n)? \n".format(args.config_dir))
        if ans != "y":
            exit()
        shutil.rmtree(args.config_dir)

    all_json_files, run_lines = config_generator_to_script_lines(generators, config_dir=args.config_dir)

    run_lines = [line.strip() for line in run_lines]

    print("configs")
    print(json.dumps(all_json_files, indent=4))
    print("runs")
    print(json.dumps(run_lines, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_dir",
        type=str,
        default=os.path.expanduser(CONFIG_DIR),
        help="set base folder for where to generate new config files for data generation",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=os.path.expanduser(DATASET_DIR),
        help="set path to folder with datasets",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.expanduser(OUTPUT_DIR),
        help="set base folder for where to generate new config files for data generation",
    )

    args = parser.parse_args()
    main(args)
