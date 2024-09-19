# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

__version__ = "1.0.1"

# try to import all environment interfaces here
try:
    from mimicgen.env_interfaces.robosuite import *
except ImportError as e:
    print("WARNING: robosuite environment interfaces not imported...")
    print("Got error: {}".format(e))

# import tasks to make sure they are added to robosuite task registry
try:
    from mimicgen.envs.robosuite.threading import *
    from mimicgen.envs.robosuite.coffee import *
    from mimicgen.envs.robosuite.three_piece_assembly import *
    from mimicgen.envs.robosuite.mug_cleanup import *
    from mimicgen.envs.robosuite.stack import *
    from mimicgen.envs.robosuite.nut_assembly import *
    from mimicgen.envs.robosuite.pick_place import *
except ImportError as e:
    print("WARNING: robosuite environments not imported...")
    print("Got error: {}".format(e))

try:
    from mimicgen.envs.robosuite.hammer_cleanup import *
    from mimicgen.envs.robosuite.kitchen import *
except ImportError as e:
    print("WARNING: robosuite task zoo environments not imported, possibly because robosuite_task_zoo is not installed...")
    print("Got error: {}".format(e))

# stores released dataset links and rollout horizons in global dictionary.
# Structure is given below for each type of dataset:

# robosuite
# {
#   task:
#       url: path in Hugging Face repo
#       horizon: value
#   ...
# }
DATASET_REGISTRY = {}

# Hugging Face repo ID
HF_REPO_ID = "amandlek/mimicgen_datasets"


def register_dataset_link(dataset_type, task, link, horizon):
    """
    Helper function to register dataset link in global dictionary.
    Also takes a @horizon parameter - this corresponds to the evaluation
    rollout horizon that should be used during training.

    Args:
        dataset_type (str): identifies the type of dataset (e.g. source human data, 
            core experiment data, object transfer data)
        task (str): name of task for this dataset
        link (str): download link for the dataset
        horizon (int): evaluation rollout horizon that should be used with this dataset
    """
    if dataset_type not in DATASET_REGISTRY:
        DATASET_REGISTRY[dataset_type] = dict()
    DATASET_REGISTRY[dataset_type][task] = dict(url=link, horizon=horizon)


def register_all_links():
    """
    Record all dataset links in this function.
    """

    ### source human datasets used to generate all data ###
    dataset_type = "source"

    # info for each dataset (name, evaluation horizon, link)
    dataset_infos = [
        ("hammer_cleanup", 500, "source/hammer_cleanup.hdf5"),
        ("kitchen", 800, "source/kitchen.hdf5"),
        ("coffee", 400, "source/coffee.hdf5"),
        ("coffee_preparation", 800, "source/coffee_preparation.hdf5"),
        ("nut_assembly", 500, "source/nut_assembly.hdf5"),
        ("mug_cleanup", 500, "source/mug_cleanup.hdf5"),
        ("pick_place", 1000, "source/pick_place.hdf5"),
        ("square", 400, "source/square.hdf5"),
        ("stack", 400, "source/stack.hdf5"),
        ("stack_three", 400, "source/stack_three.hdf5"),
        ("threading", 400, "source/threading.hdf5"),
        ("three_piece_assembly", 500, "source/three_piece_assembly.hdf5"),
    ]
    for task, horizon, link in dataset_infos:
        register_dataset_link(
            dataset_type=dataset_type,
            task=task,
            horizon=horizon,
            link=link,
        )

    ### core generated datasets ###
    dataset_type = "core"
    dataset_infos = [
        ("hammer_cleanup_d0", 500, "core/hammer_cleanup_d0.hdf5"),
        ("hammer_cleanup_d1", 500, "core/hammer_cleanup_d1.hdf5"),
        ("kitchen_d0", 800, "core/kitchen_d0.hdf5"),
        ("kitchen_d1", 800, "core/kitchen_d1.hdf5"),
        ("coffee_d0", 400, "core/coffee_d0.hdf5"),
        ("coffee_d1", 400, "core/coffee_d1.hdf5"),
        ("coffee_d2", 400, "core/coffee_d2.hdf5"),
        ("coffee_preparation_d0", 800, "core/coffee_preparation_d0.hdf5"),
        ("coffee_preparation_d1", 800, "core/coffee_preparation_d1.hdf5"),
        ("nut_assembly_d0", 500, "core/nut_assembly_d0.hdf5"),
        ("mug_cleanup_d0", 500, "core/mug_cleanup_d0.hdf5"),
        ("mug_cleanup_d1", 500, "core/mug_cleanup_d1.hdf5"),
        ("pick_place_d0", 1000, "core/pick_place_d0.hdf5"),
        ("square_d0", 400, "core/square_d0.hdf5"),
        ("square_d1", 400, "core/square_d1.hdf5"),
        ("square_d2", 400, "core/square_d2.hdf5"),
        ("stack_d0", 400, "core/stack_d0.hdf5"),
        ("stack_d1", 400, "core/stack_d1.hdf5"),
        ("stack_three_d0", 400, "core/stack_three_d0.hdf5"),
        ("stack_three_d1", 400, "core/stack_three_d1.hdf5"),
        ("threading_d0", 400, "core/threading_d0.hdf5"),
        ("threading_d1", 400, "core/threading_d1.hdf5"),
        ("threading_d2", 400, "core/threading_d2.hdf5"),
        ("three_piece_assembly_d0", 500, "core/three_piece_assembly_d0.hdf5"),
        ("three_piece_assembly_d1", 500, "core/three_piece_assembly_d1.hdf5"),
        ("three_piece_assembly_d2", 500, "core/three_piece_assembly_d2.hdf5"),
    ]
    for task, horizon, link in dataset_infos:
        register_dataset_link(
            dataset_type=dataset_type,
            task=task,
            horizon=horizon,
            link=link,
        )

    ### object transfer datasets ###
    dataset_type = "object"
    dataset_infos = [
        ("mug_cleanup_o1", 500, "object/mug_cleanup_o1.hdf5"),
        ("mug_cleanup_o2", 500, "object/mug_cleanup_o2.hdf5"),
    ]
    for task, horizon, link in dataset_infos:
        register_dataset_link(
            dataset_type=dataset_type,
            task=task,
            horizon=horizon,
            link=link,
        )

    ### robot transfer datasets ###
    dataset_type = "robot"
    dataset_infos = [
        ("square_d0_panda", 400, "robot/square_d0_panda.hdf5"),
        ("square_d0_sawyer", 400, "robot/square_d0_sawyer.hdf5"),
        ("square_d0_iiwa", 400, "robot/square_d0_iiwa.hdf5"),
        ("square_d0_ur5e", 400, "robot/square_d0_ur5e.hdf5"),
        ("square_d1_panda", 400, "robot/square_d1_panda.hdf5"),
        ("square_d1_sawyer", 400, "robot/square_d1_sawyer.hdf5"),
        ("square_d1_iiwa", 400, "robot/square_d1_iiwa.hdf5"),
        ("square_d1_ur5e", 400, "robot/square_d1_ur5e.hdf5"),
        ("threading_d0_panda", 400, "robot/threading_d0_panda.hdf5"),
        ("threading_d0_sawyer", 400, "robot/threading_d0_sawyer.hdf5"),
        ("threading_d0_iiwa", 400, "robot/threading_d0_iiwa.hdf5"),
        ("threading_d0_ur5e", 400, "robot/threading_d0_ur5e.hdf5"),
        ("threading_d1_panda", 400, "robot/threading_d1_panda.hdf5"),
        ("threading_d1_sawyer", 400, "robot/threading_d1_sawyer.hdf5"),
        ("threading_d1_iiwa", 400, "robot/threading_d1_iiwa.hdf5"),
        ("threading_d1_ur5e", 400, "robot/threading_d1_ur5e.hdf5"),
    ]
    for task, horizon, link in dataset_infos:
        register_dataset_link(
            dataset_type=dataset_type,
            task=task,
            horizon=horizon,
            link=link,
        )

    ### large_interpolation datasets (larger eval horizons due to longer interpolation segments) ###
    dataset_type = "large_interpolation"
    dataset_infos = [
        ("coffee_d1", 550, "large_interpolation/coffee_d1.hdf5"),
        ("pick_place_d0", 1400, "large_interpolation/pick_place_d0.hdf5"),
        ("square_d1", 500, "large_interpolation/square_d1.hdf5"),
        ("stack_d1", 500, "large_interpolation/stack_d1.hdf5"),
        ("threading_d1", 500, "large_interpolation/threading_d1.hdf5"),
        ("three_piece_assembly_d1", 700, "large_interpolation/three_piece_assembly_d1.hdf5"),
    ]
    for task, horizon, link in dataset_infos:
        register_dataset_link(
            dataset_type=dataset_type,
            task=task,
            horizon=horizon,
            link=link,
        )

register_all_links()
