# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Collection of utilities related to robomimic.
"""
import sys
import json
import traceback
import argparse
from copy import deepcopy

import robomimic
from robomimic.utils.log_utils import PrintLogger
import robomimic.utils.env_utils as EnvUtils
from robomimic.scripts.playback_dataset import playback_dataset, DEFAULT_CAMERAS

from mimicgen.utils.misc_utils import deep_update


def make_print_logger(txt_file):
    """
    Makes a logger that mirrors stdout and stderr to a text file.

    Args:
        txt_file (str): path to txt file to write
    """
    logger = PrintLogger(txt_file)
    sys.stdout = logger
    sys.stderr = logger
    return logger


def create_env(
    env_meta,
    env_name=None,
    env_class=None,
    robot=None,
    gripper=None,
    env_meta_update_kwargs=None,
    camera_names=None,
    camera_height=84,
    camera_width=84,
    render=None, 
    render_offscreen=None, 
    use_image_obs=None, 
    use_depth_obs=None, 
):
    """
    Helper function to create the environment from dataset metadata and arguments.

    Args:
        env_meta (dict): environment metadata compatible with robomimic, see
            https://robomimic.github.io/docs/modules/environments.html
        env_name (str or None): if provided, override environment name 
            in @env_meta
        env_class (class or None): if provided, use this class instead of the
            one inferred from @env_meta
        robot (str or None): if provided, override the robot argument in
            @env_meta. Currently only supported by robosuite environments.
        gripper (str or None): if provided, override the gripper argument in
            @env_meta. Currently only supported by robosuite environments.
        env_meta_update_kwargs (dict or None): if provided, update the environment
            metadata with these kwargs
        camera_names (list of str or None): list of camera names that correspond to image observations
        camera_height (int): camera height for all cameras
        camera_width (int): camera width for all cameras
        render (bool or None): optionally override rendering behavior
        render_offscreen (bool or None): optionally override rendering behavior
        use_image_obs (bool or None): optionally override rendering behavior
        use_depth_obs (bool or None): optionally override rendering behavior
    """
    env_meta = deepcopy(env_meta)

    # maybe override some settings in environment metadata
    if env_name is not None:
        env_meta["env_name"] = env_name
    if robot is not None:
        # for now, only support this argument for robosuite environments
        assert EnvUtils.is_robosuite_env(env_meta)
        assert robot in ["IIWA", "Sawyer", "UR5e", "Panda", "Jaco", "Kinova3"]
        env_meta["env_kwargs"]["robots"] = [robot]
    if gripper is not None:
        # for now, only support this argument for robosuite environments
        assert EnvUtils.is_robosuite_env(env_meta)
        assert gripper in ["PandaGripper", "RethinkGripper", "Robotiq85Gripper", "Robotiq140Gripper"]
        env_meta["env_kwargs"]["gripper_types"] = [gripper]

    # maybe update environment metadata with additional kwargs
    if env_meta_update_kwargs is not None:
        deep_update(env_meta, env_meta_update_kwargs)

    if camera_names is None:
        camera_names = []

    # create environment
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        env_class=env_class,
        camera_names=camera_names, 
        camera_height=camera_height, 
        camera_width=camera_width, 
        reward_shaping=False,
        render=render,
        render_offscreen=render_offscreen,
        use_image_obs=use_image_obs,
        use_depth_obs=use_depth_obs,
    )

    return env


def make_dataset_video(
    dataset_path,
    video_path,
    num_render=None,
    render_image_names=None,
    use_obs=False,
    video_skip=5,
):
    """
    Helper function to set up args and call @playback_dataset from robomimic
    to get video of generated dataset.
    """
    print("\nmake_dataset_video(\n\tdataset_path={},\n\tvideo_path={},{}\n)".format(
        dataset_path,
        video_path,
        "\n\tnum_render={},".format(num_render) if num_render is not None else "",
    ))
    playback_args = argparse.Namespace()
    playback_args.dataset = dataset_path
    playback_args.filter_key = None
    playback_args.n = num_render
    playback_args.use_obs = use_obs
    playback_args.use_actions = False
    playback_args.render = False
    playback_args.video_path = video_path
    playback_args.video_skip = video_skip
    playback_args.render_image_names = render_image_names
    if (render_image_names is None):
        # default robosuite
        playback_args.render_image_names = ["agentview"]
    playback_args.render_depth_names = None
    playback_args.first = False

    try:
        playback_dataset(playback_args)
    except Exception as e:
        res_str = "playback failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
        print(res_str)


def get_default_env_cameras(env_meta):
    """
    Get the default set of cameras for a particular robomimic environment type.

    Args:
        env_meta (dict): environment metadata compatible with robomimic, see
            https://robomimic.github.io/docs/modules/environments.html

    Returns:
        camera_names (list of str): list of camera names that correspond to image observations
    """
    return DEFAULT_CAMERAS[EnvUtils.get_env_type(env_meta=env_meta)]
