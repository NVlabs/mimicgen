# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
A script to visualize each subtask in a source demonstration. This is a useful way to 
debug the subtask termination signals in a set of source demonstrations, as well as 
the choice of maximum subtask termination offsets. 

Examples:

    # render on-screen
    python visualize_subtasks.py --dataset /path/to/demo.hdf5 --config /path/to/config.json --render

    # render to video
    python visualize_subtasks.py --dataset /path/to/demo.hdf5 --config /path/to/config.json --video_path /path/to/video.mp4

    # specify subtask information manually instead of using a config
    python visualize_subtasks.py --dataset /path/to/demo.hdf5 --signals grasp_1 insert_1 grasp_2 --offsets 10 10 10 --render

"""

import os
import sys
import json
import h5py
import argparse
import imageio
import numpy as np

import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.envs.env_base import EnvBase
from robomimic.utils.file_utils import get_env_metadata_from_dataset

import mimicgen
import mimicgen.utils.file_utils as MG_FileUtils
import mimicgen.utils.robomimic_utils as RobomimicUtils
from mimicgen.utils.misc_utils import add_red_border_to_frame
from mimicgen.configs import MG_TaskSpec


def visualize_subtasks_with_env(
    env,
    initial_state,
    states,
    subtask_end_indices,
    render=False,
    video_writer=None,
    video_skip=5,
    camera_names=None,
):
    """
    Helper function to visualize each subtask in a trajectory using the simulator environment. 
    If using on-screen rendering, the script will pause for input at the end of each subtask. If 
    writing to a video, each subtask will toggle between having a red border around each 
    frame and no border in the video.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (list): list of simulation states to load
        subtask_end_indices (list): list containing the end index for each subtask
        render (bool): if True, render on-screen
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.
    """
    assert isinstance(env, EnvBase)

    write_video = (video_writer is not None)
    video_count = 0
    assert not (render and write_video)
    assert render or write_video

    # load the initial state
    env.reset()
    env.reset_to(initial_state)
    traj_len = len(states)

    cur_subtask_ind = 0
    should_add_border_to_frame = False
    for i in range(traj_len):
        # reset to state
        env.reset_to({"states" : states[i]})

        # whether we are on last index of current subtask
        is_last_subtask_ind = (i == subtask_end_indices[cur_subtask_ind] - 1)

        # on-screen render
        if render:
            env.render(mode="human", camera_name=camera_names[0])

            if is_last_subtask_ind:
                # pause on last index of current subtask
                input("Pausing after subtask {} execution. Press any key to continue...".format(cur_subtask_ind))
                cur_subtask_ind += 1

        # video render
        if write_video:
            if video_count % video_skip == 0:
                video_img = []
                for cam_name in camera_names:
                    video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                if should_add_border_to_frame:
                    video_img = add_red_border_to_frame(video_img)
                video_writer.append_data(video_img)
            video_count += 1

            if is_last_subtask_ind:
                # toggle whether to add red border for next subtask
                should_add_border_to_frame = (not should_add_border_to_frame)
                cur_subtask_ind += 1


def visualize_subtasks_with_obs(
    traj_grp,
    subtask_end_indices,
    video_writer,
    video_skip=5,
    image_names=None,
):
    """
    Helper function to visualize each subtask in a trajectory by writing image observations
    to a video. Each subtask will toggle between having a red border around each 
    frame and no border in the video.

    Args:
        traj_grp (hdf5 file group): hdf5 group which corresponds to the dataset trajectory to playback
        subtask_end_indices (list): list containing the end index for each subtask
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        image_names (list): determines which image observations are used for rendering. Pass more than
            one to output a video with multiple image observations concatenated horizontally.
    """
    assert image_names is not None, "error: must specify at least one image observation to use in @image_names"
    video_count = 0

    traj_len = traj_grp["actions"].shape[0]
    should_add_border_to_frame = False
    cur_subtask_ind = 0
    for i in range(traj_len):
        # whether we are on last index of current subtask
        is_last_subtask_ind = (i == subtask_end_indices[cur_subtask_ind] - 1)

        if video_count % video_skip == 0:
            # concatenate image obs together
            im = [traj_grp["obs/{}".format(k)][i] for k in image_names]
            frame = np.concatenate(im, axis=1)
            if should_add_border_to_frame:
                frame = add_red_border_to_frame(frame)
            video_writer.append_data(frame)
        video_count += 1

        if is_last_subtask_ind:
            # toggle whether to add red border for next subtask
            should_add_border_to_frame = (not should_add_border_to_frame)
            cur_subtask_ind += 1


def visualize_subtasks(args):
    # some arg checking
    write_video = (args.video_path is not None)

    # either on-screen or video but not both
    assert not (args.render and write_video)

    # either config or signals and offsets should be provided, but not both
    assert (args.config is not None) or ((args.signals is not None) and (args.offsets is not None))
    assert (args.config is None) or ((args.signals is None) and (args.offsets is None))

    # Auto-fill camera rendering info if not specified
    if args.render_image_names is None:
        env_meta = get_env_metadata_from_dataset(dataset_path=args.dataset)
        args.render_image_names = RobomimicUtils.get_default_env_cameras(env_meta=env_meta)

    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.render_image_names) == 1

    if args.use_obs:
        assert not args.render
    else:
        # create environment only if not playing back with observations

        # need to make sure ObsUtils knows which observations are images, but it doesn't matter 
        # for playback since observations are unused. Pass a dummy spec here.
        dummy_spec = dict(
            obs=dict(
                    low_dim=["robot0_eef_pos"],
                    rgb=[],
                    # image=[],
                ),
        )
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

        env_meta = get_env_metadata_from_dataset(dataset_path=args.dataset)
        env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=args.render, render_offscreen=write_video)
        
        # some operations for playback are env-type-specific
        is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    # get demonstrations to visualize subtasks for
    dataset_path = args.dataset
    demo_keys = MG_FileUtils.get_all_demos_from_dataset(
        dataset_path=dataset_path,
        filter_key=args.filter_key,
        start=None,
        n=args.n,
    )

    # we will parse the source dataset to get subtask boundaries using either the task spec in the
    # provided config or the provided arguments
    task_spec = None
    subtask_term_signals = None
    subtask_term_offset_ranges = None
    if args.config is not None:
        with open(args.config, 'r') as f_config:
            mg_config = json.load(f_config)
        task_spec = MG_TaskSpec.from_json(json_dict=mg_config["task"]["task_spec"])
    else:
        subtask_term_signals = args.signals + [None]
        subtask_term_offset_ranges = [(0, offset) for offset in args.offsets] + [None]

    # parse dataset to get subtask boundaries
    _, subtask_indices, _, subtask_term_offset_ranges_ret = MG_FileUtils.parse_source_dataset(
        dataset_path=dataset_path,
        demo_keys=demo_keys,
        task_spec=task_spec,
        subtask_term_signals=subtask_term_signals,
        subtask_term_offset_ranges=subtask_term_offset_ranges,
    )

    # apply maximum offset to each subtask boundary
    offsets_to_apply = [x[1] for x in subtask_term_offset_ranges_ret]
    offsets_to_apply[-1] = 0
    # subtask_indices is shape (N, S, 2) where N is num demos, S is num subtasks and each entry is 2-tuple of start and end
    subtask_end_indices = subtask_indices[:, :, 1]
    subtask_end_indices = subtask_end_indices + np.array(offsets_to_apply)[None] # offsets shape (1, S)

    f = h5py.File(args.dataset, "r")

    # maybe dump video
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    for ind in range(len(demo_keys)):
        ep = demo_keys[ind]
        print("Playing back episode: {}".format(ep))

        if args.use_obs:
            traj_grp = f["data/{}".format(ep)]
            visualize_subtasks_with_obs(
                traj_grp=traj_grp,
                subtask_end_indices=subtask_end_indices[ind],
                video_writer=video_writer,
                video_skip=args.video_skip,
                image_names=args.render_image_names,
            )
            continue

        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        if is_robosuite_env:
            initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
        visualize_subtasks_with_env(
            env=env,
            initial_state=initial_state,
            states=states,
            subtask_end_indices=subtask_end_indices[ind],
            render=args.render,
            video_writer=video_writer,
            video_skip=args.video_skip,
            camera_names=args.render_image_names,
        )

    f.close()
    if write_video:
        video_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
        required=True,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="use config to infer sequence of subtask signals and offsets",
    )
    parser.add_argument(
        "--signals",
        type=str,
        nargs='+',
        default=None,
        help="specify sequence of subtask termination signals for all except last subtask",
    )
    parser.add_argument(
        "--offsets",
        type=int,
        nargs='+',
        default=None,
        help="specify sequence of maximum subtask termination offsets for all except last subtask",
    )
    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
        help="(optional) filter key, to select a subset of trajectories in the file",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories",
    )
    parser.add_argument(
        "--use-obs",
        action='store_true',
        help="visualize trajectories with dataset image observations instead of simulator",
    )
    parser.add_argument(
        "--render",
        action='store_true',
        help="on-screen rendering",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render trajectories to this video file path",
    )
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )
    parser.add_argument(
        "--render_image_names",
        type=str,
        nargs='+',
        default=None,
        help="(optional) camera name(s) / image observation(s) to use for rendering on-screen or to video",
    )
    args = parser.parse_args()
    visualize_subtasks(args)
