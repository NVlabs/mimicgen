# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
A script to playback demonstrations (using visual observations and the pygame renderer)
in order to allow a user to annotate portions of the demonstrations. This is useful 
to annotate the end of each object-centric subtask in each source demonstration used 
by MimicGen, as an alternative to implementing subtask termination signals directly 
in the simulation environment.

Examples:

    # specify the sequence of signals that should be annotated and the dataset images to render on-screen
    python annotate_subtasks.py --dataset /path/to/demo.hdf5 --signals grasp_1 insert_1 grasp_2 \
        --render_image_names agentview_image robot0_eye_in_hand_image

    # limit annotation to first 2 demos
    python annotate_subtasks.py --dataset /path/to/demo.hdf5 --signals grasp_1 insert_1 grasp_2 \
        --render_image_names agentview_image robot0_eye_in_hand_image --n 2

    # limit annotation to demo 2 and 3
    python annotate_subtasks.py --dataset /path/to/demo.hdf5 --signals grasp_1 insert_1 grasp_2 \
        --render_image_names agentview_image robot0_eye_in_hand_image --n 2 --start 1

    # scale up dataset images when rendering to screen by factor of 10
    python annotate_subtasks.py --dataset /path/to/demo.hdf5 --signals grasp_1 insert_1 grasp_2 \
        --render_image_names agentview_image robot0_eye_in_hand_image --image_scale 10

"""

import os
import sys
import h5py
import argparse
import imageio
import numpy as np

# for rendering images on-screen
import cv2
try:
    import pygame
except ImportError as e:
    print("Got error: {}".format(e))
    print("")
    print("pygame is required. Please install with `pip install pygame`")

import robomimic
from robomimic.utils.file_utils import get_env_metadata_from_dataset

import mimicgen
import mimicgen.utils.file_utils as MG_FileUtils
import mimicgen.utils.misc_utils as MiscUtils


# scaling size for images when rendering to screen
# IMAGE_SCALE = 10
IMAGE_SCALE = 5
# IMAGE_SCALE = 1

# Grid of playback rates for the user to cycle through (e.g. 1hz, 5 hz, ...)
RATE_GRID = MiscUtils.Grid(
    values=[1, 5, 10, 20, 40],
    initial_ind=0,
)


def print_keyboard_commands():
    """
    Helper function to print keyboard annotation commands.
    """
    def print_command(char, info):
        char += " " * (11 - len(char))
        print("{}\t{}".format(char, info))

    print("")
    print_command("Keys", "Command")
    print_command("up-down", "increase / decrease playback speed")
    print_command("left-right", "seek left / right by N frames")
    print_command("spacebar", "press and release to annotate the end of a subtask")
    print_command("f", "next demo and save annotations")
    print_command("r", "repeat demo and clear annotations")
    print("")


def make_pygame_screen(
    traj_grp,
    image_names,
    image_scale,
):
    """
    Makes pygame screen.

    Args:
        traj_grp (h5py.Group): group for a demonstration trajectory
        image_names (list): list of image names that will be used for rendering
        image_scale (int): scaling factor for the image to diplay in window

    Returns:
        screen: pygame screen object
    """
    # grab first image from all image modalities to infer size of window
    im = [traj_grp["obs/{}".format(k)][0] for k in image_names]
    frame = np.concatenate(im, axis=1)
    width, height = frame.shape[:2]
    width *= image_scale
    height *= image_scale
    screen = pygame.display.set_mode((height, width))
    return screen


def handle_pygame_events(
    frame_ind,
    subtask_signals,
    subtask_ind,
    rate_obj,
    need_repeat,
    annotation_done,
    playback_rate_grid,
):
    """
    Reads events from pygame window in order to provide the
    following keyboard annotation functionality:

        up-down     | increase / decrease playback speed
        left-right  | seek left / right by N frames
        spacebar    | press and release to annotate the end of a subtask
        f           | next demo and save annotations
        r           | repeat demo and clear annotations

    Args:
        frame_ind (int): index of current frame in demonstration
        subtask_signals (list): list of subtask termination signals that we will annotate
        subtask_ind (int): index of current subtask (state variable)
        rate_obj (Rate): rate object to maintain playback rate
        need_repeat (bool): whether the demo should be repeated (state variable)
        annotation_done (bool): whether user is done annotating this demo (state variable)
        playback_rate_grid (Grid): grid object to easily toggle between different playback rates

    Returns:
        subtask_end_ind (int or None): end index for current subtask, annotated by human, or None if no annotation
        subtask_ind (int): possibly updated subtask index
        need_repeat (bool): possibly updated value
        annotation_done (bool): possibly updated value
        seek (int): how much to seek forward or backward in demonstration (value read from user command)
    """

    subtask_end_ind = None
    seek = 0
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            sys.exit()

        if event.type == pygame.KEYUP:
            # print("released key {}".format(event.key))
            if event.key == pygame.K_SPACE:
                # annotate end of current subtask and move on to next subtask
                subtask_end_ind = frame_ind + 1
                print("")
                print("*" * 50)
                print("cmd: annotated end of subtask {} (signal {}) at index {}".format(subtask_ind, subtask_signals[subtask_ind], frame_ind + 1))
                print("*" * 50)
                print("")
                subtask_ind += 1
            elif event.key == pygame.K_UP:
                # speed up traversal
                rate_obj.update_hz(playback_rate_grid.next())
                print("cmd: playback rate increased to {} hz".format(rate_obj.hz))
            elif event.key == pygame.K_DOWN:
                # slow down traversal
                rate_obj.update_hz(playback_rate_grid.prev())
                print("cmd: playback rate decreased to {} hz".format(rate_obj.hz))
            elif event.key == pygame.K_LEFT:
                # seek left
                seek = -10
                print("cmd: seek {} frames".format(seek))
            elif event.key == pygame.K_RIGHT:
                # seek right
                seek = 10
                print("cmd: seek {} frames".format(seek))
            elif event.key == pygame.K_r:
                # repeat demo
                need_repeat = True
                print("cmd: repeat demo")
            elif event.key == pygame.K_f:
                # next demo
                annotation_done = True
                print("cmd: next demo")

    return subtask_end_ind, subtask_ind, need_repeat, annotation_done, seek


def annotate_subtasks_in_trajectory(
    ep,
    traj_grp,
    subtask_signals,
    screen,
    video_skip, 
    image_names,
    playback_rate_grid,
):
    """
    This function reads all "rgb" observations in the dataset trajectory and
    writes them into a video.

    Args:
        ep (str): name of hdf5 group for this demo
        traj_grp (hdf5 file group): hdf5 group which corresponds to the dataset trajectory to annotate
        subtask_signals (list): list of subtask termination signals that will be annotated
        screen: pygame screen
        video_skip (int): determines rate at which environment frames are written to video
        image_names (list): determines which image observations are used for rendering. Pass more than
            one to output a video with multiple image observations concatenated horizontally.
        playback_rate_grid (Grid): grid object to easily toggle between different playback rates
    """
    assert image_names is not None, "error: must specify at least one image observation to use in @image_names"

    traj_len = traj_grp["actions"].shape[0]

    rate_obj = MiscUtils.Rate(hz=playback_rate_grid.get())
    rate_measure = MiscUtils.RateMeasure(name="rate_measure")

    # repeat this demonstration until we have permission to move on
    annotation_done = False
    while not annotation_done:
        print("Starting annotation for demo: {}".format(ep))
        print_keyboard_commands()

        need_repeat = False
        subtask_end_inds = []

        # keep looping through the video, reading user input from keyboard, until
        # user indicates that demo is done being annotated
        frame_ind = 0
        subtask_ind = 0
        should_add_border_to_frame = False
        while (not need_repeat) and (not annotation_done):

            # maybe render frame to screen
            if frame_ind % video_skip == 0:
                # concatenate image obs together
                im = [traj_grp["obs/{}".format(k)][frame_ind] for k in image_names]
                frame = np.concatenate(im, axis=1)
                # upscale frame to appropriate resolution
                frame = cv2.resize(frame, 
                    dsize=(frame.shape[1] * IMAGE_SCALE, frame.shape[0] * IMAGE_SCALE), 
                    interpolation=cv2.INTER_CUBIC)
                # maybe add red border
                if should_add_border_to_frame:
                    frame = MiscUtils.add_red_border_to_frame(frame)
                # write frame to window
                frame = frame.transpose((1, 0, 2))
                pygame.pixelcopy.array_to_surface(screen, frame)
                pygame.display.update()

            subtask_end_ind, subtask_ind, need_repeat, annotation_done, seek = handle_pygame_events(
                frame_ind=frame_ind,
                subtask_signals=subtask_signals,
                subtask_ind=subtask_ind,
                rate_obj=rate_obj,
                need_repeat=need_repeat,
                annotation_done=annotation_done,
                playback_rate_grid=playback_rate_grid,
            )

            if subtask_end_ind is not None:
                # store new annotation and toggle rendering of red border
                subtask_end_inds.append(subtask_end_ind)
                should_add_border_to_frame = (not should_add_border_to_frame)

            # try to enforce rate
            rate_obj.sleep()
            rate_measure.measure()

            # increment frame index appropriately (either by 1 or by seek amount), then
            # clamp within bounds
            mask = int(seek != 0)
            frame_ind += (1 - mask) * 1 + mask * seek
            frame_ind = max(min(frame_ind, traj_len - 1), 0)

        # if we don't need to repeat the demo, we're done
        annotation_done = annotation_done or (not need_repeat)

        # check that we got the right number of annotations
        if len(subtask_end_inds) != len(subtask_signals):
            print("")
            print("*" * 50)
            print("Number of termination annotations {} does not match expected number {}...".format(len(subtask_end_inds), len(subtask_signals)))
            print("Repeating annotation.")
            print("*" * 50)
            print("")
            annotation_done = False

    # write subtask_termination_signals to hdf5
    assert len(subtask_end_inds) == len(subtask_signals)

    if "subtask_term_signals" in traj_grp["datagen_info"]:
        del traj_grp["datagen_info"]["subtask_term_signals"]
    for subtask_ind in range(len(subtask_signals)):
        # subtask termination signal is 0 until subtask is complete, and 1 afterwards
        subtask_signal_array = np.ones(traj_len, dtype=int)
        subtask_signal_array[:subtask_end_inds[subtask_ind]] = 0
        traj_grp.create_dataset("datagen_info/subtask_term_signals/{}".format(subtask_signals[subtask_ind]), data=subtask_signal_array)

    # report rate measurements
    print("\nFrame Rate (Hz) Statistics for demo {} annotation".format(ep))
    print(rate_measure)


def annotate_subtasks(args):
    # Auto-fill camera rendering info if not specified
    if args.render_image_names is None:
        env_meta = get_env_metadata_from_dataset(dataset_path=args.dataset)
        args.render_image_names = RobomimicUtils.get_default_env_cameras(env_meta=env_meta)

    # get demonstrations to annotate
    dataset_path = args.dataset
    demo_keys = MG_FileUtils.get_all_demos_from_dataset(
        dataset_path=dataset_path,
        filter_key=args.filter_key,
        start=args.start,
        n=args.n,
    )

    # Verify that the dataset has been processed and has datagen_info.
    MG_FileUtils.get_env_interface_info_from_dataset(
        dataset_path=dataset_path,
        demo_keys=demo_keys,
    )

    # Open the file in read-write mode to add in annotations as subtask_term_signals in datagen_info.
    f = h5py.File(dataset_path, "a")

    # make pygame screen first
    screen = make_pygame_screen(
        traj_grp=f["data/{}".format(demo_keys[0])],
        image_names=args.render_image_names,
        image_scale=args.image_scale,
    )

    for ind in range(len(demo_keys)):
        ep = demo_keys[ind]
        print("Annotating episode: {}".format(ep))

        annotate_subtasks_in_trajectory(
            ep=ep,
            traj_grp=f["data/{}".format(ep)],
            subtask_signals=args.signals,
            screen=screen, 
            video_skip=args.video_skip,
            image_names=args.render_image_names,
            playback_rate_grid=RATE_GRID,
        )

    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to hdf5 dataset which will be modified in-place",
    )
    parser.add_argument(
        "--signals",
        type=str,
        nargs='+',
        required=True,
        help="specify sequence of subtask termination signals for all except last subtask -- these will be written using the annotations",
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
        help="(optional) stop after n trajectories are annotated",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="(optional) start after this many trajectories in the dataset",
    )
    parser.add_argument(
        "--video_skip",
        type=int,
        default=1,
        help="(optional) render frames on-screen every n steps",
    )
    parser.add_argument(
        "--render_image_names",
        type=str,
        nargs='+',
        default=None,
        help="(optional) camera name(s) / image observation(s) to use for rendering on-screen or to video. Default is"
             "None, which corresponds to a predefined camera for each env type",
    )
    parser.add_argument(
        "--image_scale",
        type=int,
        default=5,
        help="(optional) scaling size for images when rendering to screen",
    )

    args = parser.parse_args()
    annotate_subtasks(args)
