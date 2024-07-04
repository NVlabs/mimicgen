# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Base class for data generator.
"""
import h5py
import sys
import numpy as np

import mimicgen
import mimicgen.utils.pose_utils as PoseUtils
import mimicgen.utils.file_utils as MG_FileUtils

from mimicgen.configs.task_spec import MG_TaskSpec
from mimicgen.datagen.datagen_info import DatagenInfo
from mimicgen.datagen.selection_strategy import make_selection_strategy
from mimicgen.datagen.waypoint import WaypointSequence, WaypointTrajectory


class DataGenerator(object):
    """
    The main data generator object that loads a source dataset, parses it, and 
    generates new trajectories.
    """
    def __init__(
        self,
        task_spec,
        dataset_path,
        demo_keys=None,
    ):
        """
        Args:
            task_spec (MG_TaskSpec instance): task specification that will be
                used to generate data
            dataset_path (str): path to hdf5 dataset to use for generation
            demo_keys (list of str): list of demonstration keys to use
                in file. If not provided, all demonstration keys will be
                used.
        """
        assert isinstance(task_spec, MG_TaskSpec)
        self.task_spec = task_spec
        self.dataset_path = dataset_path

        # sanity check on task spec offset ranges - final subtask should not have any offset randomization
        assert self.task_spec[-1]["subtask_term_offset_range"][0] == 0
        assert self.task_spec[-1]["subtask_term_offset_range"][1] == 0

        # demonstration keys to use from hdf5 as source dataset
        if demo_keys is None:
            # get all demonstration keys from file
            demo_keys = MG_FileUtils.get_all_demos_from_dataset(dataset_path=self.dataset)
        self.demo_keys = demo_keys

        # parse source dataset
        self._load_dataset(dataset_path=dataset_path, demo_keys=demo_keys)

    def _load_dataset(self, dataset_path, demo_keys):
        """
        Load important information from a dataset into internal memory.
        """
        print("\nDataGenerator: loading dataset at path {}...".format(dataset_path))
        self.src_dataset_infos, self.src_subtask_indices, self.subtask_names, _ = MG_FileUtils.parse_source_dataset(
            dataset_path=dataset_path,
            demo_keys=demo_keys,
            task_spec=self.task_spec,
        )
        print("\nDataGenerator: done loading\n")

    def __repr__(self):
        """
        Pretty print this object.
        """
        msg = str(self.__class__.__name__)
        msg += " (\n\tdataset_path={}\n\tdemo_keys={}\n)".format(
            self.dataset_path,
            self.demo_keys,
        )
        return msg

    def randomize_subtask_boundaries(self):
        """
        Apply random offsets to sample subtask boundaries according to the task spec.
        Recall that each demonstration is segmented into a set of subtask segments, and the
        end index of each subtask can have a random offset.
        """

        # initial subtask start and end indices - shape (N, S, 2)
        src_subtask_indices = np.array(self.src_subtask_indices)

        # for each subtask (except last one), sample all end offsets at once for each demonstration
        # add them to subtask end indices, and then set them as the start indices of next subtask too
        for i in range(src_subtask_indices.shape[1] - 1):
            end_offsets = np.random.randint(
                low=self.task_spec[i]["subtask_term_offset_range"][0],
                high=self.task_spec[i]["subtask_term_offset_range"][1] + 1,
                size=src_subtask_indices.shape[0]
            )
            src_subtask_indices[:, i, 1] = src_subtask_indices[:, i, 1] + end_offsets
            # don't forget to set these as start indices for next subtask too
            src_subtask_indices[:, i + 1, 0] = src_subtask_indices[:, i, 1]

        # ensure non-empty subtasks
        assert np.all((src_subtask_indices[:, :, 1] - src_subtask_indices[:, :, 0]) > 0), "got empty subtasks!"

        # ensure subtask indices increase (both starts and ends)
        assert np.all((src_subtask_indices[:, 1:, :] - src_subtask_indices[:, :-1, :]) > 0), "subtask indices do not strictly increase"

        # ensure subtasks are in order
        subtask_inds_flat = src_subtask_indices.reshape(src_subtask_indices.shape[0], -1)
        assert np.all((subtask_inds_flat[:, 1:] - subtask_inds_flat[:, :-1]) >= 0), "subtask indices not in order"

        return src_subtask_indices

    def select_source_demo(
        self,
        eef_pose,
        object_pose,
        subtask_ind,
        src_subtask_inds,
        subtask_object_name,
        selection_strategy_name,
        selection_strategy_kwargs=None,
    ):
        """
        Helper method to run source subtask segment selection.

        Args:
            eef_pose (np.array): current end effector pose
            object_pose (np.array): current object pose for this subtask
            subtask_ind (int): index of subtask
            src_subtask_inds (np.array): start and end indices for subtask segment in source demonstrations of shape (N, 2)
            subtask_object_name (str): name of reference object for this subtask
            selection_strategy_name (str): name of selection strategy
            selection_strategy_kwargs (dict): extra kwargs for running selection strategy

        Returns:
            selected_src_demo_ind (int): selected source demo index
        """
        if subtask_object_name is None:
            # no reference object - only random selection is supported
            assert selection_strategy_name == "random"

        # We need to collect the datagen info objects over the timesteps for the subtask segment in each source 
        # demo, so that it can be used by the selection strategy.
        src_subtask_datagen_infos = []
        for i in range(len(self.demo_keys)):
            # datagen info over all timesteps of the src trajectory
            src_ep_datagen_info = self.src_dataset_infos[i]

            # time indices for subtask
            subtask_start_ind = src_subtask_inds[i][0]
            subtask_end_ind = src_subtask_inds[i][1]

            # get subtask segment using indices
            src_subtask_datagen_infos.append(DatagenInfo(
                eef_pose=src_ep_datagen_info.eef_pose[subtask_start_ind : subtask_end_ind],
                # only include object pose for relevant object in subtask
                object_poses={ subtask_object_name : src_ep_datagen_info.object_poses[subtask_object_name][subtask_start_ind : subtask_end_ind] } if (subtask_object_name is not None) else None,
                # subtask termination signal is unused
                subtask_term_signals=None,
                target_pose=src_ep_datagen_info.target_pose[subtask_start_ind : subtask_end_ind],
                gripper_action=src_ep_datagen_info.gripper_action[subtask_start_ind : subtask_end_ind],
            ))

        # make selection strategy object
        selection_strategy_obj = make_selection_strategy(selection_strategy_name)

        # run selection
        if selection_strategy_kwargs is None:
            selection_strategy_kwargs = dict()
        selected_src_demo_ind = selection_strategy_obj.select_source_demo(
            eef_pose=eef_pose,
            object_pose=object_pose,
            src_subtask_datagen_infos=src_subtask_datagen_infos,
            **selection_strategy_kwargs,
        )

        return selected_src_demo_ind

    def generate(
        self,
        env,
        env_interface,
        select_src_per_subtask=False,
        transform_first_robot_pose=False,
        interpolate_from_last_target_pose=True,
        render=False,
        video_writer=None,
        video_skip=5,
        camera_names=None,
        pause_subtask=False,
    ):
        """
        Attempt to generate a new demonstration.

        Args:
            env (robomimic EnvBase instance): environment to use for data collection
            
            env_interface (MG_EnvInterface instance): environment interface for some data generation operations

            select_src_per_subtask (bool): if True, select a different source demonstration for each subtask 
                during data generation, else keep the same one for the entire episode

            transform_first_robot_pose (bool): if True, each subtask segment will consist of the first
                robot pose and the target poses instead of just the target poses. Can sometimes help
                improve data generation quality as the interpolation segment will interpolate to where 
                the robot started in the source segment instead of the first target pose. Note that the
                first subtask segment of each episode will always include the first robot pose, regardless
                of this argument.

            interpolate_from_last_target_pose (bool): if True, each interpolation segment will start from
                the last target pose in the previous subtask segment, instead of the current robot pose. Can
                sometimes improve data generation quality.

            render (bool): if True, render on-screen

            video_writer (imageio writer): video writer

            video_skip (int): determines rate at which environment frames are written to video

            camera_names (list): determines which camera(s) are used for rendering. Pass more than
                one to output a video with multiple camera views concatenated horizontally.

            pause_subtask (bool): if True, pause after every subtask during generation, for
                debugging.

        Returns:
            results (dict): dictionary with the following items:
                initial_state (dict): initial simulator state for the executed trajectory
                states (list): simulator state at each timestep
                observations (list): observation dictionary at each timestep
                datagen_infos (list): datagen_info at each timestep
                actions (np.array): action executed at each timestep
                success (bool): whether the trajectory successfully solved the task or not
                src_demo_inds (list): list of selected source demonstration indices for each subtask
                src_demo_labels (np.array): same as @src_demo_inds, but repeated to have a label for each timestep of the trajectory
        """

        # sample new task instance
        env.reset()
        new_initial_state = env.get_state()

        # sample new subtask boundaries
        all_subtask_inds = self.randomize_subtask_boundaries() # shape [N, S, 2], last dim is start and end action lengths

        # some state variables used during generation
        selected_src_demo_ind = None
        prev_executed_traj = None

        # save generated data in these variables
        generated_states = []
        generated_obs = []
        generated_datagen_infos = []
        generated_actions = []
        generated_success = False
        generated_src_demo_inds = [] # store selected src demo ind for each subtask in each trajectory
        generated_src_demo_labels = [] # like @generated_src_demo_inds, but padded to align with size of @generated_actions

        for subtask_ind in range(len(self.task_spec)):

            # some things only happen on first subtask
            is_first_subtask = (subtask_ind == 0)

            # get datagen info in current environment to get required info for selection (e.g. eef pose, object pose)
            cur_datagen_info = env_interface.get_datagen_info()

            # name of object for this subtask
            subtask_object_name = self.task_spec[subtask_ind]["object_ref"]

            # corresponding current object pose
            cur_object_pose = cur_datagen_info.object_poses[subtask_object_name] if (subtask_object_name is not None) else None

            # We need source demonstration selection for the first subtask (always), and possibly for 
            # other subtasks if @select_src_per_subtask is set.
            need_source_demo_selection = (is_first_subtask or select_src_per_subtask)

            # Run source demo selection or use selected demo from previous iteration
            if need_source_demo_selection:
                selected_src_demo_ind = self.select_source_demo(
                    eef_pose=cur_datagen_info.eef_pose,
                    object_pose=cur_object_pose,
                    subtask_ind=subtask_ind,
                    src_subtask_inds=all_subtask_inds[:, subtask_ind],
                    subtask_object_name=subtask_object_name,
                    selection_strategy_name=self.task_spec[subtask_ind]["selection_strategy"],
                    selection_strategy_kwargs=self.task_spec[subtask_ind]["selection_strategy_kwargs"],
                )
            assert (selected_src_demo_ind is not None)

            # selected subtask segment time indices
            selected_src_subtask_inds = all_subtask_inds[selected_src_demo_ind, subtask_ind]

            # get subtask segment, consisting of the sequence of robot eef poses, target poses, gripper actions
            src_ep_datagen_info = self.src_dataset_infos[selected_src_demo_ind]
            src_subtask_eef_poses = src_ep_datagen_info.eef_pose[selected_src_subtask_inds[0] : selected_src_subtask_inds[1]]
            src_subtask_target_poses = src_ep_datagen_info.target_pose[selected_src_subtask_inds[0] : selected_src_subtask_inds[1]]
            src_subtask_gripper_actions = src_ep_datagen_info.gripper_action[selected_src_subtask_inds[0] : selected_src_subtask_inds[1]]
            
            # get reference object pose from source demo
            src_subtask_object_pose = src_ep_datagen_info.object_poses[subtask_object_name][selected_src_subtask_inds[0]] if (subtask_object_name is not None) else None

            if is_first_subtask or transform_first_robot_pose:
                # Source segment consists of first robot eef pose and the target poses. This ensures that
                # we will interpolate to the first robot eef pose in this source segment, instead of the
                # first robot target pose.
                src_eef_poses = np.concatenate([src_subtask_eef_poses[0:1], src_subtask_target_poses], axis=0)
            else:
                # Source segment consists of just the target poses.
                src_eef_poses = np.array(src_subtask_target_poses)

            # account for extra timestep added to @src_eef_poses
            src_subtask_gripper_actions = np.concatenate([src_subtask_gripper_actions[0:1], src_subtask_gripper_actions], axis=0)

            # Transform source demonstration segment using relevant object pose.
            if subtask_object_name is not None:
                transformed_eef_poses = PoseUtils.transform_source_data_segment_using_object_pose(
                    obj_pose=cur_object_pose,
                    src_eef_poses=src_eef_poses,
                    src_obj_pose=src_subtask_object_pose,
                )
            else:
                # skip transformation if no reference object is provided
                transformed_eef_poses = src_eef_poses
            
            # We will construct a WaypointTrajectory instance to keep track of robot control targets 
            # that will be executed and then execute it.
            traj_to_execute = WaypointTrajectory()

            if interpolate_from_last_target_pose and (not is_first_subtask):
                # Interpolation segment will start from last target pose (which may not have been achieved).
                assert prev_executed_traj is not None
                last_waypoint = prev_executed_traj.last_waypoint
                init_sequence = WaypointSequence(sequence=[last_waypoint])
            else:
                # Interpolation segment will start from current robot eef pose.
                init_sequence = WaypointSequence.from_poses(
                    poses=cur_datagen_info.eef_pose[None], 
                    gripper_actions=src_subtask_gripper_actions[0:1],
                    action_noise=self.task_spec[subtask_ind]["action_noise"],
                )
            traj_to_execute.add_waypoint_sequence(init_sequence)

            # Construct trajectory for the transformed segment.
            transformed_seq = WaypointSequence.from_poses(
                poses=transformed_eef_poses, 
                gripper_actions=src_subtask_gripper_actions,
                action_noise=self.task_spec[subtask_ind]["action_noise"],
            )
            transformed_traj = WaypointTrajectory()
            transformed_traj.add_waypoint_sequence(transformed_seq)

            # Merge this trajectory into our trajectory using linear interpolation.
            # Interpolation will happen from the initial pose (@init_sequence) to the first element of @transformed_seq.
            traj_to_execute.merge(
                transformed_traj,
                num_steps_interp=self.task_spec[subtask_ind]["num_interpolation_steps"],
                num_steps_fixed=self.task_spec[subtask_ind]["num_fixed_steps"],
                action_noise=(float(self.task_spec[subtask_ind]["apply_noise_during_interpolation"]) * self.task_spec[subtask_ind]["action_noise"]),
            )

            # We initialized @traj_to_execute with a pose to allow @merge to handle linear interpolation
            # for us. However, we can safely discard that first waypoint now, and just start by executing
            # the rest of the trajectory (interpolation segment and transformed subtask segment).
            traj_to_execute.pop_first()

            # Execute the trajectory and collect data.
            exec_results = traj_to_execute.execute(
                env=env,
                env_interface=env_interface,
                render=render,
                video_writer=video_writer,
                video_skip=video_skip,
                camera_names=camera_names,
            )

            # check that trajectory is non-empty
            if len(exec_results["states"]) > 0:
                generated_states += exec_results["states"]
                generated_obs += exec_results["observations"]
                generated_datagen_infos += exec_results["datagen_infos"]
                generated_actions.append(exec_results["actions"])
                generated_success = generated_success or exec_results["success"]
                generated_src_demo_inds.append(selected_src_demo_ind)
                generated_src_demo_labels.append(selected_src_demo_ind * np.ones((exec_results["actions"].shape[0], 1), dtype=int))

            # remember last trajectory
            prev_executed_traj = traj_to_execute

            if pause_subtask:
                input("Pausing after subtask {} execution. Press any key to continue...".format(subtask_ind))

        # merge numpy arrays
        if len(generated_actions) > 0:
            generated_actions = np.concatenate(generated_actions, axis=0)
            generated_src_demo_labels = np.concatenate(generated_src_demo_labels, axis=0)

        results = dict(
            initial_state=new_initial_state,
            states=generated_states,
            observations=generated_obs,
            datagen_infos=generated_datagen_infos,
            actions=generated_actions,
            success=generated_success,
            src_demo_inds=generated_src_demo_inds,
            src_demo_labels=generated_src_demo_labels,
        )
        return results
