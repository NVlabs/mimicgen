# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Selection strategies used by MimicGen to select subtask segments from 
source human demonstrations.
"""
import abc # for abstract base class definitions
import six # preserve metaclass compatibility between python 2 and 3

import numpy as np

import mimicgen.utils.pose_utils as PoseUtils


# Global dictionary for remembering name to class mappings.
REGISTERED_SELECTION_STRATEGIES = {}


def make_selection_strategy(name, *args, **kwargs):
    """
    Creates an instance of a selection strategy class, specified by @name, 
    which is used to look it up in the registry.
    """
    assert_selection_strategy_exists(name)
    return REGISTERED_SELECTION_STRATEGIES[name](*args, **kwargs)


def register_selection_strategy(cls):
    """
    Register selection strategy class into global registry.
    """
    ignore_classes = ["MG_SelectionStrategy"]
    if cls.__name__ not in ignore_classes:
        REGISTERED_SELECTION_STRATEGIES[cls.NAME] = cls


def assert_selection_strategy_exists(name):
    """
    Allow easy way to check if selection strategy exists.
    """
    if name not in REGISTERED_SELECTION_STRATEGIES:
        raise Exception("assert_selection_strategy_exists: name {} not found. Make sure it is a registered selection strategy among {}".format(", ".join(REGISTERED_SELECTION_STRATEGIES)))


class MG_SelectionStrategyMeta(type):
    """
    This metaclass adds selection strategy classes into the global registry.
    """
    def __new__(meta, name, bases, class_dict):
        cls = super(MG_SelectionStrategyMeta, meta).__new__(meta, name, bases, class_dict)
        register_selection_strategy(cls)
        return cls


@six.add_metaclass(MG_SelectionStrategyMeta)
class MG_SelectionStrategy(object):
    """
    Defines methods and functions for selection strategies to implement.
    """
    def __init__(self):
        pass

    @property
    @classmethod
    def NAME(self):
        """
        This name (str) will be used to register the selection strategy class in the global
        registry.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def select_source_demo(
        self,
        eef_pose,
        object_pose,
        src_subtask_datagen_infos,
    ):
        """
        Selects source demonstration index using the current robot pose, relevant object pose
        for the current subtask, and relevant information from the source demonstrations for the
        current subtask.

        Args:
            eef_pose (np.array): current 4x4 eef pose
            object_pose (np.array): current 4x4 object pose, for the object in this subtask
            src_subtask_datagen_infos (list): DatagenInfo instance for the relevant subtask segment
                in the source demonstrations

        Returns:
            source_demo_ind (int): index of source demonstration - indicates which source subtask segment to use
        """
        raise NotImplementedError


class RandomStrategy(MG_SelectionStrategy):
    """
    Pick source demonstration randomly.
    """

    # name for registering this class into registry
    NAME = "random"

    def select_source_demo(
        self,
        eef_pose,
        object_pose,
        src_subtask_datagen_infos,
    ):
        """
        Selects source demonstration index using the current robot pose, relevant object pose
        for the current subtask, and relevant information from the source demonstrations for the
        current subtask.

        Args:
            eef_pose (np.array): current 4x4 eef pose
            object_pose (np.array): current 4x4 object pose, for the object in this subtask
            src_subtask_datagen_infos (list): DatagenInfo instance for the relevant subtask segment
                in the source demonstrations

        Returns:
            source_demo_ind (int): index of source demonstration - indicates which source subtask segment to use
        """

        # random selection
        n_src_demo = len(src_subtask_datagen_infos)
        return np.random.randint(0, n_src_demo)


class NearestNeighborObjectStrategy(MG_SelectionStrategy):
    """
    Pick source demonstration to be the one with the closest object pose to the object 
    in the current scene.
    """

    # name for registering this class into registry
    NAME = "nearest_neighbor_object"

    def select_source_demo(
        self,
        eef_pose,
        object_pose,
        src_subtask_datagen_infos,
        pos_weight=1.,
        rot_weight=1.,
        nn_k=3,
    ):
        """
        Selects source demonstration index using the current robot pose, relevant object pose
        for the current subtask, and relevant information from the source demonstrations for the
        current subtask.

        Args:
            eef_pose (np.array): current 4x4 eef pose
            object_pose (np.array): current 4x4 object pose, for the object in this subtask
            src_subtask_datagen_infos (list): DatagenInfo instance for the relevant subtask segment
                in the source demonstrations
            pos_weight (float): weight on position for minimizing pose distance
            rot_weight (float): weight on rotation for minimizing pose distance
            nn_k (int): pick source demo index uniformly at randomly from the top @nn_k nearest neighbors

        Returns:
            source_demo_ind (int): index of source demonstration - indicates which source subtask segment to use
        """

        # collect object poses from start of subtask source segments into array of shape [N, 4, 4]
        src_object_poses = []
        for di in src_subtask_datagen_infos:
            src_obj_pose = list(di.object_poses.values())
            assert len(src_obj_pose) == 1
            # use object pose at start of subtask segment
            src_object_poses.append(src_obj_pose[0][0])
        src_object_poses = np.array(src_object_poses)

        # split into positions and rotations
        all_src_obj_pos, all_src_obj_rot = PoseUtils.unmake_pose(src_object_poses)
        obj_pos, obj_rot = PoseUtils.unmake_pose(object_pose)

        # prepare for broadcasting
        obj_pos = obj_pos.reshape(-1, 3)
        obj_rot_T = obj_rot.T.reshape(-1, 3, 3)

        # pos dist is just L2 between positions
        pos_dists = np.sqrt(((all_src_obj_pos - obj_pos) ** 2).sum(axis=-1))

        # get angle (in axis-angle representation of delta rotation matrix) using the following formula
        # (see http://www.boris-belousov.net/2016/12/01/quat-dist/)

        # batched matrix mult, [N, 3, 3] x [1, 3, 3] -> [N, 3, 3]
        delta_R = np.matmul(all_src_obj_rot, obj_rot_T)
        arc_cos_in = (np.trace(delta_R, axis1=-2, axis2=-1) - 1.) / 2.
        arc_cos_in = np.clip(arc_cos_in, -1., 1.) # clip for numerical stability
        rot_dists = np.arccos(arc_cos_in)

        # weight distances with coefficients
        dists_to_minimize = pos_weight * pos_dists + rot_weight * rot_dists

        # clip top-k parameter to max possible value
        nn_k = min(nn_k, len(dists_to_minimize))

        # return one of the top-K nearest neighbors uniformly at random
        rand_k = np.random.randint(0, nn_k)
        top_k_neighbors_in_order = np.argsort(dists_to_minimize)[:nn_k]
        return top_k_neighbors_in_order[rand_k]


class NearestNeighborRobotDistanceStrategy(MG_SelectionStrategy):
    """
    Pick source demonstration to be the one that minimizes the distance the robot 
    end effector will need to travel from the current pose to the first pose 
    in the transformed segment.
    """

    # name for registering this class into registry
    NAME = "nearest_neighbor_robot_distance"

    def select_source_demo(
        self,
        eef_pose,
        object_pose,
        src_subtask_datagen_infos,
        pos_weight=1.,
        rot_weight=1.,
        nn_k=3,
    ):
        """
        Selects source demonstration index using the current robot pose, relevant object pose
        for the current subtask, and relevant information from the source demonstrations for the
        current subtask.

        Args:
            eef_pose (np.array): current 4x4 eef pose
            object_pose (np.array): current 4x4 object pose, for the object in this subtask
            src_subtask_datagen_infos (list): DatagenInfo instance for the relevant subtask segment
                in the source demonstrations
            pos_weight (float): weight on position for minimizing pose distance
            rot_weight (float): weight on rotation for minimizing pose distance
            nn_k (int): pick source demo index uniformly at randomly from the top @nn_k nearest neighbors

        Returns:
            source_demo_ind (int): index of source demonstration - indicates which source subtask segment to use
        """

        # collect eef and object poses from start of subtask source segments into arrays of shape [N, 4, 4]
        src_eef_poses = []
        src_object_poses = []
        for di in src_subtask_datagen_infos:
            # use eef pose at start of subtask segment
            src_eef_poses.append(di.eef_pose[0])
            # use object pose at start of subtask segment
            src_obj_pose = list(di.object_poses.values())
            assert len(src_obj_pose) == 1
            src_object_poses.append(src_obj_pose[0][0])
        src_eef_poses = np.array(src_eef_poses)
        src_object_poses = np.array(src_object_poses)

        # Get source eef poses with respect to object frames.
        # note: frame A is world, frame B is object
        src_object_poses_inv = PoseUtils.pose_inv(src_object_poses)
        src_eef_poses_in_obj = PoseUtils.pose_in_A_to_pose_in_B(
            pose_in_A=src_eef_poses,
            pose_A_in_B=src_object_poses_inv,
        )

        # Use this to find the first pose for the transformed subtask segment for each source demo.
        # Note this is the same logic used in PoseUtils.transform_source_data_segment_using_object_pose
        transformed_eef_poses = PoseUtils.pose_in_A_to_pose_in_B(
            pose_in_A=src_eef_poses_in_obj,
            pose_A_in_B=object_pose,
        )

        # split into positions and rotations
        all_transformed_eef_pos, all_transformed_eef_rot = PoseUtils.unmake_pose(transformed_eef_poses)
        eef_pos, eef_rot = PoseUtils.unmake_pose(eef_pose)

        # now measure distance from each of these transformed eef poses to our current eef pose
        # and choose the source demo that minimizes this distance

        # prepare for broadcasting
        eef_pos = eef_pos.reshape(-1, 3)
        eef_rot_T = eef_rot.T.reshape(-1, 3, 3)

        # pos dist is just L2 between positions
        pos_dists = np.sqrt(((all_transformed_eef_pos - eef_pos) ** 2).sum(axis=-1))

        # get angle (in axis-angle representation of delta rotation matrix) using the following formula
        # (see http://www.boris-belousov.net/2016/12/01/quat-dist/)

        # batched matrix mult, [N, 3, 3] x [1, 3, 3] -> [N, 3, 3]
        delta_R = np.matmul(all_transformed_eef_rot, eef_rot_T)
        arc_cos_in = (np.trace(delta_R, axis1=-2, axis2=-1) - 1.) / 2.
        arc_cos_in = np.clip(arc_cos_in, -1., 1.) # clip for numerical stability
        rot_dists = np.arccos(arc_cos_in)

        # weight distances with coefficients
        dists_to_minimize = pos_weight * pos_dists + rot_weight * rot_dists

        # clip top-k parameter to max possible value
        nn_k = min(nn_k, len(dists_to_minimize))

        # return one of the top-K nearest neighbors uniformly at random
        rand_k = np.random.randint(0, nn_k)
        top_k_neighbors_in_order = np.argsort(dists_to_minimize)[:nn_k]
        return top_k_neighbors_in_order[rand_k]


class NearestNeighborInterpolationStrategy(MG_SelectionStrategy):
    """
    Pick source demonstration to be the one with the closest object pose to the object 
    in the current scene.
    """

    # name for registering this class into registry
    NAME = "nearest_neighbor_interpolation"

    def select_source_demo(
        self,
        eef_pose,
        object_pose,
        src_subtask_datagen_infos,
        pos_weight=1.,
        rot_weight=1.,
        nn_k=3,
    ):
        """
        Another version of nearest neighbor, where we compute the initial eef pose of the transformed
        trajectory for all source demonstrations, and choose the one that's closest to the current
        eef pose.

        Args:
            eef_pose (np.array): current 4x4 eef pose
            object_pose (np.array): current 4x4 object pose, for the object in this subtask
            src_subtask_datagen_infos (list): DatagenInfo instance for the relevant subtask segment
                in the source demonstrations
            pos_weight (float): weight on position for minimizing pose distance
            rot_weight (float): weight on rotation for minimizing pose distance
            nn_k (int): pick source demo index uniformly at randomly from the top @nn_k nearest neighbors

        Returns:
            source_demo_ind (int): index of source demonstration - indicates which source subtask segment to use
        """

        # collect eef and object poses from start of subtask source segments into arrays of shape [N, 4, 4]
        src_eef_poses = []
        src_object_poses = []
        for di in src_subtask_datagen_infos:
            # use eef pose at start of subtask segment
            src_eef_poses.append(di.eef_pose[0])
            # use object pose at start of subtask segment
            src_obj_pose = list(di.object_poses.values())
            assert len(src_obj_pose) == 1
            src_object_poses.append(src_obj_pose[0][0])
        src_eef_poses = np.array(src_eef_poses)
        src_object_poses = np.array(src_object_poses)

        # initial obj poses from source demos
        all_src_world_pose_in_obj = PoseUtils.pose_inv(src_object_poses)

        # initial eef poses from source demos
        all_src_eef_pose_in_world = src_eef_poses

        # initial eef poses with respect to initial object poses from source demos
        all_src_eef_pose_in_obj = PoseUtils.pose_in_A_to_pose_in_B(pose_in_A=all_src_eef_pose_in_world, pose_A_in_B=all_src_world_pose_in_obj)

        # use new object pose to back out the desired target eef pose if we were to transform each source demo
        new_init_obj_pose_in_world = object_pose #PoseUtils.make_pose(obj_pos, obj_rot)
        new_init_eef_pose = PoseUtils.pose_in_A_to_pose_in_B(pose_in_A=all_src_eef_pose_in_obj, pose_A_in_B=new_init_obj_pose_in_world)
        new_init_eef_pos, new_init_eef_rot = PoseUtils.unmake_pose(new_init_eef_pose)

        # now measure distance from each of these target eef poses to our current eef pose (for new task instance)
        # and choose the source demo that minimizes this distance

        # prepare for broadcasting
        eef_pos, eef_rot = PoseUtils.unmake_pose(eef_pose)
        eef_pos = eef_pos.reshape(-1, 3)
        eef_rot_T = eef_rot.T.reshape(-1, 3, 3)

        # pos dist is just L2
        pos_dists = np.sqrt(((new_init_eef_pos - eef_pos) ** 2).sum(axis=-1))

        # get angle (in axis-angle representation of delta rotation matrix) using the following formula
        # (see http://www.boris-belousov.net/2016/12/01/quat-dist/)

        # batched matrix mult, [N, 3, 3] x [1, 3, 3] -> [N, 3, 3]
        delta_R = np.matmul(new_init_eef_rot, eef_rot_T)
        arc_cos_in = (np.trace(delta_R, axis1=-2, axis2=-1) - 1.) / 2.
        arc_cos_in = np.clip(arc_cos_in, -1., 1.) # clip for numerical stability
        rot_dists = np.arccos(arc_cos_in)

        # weight distances with coefficients
        dists_to_minimize = pos_weight * pos_dists + rot_weight * rot_dists

        if nn_k is None:
            # nearest neighbor
            return np.argmin(dists_to_minimize)

        # clip this to max value possible
        nn_k = min(nn_k, len(dists_to_minimize))

        # return one of the top-K nearest neighbors uniformly at random
        rand_k = np.random.randint(0, nn_k)
        top_k_neighbors_in_order = np.argsort(dists_to_minimize)[:nn_k]
        return top_k_neighbors_in_order[rand_k]
