# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Defines structure of information that is needed from an environment for data generation.
"""
import numpy as np
from copy import deepcopy


class DatagenInfo(object):
    """
    Structure of information needed from an environment for data generation. To allow for
    flexibility, not all information must be present.
    """
    def __init__(
        self,
        eef_pose=None,
        object_poses=None,
        subtask_term_signals=None,
        target_pose=None,
        gripper_action=None,
    ):
        """
        Args:
            eef_pose (np.array or None): robot end effector poses of shape [..., 4, 4]
            object_poses (dict or None): dictionary mapping object name to object poses
                of shape [..., 4, 4]
            subtask_term_signals (dict or None): dictionary mapping subtask name to a binary 
                indicator (0 or 1) on whether subtask has been completed. Each value in the
                dictionary could be an int, float, or np.array of shape [..., 1].
            target_pose (np.array or None): target end effector poses of shape [..., 4, 4]
            gripper_action (np.array or None): gripper actions of shape [..., D] where D
                is the dimension of the gripper actuation action for the robot arm
        """
        self.eef_pose = None
        if eef_pose is not None:
            self.eef_pose = np.array(eef_pose)

        self.object_poses = None
        if object_poses is not None:
            self.object_poses = { k : np.array(object_poses[k]) for k in object_poses }

        self.subtask_term_signals = None
        if subtask_term_signals is not None:
            self.subtask_term_signals = dict()
            for k in subtask_term_signals:
                if isinstance(subtask_term_signals[k], float) or isinstance(subtask_term_signals[k], int):
                    self.subtask_term_signals[k] = subtask_term_signals[k]
                else:
                    # only create numpy array if value is not a single value
                    self.subtask_term_signals[k] = np.array(subtask_term_signals[k])

        self.target_pose = None
        if target_pose is not None:
            self.target_pose = np.array(target_pose)

        self.gripper_action = None
        if gripper_action is not None:
            self.gripper_action = np.array(gripper_action)

    def to_dict(self):
        """
        Convert this instance to a dictionary containing the same information.
        """
        ret = dict()
        if self.eef_pose is not None:
            ret["eef_pose"] = np.array(self.eef_pose)
        if self.object_poses is not None:
            ret["object_poses"] = deepcopy(self.object_poses)
        if self.subtask_term_signals is not None:
            ret["subtask_term_signals"] = deepcopy(self.subtask_term_signals)
        if self.target_pose is not None:
            ret["target_pose"] = np.array(self.target_pose)
        if self.gripper_action is not None:
            ret["gripper_action"] = np.array(self.gripper_action)
        return ret
