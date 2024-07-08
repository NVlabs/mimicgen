# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Base class for environment interfaces used by MimicGen. Defines a set of
functions that should be implemented for every set of environments, and
a global registry.
"""
import abc # for abstract base class definitions
import six # preserve metaclass compatibility between python 2 and 3

import numpy as np

from mimicgen.datagen.datagen_info import DatagenInfo


# Global dictionary for remembering name - class mappings.
# 
# Organization:
#   interface_type (str)
#       class_name (str)
#           class object
REGISTERED_ENV_INTERFACES = {}


def make_interface(name, interface_type, *args, **kwargs):
    """
    Creates an instance of a env interface. Make sure to pass any other needed arguments.
    """
    if interface_type not in REGISTERED_ENV_INTERFACES:
        raise Exception("make_interface: interface type {} not found. Make sure it is a registered interface type among: {}".format(interface_type, ", ".join(REGISTERED_ENV_INTERFACES)))
    if name not in REGISTERED_ENV_INTERFACES[interface_type]:
        raise Exception("make_interface: interface name {} not found. Make sure it is a registered interface name among: {}".format(name, ', '.join(REGISTERED_ENV_INTERFACES[interface_type])))
    return REGISTERED_ENV_INTERFACES[interface_type][name](*args, **kwargs)


def register_env_interface(cls):
    """
    Register environment interface class into global registry.
    """
    ignore_classes = ["MG_EnvInterface"]
    if cls.__name__ not in ignore_classes:
        if cls.INTERFACE_TYPE not in REGISTERED_ENV_INTERFACES:
            REGISTERED_ENV_INTERFACES[cls.INTERFACE_TYPE] = dict()
        REGISTERED_ENV_INTERFACES[cls.INTERFACE_TYPE][cls.__name__] = cls


class MG_EnvInterfaceMeta(type):
    """
    This metaclass adds env interface classes into the global registry.
    """
    def __new__(meta, name, bases, class_dict):
        cls = super(MG_EnvInterfaceMeta, meta).__new__(meta, name, bases, class_dict)
        register_env_interface(cls)
        return cls


@six.add_metaclass(MG_EnvInterfaceMeta)
class MG_EnvInterface(object):
    """
    Environment interface API that MimicGen environment interfaces should conform to.
    """
    def __init__(self, env):
        """
        Args:
            env: environment object
        """
        self.env = env
        self.interface_type = type(self).INTERFACE_TYPE

    def __repr__(self):
        """
        Pretty-print env description.
        """
        return self.__class__.__name__

    """
    These should be filled out by simulator subclasses (e.g. robosuite).
    """
    @property
    @classmethod
    def INTERFACE_TYPE(self):
        """
        Returns string corresponding to interface type. This is used to group
        all subclasses together in the interface registry (for example, all robosuite
        interfaces) and helps avoid name conflicts.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_robot_eef_pose(self):
        """
        Get current robot end effector pose. Should be the same frame as used by the robot end-effector controller.

        Returns:
            pose (np.array): 4x4 eef pose matrix
        """
        raise NotImplementedError

    @abc.abstractmethod
    def target_pose_to_action(self, target_pose, relative=True):
        """
        Takes a target pose for the end effector controller and returns an action 
        (usually a normalized delta pose action) to try and achieve that target pose. 

        Args:
            target_pose (np.array): 4x4 target eef pose
            relative (bool): if True, use relative pose actions, else absolute pose actions

        Returns:
            action (np.array): action compatible with env.step (minus gripper actuation)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def action_to_target_pose(self, action, relative=True):
        """
        Converts action (compatible with env.step) to a target pose for the end effector controller.
        Inverse of @target_pose_to_action. Usually used to infer a sequence of target controller poses
        from a demonstration trajectory using the recorded actions.

        Args:
            action (np.array): environment action
            relative (bool): if True, use relative pose actions, else absolute pose actions

        Returns:
            target_pose (np.array): 4x4 target eef pose that @action corresponds to
        """
        raise NotImplementedError

    @abc.abstractmethod
    def action_to_gripper_action(self, action):
        """
        Extracts the gripper actuation part of an action (compatible with env.step).

        Args:
            action (np.array): environment action

        Returns:
            gripper_action (np.array): subset of environment action for gripper actuation
        """
        raise NotImplementedError

    """
    These should be filled out by each simulation domain (e.g. nut assembly, coffee).
    """
    @abc.abstractmethod
    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_subtask_term_signals(self):
        """
        Gets a dictionary of binary flags for each subtask in a task. The flag is 1
        when the subtask has been completed and 0 otherwise. MimicGen only uses this
        when parsing source demonstrations at the start of data generation, and it only
        uses the first 0 -> 1 transition in this signal to detect the end of a subtask.

        Returns:
            subtask_term_signals (dict): dictionary that maps subtask name to termination flag (0 or 1)
        """
        raise NotImplementedError

    """
    This method can be left as-is in most cases, as it calls other implemented methods to provide a 
    DatagenInfo object.
    """
    def get_datagen_info(self, action=None):
        """
        Get information needed for data generation, at the current
        timestep of simulation. If @action is provided, it will be used to 
        compute the target eef pose for the controller, otherwise that 
        will be excluded.

        Returns:
            datagen_info (DatagenInfo instance)
        """

        # current eef pose
        eef_pose = self.get_robot_eef_pose()

        # object poses
        object_poses = self.get_object_poses()

        # subtask termination signals
        subtask_term_signals = self.get_subtask_term_signals()

        # these must be extracted from provided action
        target_pose = None
        gripper_action = None
        if action is not None:
            target_pose = self.action_to_target_pose(action=action, relative=True)
            gripper_action = self.action_to_gripper_action(action=action)

        datagen_info = DatagenInfo(
            eef_pose=eef_pose,
            object_poses=object_poses,
            subtask_term_signals=subtask_term_signals,
            target_pose=target_pose,
            gripper_action=gripper_action,
        )
        return datagen_info
