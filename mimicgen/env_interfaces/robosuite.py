# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
MimicGen environment interface classes for basic robosuite environments.
"""
import numpy as np

import robosuite
import robosuite.utils.transform_utils as T

import mimicgen.utils.pose_utils as PoseUtils
from mimicgen.env_interfaces.base import MG_EnvInterface


class RobosuiteInterface(MG_EnvInterface):
    """
    MimicGen environment interface base class for basic robosuite environments.
    """

    # Note: base simulator interface class must fill out interface type as a class property
    INTERFACE_TYPE = "robosuite"

    def get_robot_eef_pose(self):
        """
        Get current robot end effector pose. Should be the same frame as used by the robot end-effector controller.

        Returns:
            pose (np.array): 4x4 eef pose matrix
        """

        # OSC control frame is a MuJoCo site - just retrieve its current pose
        return self.get_object_pose(
            obj_name=self.env.robots[0].controller.eef_name, 
            obj_type="site",
        )

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

        # version check for robosuite - must be v1.2+, so that we're using the correct controller convention
        assert (robosuite.__version__.split(".")[0] == "1")
        assert (robosuite.__version__.split(".")[1] >= "2")

        # target position and rotation
        target_pos, target_rot = PoseUtils.unmake_pose(target_pose)

        # current position and rotation
        curr_pose = self.get_robot_eef_pose()
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        # get maximum position and rotation action bounds
        max_dpos = self.env.robots[0].controller.output_max[0]
        max_drot = self.env.robots[0].controller.output_max[3]

        if relative:
            # normalized delta position action
            delta_position = target_pos - curr_pos
            delta_position = np.clip(delta_position / max_dpos, -1., 1.)

            # normalized delta rotation action
            delta_rot_mat = target_rot.dot(curr_rot.T)
            delta_quat = T.mat2quat(delta_rot_mat)
            delta_rotation = T.quat2axisangle(delta_quat)
            delta_rotation = np.clip(delta_rotation / max_drot, -1., 1.)
            return np.concatenate([delta_position, delta_rotation])

        # absolute position and rotation action
        target_quat = T.mat2quat(target_rot)
        abs_rotation = T.quat2axisangle(target_quat)
        return np.concatenate([target_pos, abs_rotation])

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

        # version check for robosuite - must be v1.2+, so that we're using the correct controller convention
        assert (robosuite.__version__.split(".")[0] == "1")
        assert (robosuite.__version__.split(".")[1] >= "2")

        if (not relative):
            # convert absolute action to absolute pose
            target_pos = action[:3]
            target_quat = T.axisangle2quat(action[3:6])
            target_rot = T.quat2mat(target_quat)
        else:
            # get maximum position and rotation action bounds
            max_dpos = self.env.robots[0].controller.output_max[0]
            max_drot = self.env.robots[0].controller.output_max[3]

            # unscale actions
            delta_position = action[:3] * max_dpos
            delta_rotation = action[3:6] * max_drot

            # current position and rotation
            curr_pose = self.get_robot_eef_pose()
            curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

            # get pose target
            target_pos = curr_pos + delta_position
            delta_quat = T.axisangle2quat(delta_rotation)
            delta_rot_mat = T.quat2mat(delta_quat)
            target_rot = delta_rot_mat.dot(curr_rot)

        target_pose = PoseUtils.make_pose(target_pos, target_rot)
        return target_pose

    def action_to_gripper_action(self, action):
        """
        Extracts the gripper actuation part of an action (compatible with env.step).

        Args:
            action (np.array): environment action

        Returns:
            gripper_action (np.array): subset of environment action for gripper actuation
        """

        # last dimension is gripper action
        return action[-1:]

    # robosuite-specific helper method for getting object poses
    def get_object_pose(self, obj_name, obj_type):
        """
        Returns 4x4 object pose given the name of the object and the type.

        Args:
            obj_name (str): name of object
            obj_type (str): type of object - either "body", "geom", or "site"

        Returns:
            obj_pose (np.array): 4x4 object pose
        """
        assert obj_type in ["body", "geom", "site"]

        if obj_type == "body":
            obj_id = self.env.sim.model.body_name2id(obj_name)
            obj_pos = np.array(self.env.sim.data.body_xpos[obj_id])
            obj_rot = np.array(self.env.sim.data.body_xmat[obj_id].reshape(3, 3))
        elif obj_type == "geom":
            obj_id = self.env.sim.model.geom_name2id(obj_name)
            obj_pos = np.array(self.env.sim.data.geom_xpos[obj_id])
            obj_rot = np.array(self.env.sim.data.geom_xmat[obj_id].reshape(3, 3))
        elif obj_type == "site":
            obj_id = self.env.sim.model.site_name2id(obj_name)
            obj_pos = np.array(self.env.sim.data.site_xpos[obj_id])
            obj_rot = np.array(self.env.sim.data.site_xmat[obj_id].reshape(3, 3))

        return PoseUtils.make_pose(obj_pos, obj_rot)


class MG_Coffee(RobosuiteInterface):
    """
    Corresponds to robosuite Coffee task and variants.
    """
    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """

        # two relevant objects - coffee pod and coffee machine
        return dict(
            coffee_pod=self.get_object_pose(obj_name=self.env.coffee_pod.root_body, obj_type="body"),
            coffee_machine=self.get_object_pose(obj_name=self.env.coffee_machine.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        """
        Gets a dictionary of binary flags for each subtask in a task. The flag is 1
        when the subtask has been completed and 0 otherwise. MimicGen only uses this
        when parsing source demonstrations at the start of data generation, and it only
        uses the first 0 -> 1 transition in this signal to detect the end of a subtask.

        Returns:
            subtask_term_signals (dict): dictionary that maps subtask name to termination flag (0 or 1)
        """
        signals = dict()

        metrics = self.env._get_partial_task_metrics()

        # first subtask is grasping coffee pod (motion relative to pod)
        signals["grasp"] = int(metrics["grasp"])

        # final subtask is inserting pod into machine and closing the lid (motion relative to machine) - but final subtask signal is not needed
        return signals


class MG_Threading(RobosuiteInterface):
    """
    Corresponds to robosuite Threading task and variants.
    """
    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """

        # two relevant objects - needle and tripod
        return dict(
            needle=self.get_object_pose(obj_name=self.env.needle.root_body, obj_type="body"),
            tripod=self.get_object_pose(obj_name=self.env.tripod.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        """
        Gets a dictionary of binary flags for each subtask in a task. The flag is 1
        when the subtask has been completed and 0 otherwise. MimicGen only uses this
        when parsing source demonstrations at the start of data generation, and it only
        uses the first 0 -> 1 transition in this signal to detect the end of a subtask.

        Returns:
            subtask_term_signals (dict): dictionary that maps subtask name to termination flag (0 or 1)
        """
        signals = dict()

        # first subtask is grasping needle (motion relative to needle)
        signals["grasp"] = int(self.env._check_grasp(
            gripper=self.env.robots[0].gripper,
            object_geoms=[g for g in self.env.needle.contact_geoms])
        )

        # final subtask is inserting needle into tripod (motion relative to tripod) - but final subtask signal is not needed
        return signals


class MG_ThreePieceAssembly(RobosuiteInterface):
    """
    Corresponds to robosuite ThreePieceAssembly task and variants.
    """
    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """

        # three relevant objects - base piece, piece_1, piece_2
        return dict(
            base=self.get_object_pose(obj_name=self.env.base.root_body, obj_type="body"),
            piece_1=self.get_object_pose(obj_name=self.env.piece_1.root_body, obj_type="body"),
            piece_2=self.get_object_pose(obj_name=self.env.piece_2.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        """
        Gets a dictionary of binary flags for each subtask in a task. The flag is 1
        when the subtask has been completed and 0 otherwise. MimicGen only uses this
        when parsing source demonstrations at the start of data generation, and it only
        uses the first 0 -> 1 transition in this signal to detect the end of a subtask.

        Returns:
            subtask_term_signals (dict): dictionary that maps subtask name to termination flag (0 or 1)
        """
        signals = dict()

        metrics = self.env._get_partial_task_metrics()

        # first subtask is grasping piece_1 (motion relative to piece_1)
        signals["grasp_1"] = int(self.env._check_grasp(
            gripper=self.env.robots[0].gripper,
            object_geoms=[g for g in self.env.piece_1.contact_geoms])
        )

        # second subtask is inserting piece_1 into the base (motion relative to base)
        signals["insert_1"] = int(metrics["first_piece_assembled"])

        # third subtask is grasping piece_2 (motion relative to piece_2)
        signals["grasp_2"] = int(self.env._check_grasp(
            gripper=self.env.robots[0].gripper,
            object_geoms=[g for g in self.env.piece_2.contact_geoms])
        )

        # final subtask is inserting piece_2 into piece_1 (motion relative to piece_1) - but final subtask signal is not needed
        return signals


class MG_Square(RobosuiteInterface):
    """
    Corresponds to robosuite Square task and variants.
    """
    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """

        # two relevant objects - square nut and peg
        return dict(
            square_nut=self.get_object_pose(obj_name=self.env.nuts[self.env.nut_to_id["square"]].root_body, obj_type="body"),
            square_peg=self.get_object_pose(obj_name="peg1", obj_type="body"),
        )

    def get_subtask_term_signals(self):
        """
        Gets a dictionary of binary flags for each subtask in a task. The flag is 1
        when the subtask has been completed and 0 otherwise. MimicGen only uses this
        when parsing source demonstrations at the start of data generation, and it only
        uses the first 0 -> 1 transition in this signal to detect the end of a subtask.

        Returns:
            subtask_term_signals (dict): dictionary that maps subtask name to termination flag (0 or 1)
        """
        signals = dict()

        # first subtask is grasping square nut (motion relative to square nut)
        signals["grasp"] = int(self.env._check_grasp(
            gripper=self.env.robots[0].gripper,
            object_geoms=[g for g in self.env.nuts[self.env.nut_to_id["square"]].contact_geoms])
        )

        # final subtask is inserting square nut onto square peg (motion relative to square peg) - but final subtask signal is not needed
        return signals


class MG_Stack(RobosuiteInterface):
    """
    Corresponds to robosuite Stack task and variants.
    """
    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """

        # two relevant objects - cubeA and cubeB
        return dict(
            cubeA=self.get_object_pose(obj_name=self.env.cubeA.root_body, obj_type="body"),
            cubeB=self.get_object_pose(obj_name=self.env.cubeB.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        """
        Gets a dictionary of binary flags for each subtask in a task. The flag is 1
        when the subtask has been completed and 0 otherwise. MimicGen only uses this
        when parsing source demonstrations at the start of data generation, and it only
        uses the first 0 -> 1 transition in this signal to detect the end of a subtask.

        Returns:
            subtask_term_signals (dict): dictionary that maps subtask name to termination flag (0 or 1)
        """
        signals = dict()

        # first subtask is grasping cubeA (motion relative to cubeA)
        signals["grasp"] = int(self.env._check_grasp(gripper=self.env.robots[0].gripper, object_geoms=self.env.cubeA))

        # final subtask is placing cubeA on cubeB (motion relative to cubeB) - but final subtask signal is not needed
        return signals


class MG_StackThree(RobosuiteInterface):
    """
    Corresponds to robosuite StackThree task and variants.
    """
    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """

        # three relevant objects - three cubes
        return dict(
            cubeA=self.get_object_pose(obj_name=self.env.cubeA.root_body, obj_type="body"),
            cubeB=self.get_object_pose(obj_name=self.env.cubeB.root_body, obj_type="body"),
            cubeC=self.get_object_pose(obj_name=self.env.cubeC.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        """
        Gets a dictionary of binary flags for each subtask in a task. The flag is 1
        when the subtask has been completed and 0 otherwise. MimicGen only uses this
        when parsing source demonstrations at the start of data generation, and it only
        uses the first 0 -> 1 transition in this signal to detect the end of a subtask.

        Returns:
            subtask_term_signals (dict): dictionary that maps subtask name to termination flag (0 or 1)
        """
        signals = dict()

        # first subtask is grasping cubeA (motion relative to cubeA)
        signals["grasp_1"] = int(self.env._check_grasp(gripper=self.env.robots[0].gripper, object_geoms=self.env.cubeA))

        # second subtask is placing cubeA on cubeB (motion relative to cubeB)
        signals["stack_1"] = int(self.env._check_cubeA_stacked())

        # third subtask is grasping cubeC (motion relative to cubeC)
        signals["grasp_2"] = int(self.env._check_grasp(gripper=self.env.robots[0].gripper, object_geoms=self.env.cubeC))

        # final subtask is placing cubeC on cubeA (motion relative to cubeA) - but final subtask signal is not needed
        return signals


class MG_HammerCleanup(RobosuiteInterface):
    """
    Corresponds to robosuite HammerCleanup task and variants.
    """
    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """

        # two relevant objects - hammer and drawer
        return dict(
            hammer=self.get_object_pose(obj_name=self.env.sorting_object.root_body, obj_type="body"),
            drawer=self.get_object_pose(obj_name=self.env.cabinet_object.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        """
        Gets a dictionary of binary flags for each subtask in a task. The flag is 1
        when the subtask has been completed and 0 otherwise. MimicGen only uses this
        when parsing source demonstrations at the start of data generation, and it only
        uses the first 0 -> 1 transition in this signal to detect the end of a subtask.

        Returns:
            subtask_term_signals (dict): dictionary that maps subtask name to termination flag (0 or 1)
        """
        signals = dict()

        # first subtask is opening the drawer (motion relative to drawer)
        # check that drawer is open enough and end effector is far enough from drawer after opening it
        drawer_pos, _ = PoseUtils.unmake_pose(self.get_object_pose(obj_name="CabinetObject_drawer_link", obj_type="body"))
        eef_pos, _ = PoseUtils.unmake_pose(self.get_robot_eef_pose())
        eef_drawer_dist = np.linalg.norm(eef_pos - drawer_pos)
        signals["open"] = int(
            (self.env.sim.data.qpos[self.env.cabinet_qpos_addrs] < -0.10) and (eef_drawer_dist > 0.24)
        )

        # second subtask is grasping the hammer (motion relative to hammer)
        signals["grasp"] = int(self.env._check_grasp(
            gripper=self.env.robots[0].gripper,
            object_geoms=[g for g in self.env.sorting_object.contact_geoms]
        ))

        # final subtask is placing the hammer into the drawer and closing the drawer (motion relative to drawer) - but final subtask signal not needed
        return signals


class MG_MugCleanup(RobosuiteInterface):
    """
    Corresponds to robosuite MugCleanup task and variants.
    """
    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """

        # two relevant objects - mug and drawer
        return dict(
            object=self.get_object_pose(obj_name=self.env.cleanup_object.root_body, obj_type="body"),
            drawer=self.get_object_pose(obj_name=self.env.drawer.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        """
        Gets a dictionary of binary flags for each subtask in a task. The flag is 1
        when the subtask has been completed and 0 otherwise. MimicGen only uses this
        when parsing source demonstrations at the start of data generation, and it only
        uses the first 0 -> 1 transition in this signal to detect the end of a subtask.

        Returns:
            subtask_term_signals (dict): dictionary that maps subtask name to termination flag (0 or 1)
        """
        signals = dict()

        # first subtask is opening the drawer (motion relative to drawer)
        # check that drawer is open enough and end effector is far enough from drawer after opening it
        drawer_pos, _ = PoseUtils.unmake_pose(self.get_object_pose(obj_name="DrawerObject_drawer_link", obj_type="body"))
        eef_pos, _ = PoseUtils.unmake_pose(self.get_robot_eef_pose())
        eef_drawer_dist = np.linalg.norm(eef_pos - drawer_pos)
        signals["open"] = int(
            (self.env.sim.data.qpos[self.env.drawer_qpos_addr] < -0.10) and (eef_drawer_dist > 0.24)
        )

        # second subtask is grasping the mug (motion relative to mug)
        signals["grasp"] = int(self.env._check_grasp_tolerant(
            gripper=self.env.robots[0].gripper,
            object_geoms=[g for g in self.env.cleanup_object.contact_geoms]
        ))

        # final subtask is placing the mug into the drawer and closing the drawer (motion relative to drawer) - but final subtask signal not needed
        return signals


class MG_NutAssembly(RobosuiteInterface):
    """
    Corresponds to robosuite NutAssembly task and variants.
    """
    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """

        # four relevant objects - square and round nuts and pegs
        return dict(
            square_nut=self.get_object_pose(obj_name=self.env.nuts[self.env.nut_to_id["square"]].root_body, obj_type="body"),
            round_nut=self.get_object_pose(obj_name=self.env.nuts[self.env.nut_to_id["round"]].root_body, obj_type="body"),
            square_peg=self.get_object_pose(obj_name="peg1", obj_type="body"),
            round_peg=self.get_object_pose(obj_name="peg2", obj_type="body"),
        )

    def get_subtask_term_signals(self):
        """
        Gets a dictionary of binary flags for each subtask in a task. The flag is 1
        when the subtask has been completed and 0 otherwise. MimicGen only uses this
        when parsing source demonstrations at the start of data generation, and it only
        uses the first 0 -> 1 transition in this signal to detect the end of a subtask.

        Returns:
            subtask_term_signals (dict): dictionary that maps subtask name to termination flag (0 or 1)
        """
        signals = dict()

        # checks which objects are on their correct pegs and records them in @self.objects_on_pegs
        self.env._check_success()

        # first subtask is grasping square nut (motion relative to square nut)
        signals["grasp_square_nut"] = int(self.env._check_grasp(
            gripper=self.env.robots[0].gripper,
            object_geoms=[g for g in self.env.nuts[self.env.nut_to_id["square"]].contact_geoms])
        )

        # second subtask is inserting square nut onto square peg (motion relative to square peg)
        signals["insert_square_nut"] = int(self.env.objects_on_pegs[self.env.nut_to_id["square"]])

        # third subtask is grasping round nut (motion relative to round nut)
        signals["grasp_round_nut"] = int(self.env._check_grasp(
            gripper=self.env.robots[0].gripper,
            object_geoms=[g for g in self.env.nuts[self.env.nut_to_id["round"]].contact_geoms])
        )

        # final subtask is inserting round nut onto round peg (motion relative to round peg) - but final subtask signal is not needed
        return signals


class MG_PickPlace(RobosuiteInterface):
    """
    Corresponds to robosuite PickPlace task and variants.
    """
    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """

        # four relevant objects - milk, bread, cereal, can
        object_poses = dict()
        for obj_name in self.env.object_to_id:
            obj = self.env.objects[self.env.object_to_id[obj_name]]
            object_poses[obj_name] = self.get_object_pose(obj_name=obj.root_body, obj_type="body")
        return object_poses

    def get_subtask_term_signals(self):
        """
        Gets a dictionary of binary flags for each subtask in a task. The flag is 1
        when the subtask has been completed and 0 otherwise. MimicGen only uses this
        when parsing source demonstrations at the start of data generation, and it only
        uses the first 0 -> 1 transition in this signal to detect the end of a subtask.

        Returns:
            subtask_term_signals (dict): dictionary that maps subtask name to termination flag (0 or 1)
        """
        signals = dict()

        # checks which objects are in their correct bins and records them in @self.objects_in_bins
        self.env._check_success()

        object_names_in_order = ["milk", "cereal", "bread", "can"]
        assert set(self.env.object_to_id.keys()) == set(object_names_in_order)
        n_obj = len(object_names_in_order)

        # each subtask is a grasp and then a place
        for i, obj_name in enumerate(object_names_in_order):
            obj_id = self.env.object_to_id[obj_name]

            # first subtask for each object is grasping (motion relative to the object)
            signals["grasp_{}".format(obj_name)] = int(self.env._check_grasp(
                gripper=self.env.robots[0].gripper,
                object_geoms=[g for g in self.env.objects[obj_id].contact_geoms])
            )

            # skip final subtask - unneeded
            if i < (n_obj - 1):
                # second subtask for each object is placement into bin (motion relative to bin)
                signals["place_{}".format(obj_name)] = int(self.env.objects_in_bins[obj_id])

        return signals


class MG_Kitchen(RobosuiteInterface):
    """
    Corresponds to robosuite Kitchen task and variants.
    """
    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """

        # five relevant objects - bread, pot, stove, button, and serving region
        return dict(
            bread=self.get_object_pose(obj_name=self.env.bread_ingredient.root_body, obj_type="body"),
            pot=self.get_object_pose(obj_name=self.env.pot_object.root_body, obj_type="body"),
            stove=self.get_object_pose(obj_name=self.env.stove_object_1.root_body, obj_type="body"),
            button=self.get_object_pose(obj_name=self.env.button_object_1.root_body, obj_type="body"),
            serving_region=self.get_object_pose(obj_name=self.env.serving_region.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        """
        Gets a dictionary of binary flags for each subtask in a task. The flag is 1
        when the subtask has been completed and 0 otherwise. MimicGen only uses this
        when parsing source demonstrations at the start of data generation, and it only
        uses the first 0 -> 1 transition in this signal to detect the end of a subtask.

        Returns:
            subtask_term_signals (dict): dictionary that maps subtask name to termination flag (0 or 1)
        """
        signals = dict()
        
        # first subtask is to flip the switch to turn stove on (motion relative to button)
        signals["stove_on"] = int(self.env.buttons_on[1])

        # second subtask is to grasp pot (motion relative to pot)
        grasped_pot = self.env._check_grasp(
            gripper=self.env.robots[0].gripper,
            object_geoms=[g for g in self.env.pot_object.contact_geoms]
        )
        signals["grasp_pot"] = int(grasped_pot)

        # third subtask is to place pot on stove (motion relative to stove)

        # check for pot-stove contact and that hand is not grasping pot
        pot_bottom_in_contact_with_stove = self.env.check_contact("PotObject_body_0", "Stove1_collision_burner")
        signals["place_pot_on_stove"] = int(pot_bottom_in_contact_with_stove and not grasped_pot)

        # fourth subtask is to grasp bread (motion relative to bread)
        signals["grasp_bread"] = int(self.env._check_grasp(
            gripper=self.env.robots[0].gripper,
            object_geoms=[g for g in self.env.bread_ingredient.contact_geoms]
        ))

        # fifth subtask is to place bread in pot and grasp pot (motion relative to pot)
        signals["place_bread_in_pot"] = int(self.env.check_contact(self.env.bread_ingredient, self.env.pot_object) and grasped_pot)

        # sixth subtask is to place pot in front of serving region and then push it into the serving region (motion relative to serving region)
        pot_pos = self.env.sim.data.body_xpos[self.env.pot_object_id]
        serving_region_pos = self.env.sim.data.body_xpos[self.env.serving_region_id]
        dist_serving_pot = serving_region_pos - pot_pos
        pot_in_serving_region = np.abs(dist_serving_pot[0]) < 0.05 and np.abs(dist_serving_pot[1]) < 0.10 and np.abs(dist_serving_pot[2]) < 0.05
        signals["serve"] = int(pot_in_serving_region)

        # final subtask is to turn off the stove (motion relative to button) - but final subtask signal not needed
        return signals


class MG_CoffeePreparation(RobosuiteInterface):
    """
    Corresponds to robosuite CoffeePreparation task and variants.
    """
    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """

        # four relevant objects - coffee pod, coffee machine, drawer, and mug
        return dict(
            coffee_pod=self.get_object_pose(obj_name=self.env.coffee_pod.root_body, obj_type="body"),
            coffee_machine=self.get_object_pose(obj_name=self.env.coffee_machine.root_body, obj_type="body"),
            drawer=self.get_object_pose(obj_name=self.env.cabinet_object.root_body, obj_type="body"),
            mug=self.get_object_pose(obj_name=self.env.mug.root_body, obj_type="body"),
        )

    def get_subtask_term_signals(self):
        """
        Gets a dictionary of binary flags for each subtask in a task. The flag is 1
        when the subtask has been completed and 0 otherwise. MimicGen only uses this
        when parsing source demonstrations at the start of data generation, and it only
        uses the first 0 -> 1 transition in this signal to detect the end of a subtask.

        Returns:
            subtask_term_signals (dict): dictionary that maps subtask name to termination flag (0 or 1)
        """
        signals = dict()

        metrics = self.env._get_partial_task_metrics()

        # first subtask is grasping mug (motion relative to mug)
        signals["mug_grasp"] = int(metrics["mug_grasp"])

        # second subtask is placing the mug on the coffee machine base and then opening the lid (motion relative to coffee machine)
        signals["mug_place"] = int(self.env._check_mug_placement() and (self.env.sim.data.qpos[self.env.hinge_qpos_addr] > 2.08))

        # third subtask is opening the drawer (motion relative to drawer)
        signals["drawer_open"] = int(self.env.sim.data.qpos[self.env.cabinet_qpos_addr] < -0.19)

        # fourth subtask is grasping the coffee pod (motion relative to coffee pod)
        signals["pod_grasp"] = int(metrics["grasp"])

        # final subtask is inserting pod into machine and closing the lid (motion relative to machine) - but final subtask signal is not needed
        return signals
