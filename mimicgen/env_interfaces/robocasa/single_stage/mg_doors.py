from mimicgen.env_interfaces.robosuite import RobosuiteInterface

import numpy as np


class MG_OpenSingleDoor(RobosuiteInterface):

    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """
        return dict(
            handle=self.get_object_pose(obj_name=self.env.door_fxtr.handle_name, obj_type="geom"),
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
        contact_handle = self.env.check_contact(
            self.env.robots[0].gripper["right"],
            self.env.door_fxtr.handle_name,
        )
        signals["stage_contact_handle"] = int(contact_handle)
        signals["success"] = int(self.env._check_success())
        return signals


class MG_OpenDoubleDoor(RobosuiteInterface):

    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """
        return dict(
            handle_right=self.get_object_pose(obj_name=self.env.door_fxtr.right_handle_name, obj_type="geom"),
            handle_left=self.get_object_pose(obj_name=self.env.door_fxtr.left_handle_name, obj_type="geom"),
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
        door_state = self.env.door_fxtr.get_door_state(self.env)

        for side in ["left", "right"]:
            contact_handle = self.env.check_contact(
                self.env.robots[0].gripper["right"],
                self.env.door_fxtr.left_handle_name if side == "left" else self.env.door_fxtr.right_handle_name,
            )

            door_open = door_state["{}_door".format(side)] > 0.90

            signals["stage_contact_{}_handle".format(side)] = int(contact_handle)
            signals["stage_open_{}_door".format(side)] = int(door_open) # and robot_cleared_door)
        signals["success"] = int(self.env._check_success())

        return signals
    

class MG_CloseSingleDoor(RobosuiteInterface):

    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """
        return dict(
            handle=self.get_object_pose(obj_name=self.env.door_fxtr.handle_name, obj_type="geom"),
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

        door_points = []
        for p_name in ["p1", "p2", "p3"]:
            site_name = "{}_door_{}".format(self.env.door_fxtr.name, p_name)
            p = self.env.sim.data.site_xpos[self.env.sim.model.site_name2id(site_name)]
            door_points.append(p)
        gripper_site_pos = self.env.sim.data.site_xpos[self.env.robots[0].eef_site_id["right"]]
        robot_cleared_door = np.dot(
            gripper_site_pos - door_points[1],
            np.cross(door_points[1] - door_points[0], door_points[2] - door_points[0])
        ) > 0

        signals["stage_clear_door"] = int(robot_cleared_door)
        signals["success"] = int(self.env._check_success())

        return signals


class MG_CloseDoubleDoor(RobosuiteInterface):

    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """
        fxtr_name = self.env.door_fxtr.name
        return dict(
            door_right=self.get_object_pose(obj_name="{}_hingeleftdoor".format(fxtr_name), obj_type="body"),
            door_left=self.get_object_pose(obj_name="{}_hingeleftdoor".format(fxtr_name), obj_type="body"),
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
        door_state = self.env.door_fxtr.get_door_state(self.env)

        for side in ["left", "right"]:
            door_points = []
            for p_name in ["p1", "p2", "p3"]:
                site_name = "{}_{}door_{}".format(self.env.door_fxtr.name, side, p_name)
                p = self.env.sim.data.site_xpos[self.env.sim.model.site_name2id(site_name)]
                door_points.append(p)
            gripper_site_pos = self.env.sim.data.site_xpos[self.env.robots[0].eef_site_id["right"]]
            robot_cleared_door = np.dot(
                gripper_site_pos - door_points[1],
                np.cross(door_points[1] - door_points[0], door_points[2] - door_points[0])
            ) * (-1 if side == "right" else 1) > 0
            
            door_closed = door_state["{}_door".format(side)] < 0.10
            
            signals["stage_clear_{}_door".format(side)] = int(robot_cleared_door)
            signals["stage_close_{}_door".format(side)] = int(door_closed)
        
        signals["success"] = int(self.env._check_success())

        return signals