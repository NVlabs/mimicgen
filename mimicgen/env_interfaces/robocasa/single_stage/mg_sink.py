from mimicgen.env_interfaces.robosuite import RobosuiteInterface

from robosuite.utils.mjcf_utils import find_elements


class MG_TurnOnSinkFaucet(RobosuiteInterface):

    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """
        return dict(
            handle=self.get_object_pose(obj_name=self.env.sink.name + "_handle_main", obj_type="geom"),
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
        check_contact = self.env.check_contact(
            self.env.robots[0].gripper["right"],
            self.env.sink.name + "_handle_main",
        )
        signals["stage_contact_handle"] = int(check_contact)
        signals["success"] = int(self.env._check_success())
        return signals
    

class MG_TurnOffSinkFaucet(RobosuiteInterface):

    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """
        return dict(
            handle=self.get_object_pose(obj_name=self.env.sink.name + "_handle_main", obj_type="geom"),
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
        check_contact = self.env.check_contact(
            self.env.robots[0].gripper["right"],
            self.env.sink.name + "_handle_main",
        )
        signals["stage_contact_handle"] = int(check_contact)
        signals["success"] = int(self.env._check_success())
        return signals
    

class MG_TurnSinkSpout(RobosuiteInterface):

    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """
        return dict(
            spout=self.get_object_pose(obj_name=self.env.sink.name + "_spout_top", obj_type="geom"),
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
        spout_body = find_elements(
            self.env.sink.worldbody,
            tags="body",
            attribs={"name": "{}{}".format(self.env.sink.name, "_spout")},
            return_first=True,
        )
        spout_geoms = find_elements(spout_body, tags="geom", return_first=False)
        spout_geom_names = [e.get("name") for e in spout_geoms]
        contact_spout = self.env.check_contact(
            self.env.robots[0].gripper["right"],
            spout_geom_names,
        )
        signals["stage_contact_spout"] = int(contact_spout)
        signals["success"] = int(self.env._check_success())
        return signals