from mimicgen.env_interfaces.robosuite import RobosuiteInterface
from robosuite.utils.mjcf_utils import find_elements


class MG_TurnOnStove(RobosuiteInterface):
    
    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """
        return dict(
            knob=self.get_object_pose(obj_name=self.env.stove.name + "_knob_{}_main".format(self.env.knob), obj_type="geom"),
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
        knob_body = find_elements(
            self.env.stove.worldbody,
            tags="body",
            attribs={"name": "{}{}".format(self.env.stove.name, "_knob_{}".format(self.env.knob))},
            return_first=True,
        )
        knob_geoms = find_elements(knob_body, tags="geom", return_first=False)
        knob_geom_names = [e.get("name") for e in knob_geoms]
        check_contact = self.env.check_contact(
            self.env.robots[0].gripper["right"],
            knob_geom_names,
        )
        signals["stage_contact_knob"] = int(check_contact)
        signals["success"] = int(self.env._check_success())

        return signals
    

class MG_TurnOffStove(RobosuiteInterface):

    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """
        return dict(
            knob=self.get_object_pose(obj_name=self.env.stove.name + "_knob_{}_main".format(self.env.knob), obj_type="geom"),
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
        knob_body = find_elements(
            self.env.stove.worldbody,
            tags="body",
            attribs={"name": "{}{}".format(self.env.stove.name, "_knob_{}".format(self.env.knob))},
            return_first=True,
        )
        knob_geoms = find_elements(knob_body, tags="geom", return_first=False)
        knob_geom_names = [e.get("name") for e in knob_geoms]
        check_contact = self.env.check_contact(
            self.env.robots[0].gripper["right"],
            knob_geom_names,
        )
        signals["stage_contact_knob"] = int(check_contact)
        signals["success"] = int(self.env._check_success())
        return signals