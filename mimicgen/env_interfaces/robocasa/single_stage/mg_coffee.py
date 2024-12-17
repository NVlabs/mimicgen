from mimicgen.env_interfaces.robosuite import RobosuiteInterface


class MG_CoffeeSetupMug(RobosuiteInterface):

    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """
        return dict(
            obj=self.get_object_pose(obj_name=self.env.objects["obj"].root_body, obj_type="body"),
            coffee_machine=self.get_object_pose(obj_name="{}_{}".format(self.env.coffee_machine.name, "receptacle_place_site"), obj_type="site"),
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

        contact_obj = self.env.check_contact(
            self.env.robots[0].gripper["right"],
            self.env.objects["obj"],
        )
        signals["stage_contact_obj"] = int(contact_obj)
        signals["stage_place_obj"] = int(self.env._check_success())

        return signals
    

class MG_CoffeeServeMug(RobosuiteInterface):

    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """
        return dict(
            obj=self.get_object_pose(obj_name=self.env.objects["obj"].root_body, obj_type="body"),
            coffee_machine=self.get_object_pose(obj_name="{}_{}".format(self.env.coffee_machine.name, "receptacle_place_site"), obj_type="site"),
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

        contact_obj = self.env.check_contact(
            self.env.robots[0].gripper["right"],
            self.env.objects["obj"],
        )
        signals["stage_contact_obj"] = int(contact_obj)
        signals["stage_place_obj"] = int(self.env._check_success())

        return signals
    

class MG_CoffeePressButton(RobosuiteInterface):

    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """
        return dict(
            button=self.get_object_pose(obj_name="{}_{}".format(self.env.coffee_machine.name, "start_button"), obj_type="geom"),
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
        signals["success"] = int(self.env._check_success())
        return signals