from mimicgen.env_interfaces.robosuite import RobosuiteInterface

class MG_PnPCabToCounter(RobosuiteInterface):

    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """
        return dict(
            obj=self.get_object_pose(obj_name=self.env.objects["obj"].root_body, obj_type="body"),
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
        contact_obj = self.env.check_contact(self.env.robots[0].gripper["right"], self.env.objects["obj"])
        signals["stage_contact_obj"] = int(contact_obj)
        signals["stage_place_obj"] = int(self.env._check_success())
        return signals


class MG_PnPCounterToCab(RobosuiteInterface):

    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """
        return dict(
            obj=self.get_object_pose(obj_name=self.env.objects["obj"].root_body, obj_type="body"),
            cab=self.get_object_pose(obj_name="{}_bottom".format(self.env.cab.name), obj_type="geom"),
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
        contact_obj = self.env.check_contact(self.env.robots[0].gripper["right"], self.env.objects["obj"])
        signals["stage_contact_obj"] = int(contact_obj)
        signals["stage_place_obj"] = int(self.env._check_success())
        return signals


class MG_PnPCounterToSink(RobosuiteInterface):

    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """
        return dict(
            obj=self.get_object_pose(obj_name=self.env.objects["obj"].root_body, obj_type="body"),
            sink=self.get_object_pose(obj_name="{}_{}".format(self.env.sink.name, "bottom"), obj_type="geom"),
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
        contact_obj = self.env.check_contact(self.env.robots[0].gripper["right"], self.env.objects["obj"])
        signals["stage_contact_obj"] = int(contact_obj)
        signals["stage_place_obj"] = int(self.env._check_success())
        return signals


class MG_PnPSinkToCounter(RobosuiteInterface):

   def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """
        return dict(
            obj=self.get_object_pose(obj_name=self.env.objects["obj"].root_body, obj_type="body"),
            container=self.get_object_pose(obj_name=self.env.objects["container"].root_body, obj_type="body"),
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
        contact_obj = self.env.check_contact(self.env.robots[0].gripper["right"], self.env.objects["obj"])
        signals["stage_contact_obj"] = int(contact_obj)
        signals["stage_place_obj"] = int(self.env._check_success())
        return signals


class MG_PnPCounterToMicrowave(RobosuiteInterface):

    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """
        return dict(
            obj=self.get_object_pose(obj_name=self.env.objects["obj"].root_body, obj_type="body"),
            container=self.get_object_pose(obj_name=self.env.objects["container"].root_body, obj_type="body"),
            microwave=self.get_object_pose(obj_name="{}_{}".format(self.env.microwave.name, "tray"), obj_type="geom"),
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
        contact_obj = self.env.check_contact(self.env.robots[0].gripper["right"], self.env.objects["obj"])
        signals["stage_contact_obj"] = int(contact_obj)
        signals["stage_place_obj"] = int(self.env._check_success())
        return signals


class MG_PnPMicrowaveToCounter(RobosuiteInterface):

    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """
        return dict(
            obj=self.get_object_pose(obj_name=self.env.objects["obj"].root_body, obj_type="body"),
            container=self.get_object_pose(obj_name=self.env.objects["container"].root_body, obj_type="body"),
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
        contact_obj = self.env.check_contact(self.env.robots[0].gripper["right"], self.env.objects["obj"])
        signals["stage_contact_obj"] = int(contact_obj)
        signals["stage_place_obj"] = int(self.env._check_success())
        return signals


class MG_PnPCounterToStove(RobosuiteInterface):

    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """
        return dict(
            obj=self.get_object_pose(obj_name=self.env.objects["obj"].root_body, obj_type="body"),
            cookware=self.get_object_pose(obj_name=self.env.objects["container"].root_body, obj_type="body"),
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
        contact_obj = self.env.check_contact(self.env.robots[0].gripper["right"], self.env.objects["obj"])
        signals["stage_contact_obj"] = int(contact_obj)
        signals["stage_place_obj"] = int(self.env._check_success())
        return signals
    

class MG_PnPStoveToCounter(RobosuiteInterface):

    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """
        return dict(
            obj=self.get_object_pose(obj_name=self.env.objects["obj"].root_body, obj_type="body"),
            container=self.get_object_pose(obj_name=self.env.objects["container"].root_body, obj_type="body"),
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
        contact_obj = self.env.check_contact(self.env.robots[0].gripper["right"], self.env.objects["obj"])
        signals["stage_contact_obj"] = int(contact_obj)
        signals["stage_place_obj"] = int(self.env._check_success())
        return signals