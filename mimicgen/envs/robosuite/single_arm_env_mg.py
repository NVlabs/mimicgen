# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import os
import xml.etree.ElementTree as ET
import robosuite
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.utils.mjcf_utils import string_to_array

try:
    # only needed for running hammer cleanup and kitchen tasks
    import robosuite_task_zoo
except ImportError:
    pass

import mimicgen


class SingleArmEnv_MG(SingleArmEnv):
    """
    Custom version of base class for single arm robosuite tasks for mimicgen.
    """
    def edit_model_xml(self, xml_str):
        """
        This function edits the model xml with custom changes, including resolving relative paths,
        applying changes retroactively to existing demonstration files, and other custom scripts.
        Environment subclasses should modify this function to add environment-specific xml editing features.
        Args:
            xml_str (str): Mujoco sim demonstration XML file as string
        Returns:
            str: Edited xml file as string
        """

        path = os.path.split(robosuite.__file__)[0]
        path_split = path.split("/")

        # replace mesh and texture file paths
        tree = ET.fromstring(xml_str)
        root = tree
        asset = root.find("asset")
        meshes = asset.findall("mesh")
        textures = asset.findall("texture")
        all_elements = meshes + textures

        for elem in all_elements:
            old_path = elem.get("file")
            if old_path is None:
                continue
            old_path_split = old_path.split("/")

            # replace all paths to robosuite assets
            check_lst = [loc for loc, val in enumerate(old_path_split) if val == "robosuite"]
            if len(check_lst) > 0:
                ind = max(check_lst) # last occurrence index
                new_path_split = path_split + old_path_split[ind + 1 :]
                new_path = "/".join(new_path_split)
                elem.set("file", new_path)

            # replace all paths to mimicgen assets
            check_lst = [loc for loc, val in enumerate(old_path_split) if val == "mimicgen"]
            if len(check_lst) > 0:
                ind = max(check_lst) # last occurrence index
                new_path_split = os.path.split(mimicgen.__file__)[0].split("/") + old_path_split[ind + 1 :]
                new_path = "/".join(new_path_split)
                elem.set("file", new_path)

            # note: needed since some datasets may have old paths when repo was named mimicgen_envs
            check_lst = [loc for loc, val in enumerate(old_path_split) if val == "mimicgen_envs"]
            if len(check_lst) > 0:
                ind = max(check_lst) # last occurrence index
                new_path_split = os.path.split(mimicgen.__file__)[0].split("/") + old_path_split[ind + 1 :]
                new_path = "/".join(new_path_split)
                elem.set("file", new_path)

            # replace all paths to robosuite_task_zoo assets
            check_lst = [loc for loc, val in enumerate(old_path_split) if val == "robosuite_task_zoo"]
            if len(check_lst) > 0:
                ind = max(check_lst) # last occurrence index
                new_path_split = os.path.split(robosuite_task_zoo.__file__)[0].split("/") + old_path_split[ind + 1 :]
                new_path = "/".join(new_path_split)
                elem.set("file", new_path)

        return ET.tostring(root, encoding="utf8").decode("utf8")

    def _check_grasp_tolerant(self, gripper, object_geoms):
        """
        Tolerant version of check grasp function - often needed for checking grasp with Shapenet mugs.

        TODO: only tested for panda, update for other robots.
        """
        check_1 = self._check_grasp(gripper=gripper, object_geoms=object_geoms)

        check_2 = self._check_grasp(gripper=["gripper0_finger1_collision", "gripper0_finger2_pad_collision"], object_geoms=object_geoms)

        check_3 = self._check_grasp(gripper=["gripper0_finger2_collision", "gripper0_finger1_pad_collision"], object_geoms=object_geoms)

        return check_1 or check_2 or check_3

    def _add_agentview_full_camera(self, arena):
        """
        Add camera with full perspective of tabletop.
        """
        arena.set_camera(
            camera_name="agentview_full",
            pos=string_to_array("0.753078462147161 2.062036796036723e-08 1.5194726087166726"),
            quat=string_to_array("0.6432409286499023 0.293668270111084 0.2936684489250183 0.6432408690452576"),
        )
