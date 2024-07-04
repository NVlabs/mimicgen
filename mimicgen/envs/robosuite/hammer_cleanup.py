# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Contains environments for BUDS hammer place task from robosuite task zoo repo.
(https://github.com/ARISE-Initiative/robosuite-task-zoo)
"""

import os
import random
import numpy as np
from six import with_metaclass
from copy import deepcopy

import robosuite
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.models.objects import HammerObject, MujocoXMLObject
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.mjcf_utils import CustomMaterial, array_to_string, string_to_array, find_elements, add_material
from robosuite.utils.buffers import RingBuffer

import robosuite_task_zoo
from robosuite_task_zoo.environments.manipulation.hammer_place import HammerPlaceEnv

import mimicgen
from mimicgen.envs.robosuite.single_arm_env_mg import SingleArmEnv_MG
from mimicgen.models.robosuite.objects import DrawerObject


class HammerCleanup_D0(HammerPlaceEnv, SingleArmEnv_MG):
    """
    Augment BUDS hammer place task for mimicgen.
    """
    def __init__(self, robot_init_qpos=None, **kwargs):
        self.robot_init_qpos = robot_init_qpos
        HammerPlaceEnv.__init__(self, **kwargs)

    def edit_model_xml(self, xml_str):
        # make sure we don't get a conflict for function implementation
        return SingleArmEnv_MG.edit_model_xml(self, xml_str)

    def _load_model(self):
        """
        Copied exactly from HammerPlaceEnv, but swaps out the cabinet object.
        """
        SingleArmEnv._load_model(self)

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_offset=self.table_offset,
            table_friction=(0.6, 0.005, 0.0001)
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Modify default agentview camera
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.5386131746834771, -4.392035683362857e-09, 1.4903500240372423],
            quat=[0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]
        )

        mujoco_arena.set_camera(
            camera_name="sideview",
            pos=[0.5586131746834771, 0.3, 1.2903500240372423],
            quat=[0.4144233167171478, 0.3100920617580414, 0.49641484022140503, 0.6968992352485657]
        )
        
        
        bread = CustomMaterial(
            texture="Bread",
            tex_name="bread",
            mat_name="MatBread",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "3 3", "specular": "0.4","shininess": "0.1"}
        )

        darkwood = CustomMaterial(
            texture="WoodDark",
            tex_name="darkwood",
            mat_name="MatDarkWood",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "3 3", "specular": "0.4","shininess": "0.1"}
        )

        lightwood = CustomMaterial(
            texture="WoodLight",
            tex_name="lightwood",
            mat_name="MatLightWood",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "3 3", "specular": "0.4","shininess": "0.1"}
        )

        metal = CustomMaterial(
            texture="Metal",
            tex_name="metal",
            mat_name="MatMetal",
            tex_attrib={"type": "cube"},
            mat_attrib={"specular": "1", "shininess": "0.3", "rgba": "0.9 0.9 0.9 1"}
        )

        tex_attrib = {
            "type": "cube"
        }

        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1"
        }
        
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="MatRedWood",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        
        bluewood = CustomMaterial(
            texture="WoodBlue",
            tex_name="bluewood",
            mat_name="handle1_mat",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"},
        )

        ceramic = CustomMaterial(
            texture="Ceramic",
            tex_name="ceramic",
            mat_name="MatCeramic",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        ingredient_size = [0.03, 0.018, 0.025]
        
        self.sorting_object = HammerObject(name="hammer",
                                           handle_length=(0.045, 0.05),
                                           handle_radius=(0.012, 0.012),
                                           head_density_ratio=1.0
        )

        self.cabinet_object = DrawerObject(
            name="CabinetObject")
        cabinet_object = self.cabinet_object.get_obj(); cabinet_object.set("pos", array_to_string((0.2, 0.30, 0.03))); mujoco_arena.table_body.append(cabinet_object)
        
        for obj_body in [
                self.cabinet_object,
        ]:
            for material in [lightwood, darkwood, metal, redwood, ceramic]:
                tex_element, mat_element, _, used = add_material(root=obj_body.worldbody,
                                                                 naming_prefix=obj_body.naming_prefix,
                                                                 custom_material=deepcopy(material))
                obj_body.asset.append(tex_element)
                obj_body.asset.append(mat_element)

        ingredient_size = [0.015, 0.025, 0.02]
        
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        self.placement_initializer.append_sampler(
        sampler = UniformRandomSampler(
            name="ObjectSampler-pot",
            mujoco_objects=self.sorting_object,
            x_range=[0.10,  0.18],
            y_range=[-0.20, -0.13],
            rotation=(-0.1, 0.1),
            rotation_axis='z',
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.02,
        ))
        
        mujoco_objects = [
            self.sorting_object,
        ]

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots], 
            mujoco_objects=mujoco_objects,
        )
        self.objects = [
            self.sorting_object,
            self.cabinet_object,
        ]
        self.model.merge_assets(self.sorting_object)
        self.model.merge_assets(self.cabinet_object)


class HammerCleanup_D1(HammerCleanup_D0):
    """
    Move object and drawer with wide initialization. Note we had to make some objects movable that were fixtures before.
    """
    def _check_success(self):
        """
        Update from superclass to have a more stringent check that's not buggy
        (e.g. there's no check in x-position before) and that supports
        different drawer (cabinet) positions.
        """
        object_pos = self.sim.data.body_xpos[self.sorting_object_id]
        # object_in_drawer = 1.0 > object_pos[2] > 0.94 and object_pos[1] > 0.22

        cabinet_closed = self.sim.data.qpos[self.cabinet_qpos_addrs] > -0.01

        # new contact-based drawer check - object in contact with bottom drawer geom
        drawer_bottom_geom = "CabinetObject_drawer_bottom"
        object_in_drawer = self.check_contact(drawer_bottom_geom, self.sorting_object)

        return object_in_drawer and cabinet_closed

    def _get_sorting_object(self):
        """
        Method that constructs object to place into drawer. Subclasses can override this method to
        construct different objects.
        """
        return HammerObject(
            name="hammer",
            handle_length=(0.045, 0.05),
            handle_radius=(0.012, 0.012),
            head_density_ratio=1.0,
        )

    def _get_initial_placement_bounds(self):
        """
        Internal function to get bounds for randomization of initial placements of objects (e.g.
        what happens when env.reset is called). Should return a dictionary with the following
        structure:
            object_name
                x: 2-tuple for low and high values for uniform sampling of x-position
                y: 2-tuple for low and high values for uniform sampling of y-position
                z_rot: 2-tuple for low and high values for uniform sampling of z-rotation
                reference: np array of shape (3,) for reference position in world frame (assumed to be static and not change)
        """
        return dict(
            hammer=dict(
                x=(-0.2, 0.2),
                y=(-0.25, -0.13),
                z_rot=(0., 2. * np.pi),
                reference=self.table_offset,
                init_quat=self.sorting_object.init_quat,
                # NOTE: this rotation axis needs to be y, not z because of hammer's init_quat
                rotation_axis="y",
            ),
            drawer=dict(
                x=(0.0, 0.2),
                y=(0.2, 0.3),
                # z_rot=(0., 0.),
                z_rot=(-np.pi / 6., np.pi / 6.),
                reference=self.table_offset,
            ),
        )

    def _get_placement_initializer(self):
        """
        Helper function for defining placement initializer and object sampling bounds
        """
        bounds = self._get_initial_placement_bounds()
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="ObjectSampler-hammer",
                mujoco_objects=self.sorting_object,
                x_range=bounds["hammer"]["x"],
                y_range=bounds["hammer"]["y"],
                rotation=bounds["hammer"]["z_rot"],
                rotation_axis=bounds["hammer"]["rotation_axis"],
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["hammer"]["reference"],
                z_offset=0.02,
            )
        )
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="ObjectSampler-drawer",
                mujoco_objects=self.cabinet_object,
                x_range=bounds["drawer"]["x"],
                y_range=bounds["drawer"]["y"],
                rotation=bounds["drawer"]["z_rot"],
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["drawer"]["reference"],
                z_offset=0.03,
            )
        )

    def _load_model(self):
        """
        Update to include drawer (cabinet) in placement initializer.
        """
        SingleArmEnv._load_model(self)

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # Adjust initial robot joint configuration accordingly
        if self.robot_init_qpos is not None:
            self.robots[0].init_qpos = self.robot_init_qpos

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_offset=self.table_offset,
            table_friction=(0.6, 0.005, 0.0001)
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Modify default agentview camera
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.5386131746834771, -4.392035683362857e-09, 1.4903500240372423],
            quat=[0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]
        )

        mujoco_arena.set_camera(
            camera_name="sideview",
            pos=[0.5586131746834771, 0.3, 1.2903500240372423],
            quat=[0.4144233167171478, 0.3100920617580414, 0.49641484022140503, 0.6968992352485657]
        )

        darkwood = CustomMaterial(
            texture="WoodDark",
            tex_name="darkwood",
            mat_name="MatDarkWood",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "3 3", "specular": "0.4","shininess": "0.1"}
        )

        lightwood = CustomMaterial(
            texture="WoodLight",
            tex_name="lightwood",
            mat_name="MatLightWood",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "3 3", "specular": "0.4","shininess": "0.1"}
        )

        metal = CustomMaterial(
            texture="Metal",
            tex_name="metal",
            mat_name="MatMetal",
            tex_attrib={"type": "cube"},
            mat_attrib={"specular": "1", "shininess": "0.3", "rgba": "0.9 0.9 0.9 1"}
        )

        tex_attrib = {
            "type": "cube"
        }

        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1"
        }
        
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="MatRedWood",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        
        bluewood = CustomMaterial(
            texture="WoodBlue",
            tex_name="bluewood",
            mat_name="handle1_mat",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"},
        )

        ceramic = CustomMaterial(
            texture="Ceramic",
            tex_name="ceramic",
            mat_name="MatCeramic",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        
        self.sorting_object = self._get_sorting_object()

        self.cabinet_object = DrawerObject(name="CabinetObject")

        # # old: manually set position in xml and add to mujoco arena
        # cabinet_object = self.cabinet_object.get_obj()
        # cabinet_object.set("pos", array_to_string((0.2, 0.30, 0.03)))
        # mujoco_arena.table_body.append(cabinet_object)
        
        for obj_body in [
                self.cabinet_object,
        ]:
            for material in [lightwood, darkwood, metal, redwood, ceramic]:
                tex_element, mat_element, _, used = add_material(root=obj_body.worldbody,
                                                                 naming_prefix=obj_body.naming_prefix,
                                                                 custom_material=deepcopy(material))
                obj_body.asset.append(tex_element)
                obj_body.asset.append(mat_element)
        
        self._get_placement_initializer()
        
        mujoco_objects = [
            self.sorting_object,
            self.cabinet_object,
        ]

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots], 
            mujoco_objects=mujoco_objects,
        )
        self.objects = [
            self.sorting_object,
            self.cabinet_object,
        ]
        self.model.merge_assets(self.sorting_object)
        self.model.merge_assets(self.cabinet_object)

    def _reset_internal(self):
        """
        Update to make sure placement initializer can be used to set drawer (cabinet) pose
        even though it doesn't have a joint.
        """
        SingleArmEnv._reset_internal(self)

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            for obj_pos, obj_quat, obj in object_placements.values():
                if obj is self.cabinet_object:
                    # object is fixture - set pose in model
                    body_id = self.sim.model.body_name2id(obj.root_body)
                    obj_pos_to_set = np.array(obj_pos)
                    obj_pos_to_set[2] = 0.905 # hardcode z-value to correspond to parent class
                    self.sim.model.body_pos[body_id] = obj_pos_to_set
                    self.sim.model.body_quat[body_id] = obj_quat
                else:
                    # object has free joint - use it to set pose
                    self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        self.ee_force_bias = np.zeros(3)
        self.ee_torque_bias = np.zeros(3)
        self._history_force_torque = RingBuffer(dim=6, length=16)
        self._recent_force_torque = []
