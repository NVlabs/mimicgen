# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Contains environments for BUDS kitchen task from robosuite task zoo repo.
((https://github.com/ARISE-Initiative/robosuite-task-zoo))
"""
import os
import numpy as np
from six import with_metaclass
from copy import deepcopy

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.models.objects import BoxObject, MujocoXMLObject
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.mjcf_utils import CustomMaterial, array_to_string, string_to_array, find_elements, add_material
from robosuite.utils.buffers import RingBuffer

import robosuite_task_zoo
from robosuite_task_zoo.environments.manipulation.kitchen import KitchenEnv
from robosuite_task_zoo.models.kitchen import PotObject, StoveObject, ButtonObject, ServingRegionObject

import mimicgen
from mimicgen.envs.robosuite.single_arm_env_mg import SingleArmEnv_MG


class StoveObjectNew(StoveObject):
    """
    Override some offsets for placement sampler.
    """
    @property
    def bottom_offset(self):
        # unused since we directly hardcode z
        return np.array([0, 0, -0.02])

    @property
    def top_offset(self):
        # unused since we directly hardcode z
        return np.array([0, 0, 0.02])

    @property
    def horizontal_radius(self):
        return 0.1


class ButtonObjectNew(ButtonObject):
    """
    Override some offsets for placement sampler.
    """
    @property
    def horizontal_radius(self):
        return 0.04


class ServingRegionObjectNew(MujocoXMLObject):
    """
    Override some offsets for placement sampler, and also
    turn the site into a visual-only geom so that it shows up
    in the first env step.
    """
    def __init__(self, name, joints=None):
        # our custom serving region xml - turn site into visual-only geom so that it shows up on env reset (instead
        # of after first env step)
        path_to_serving_region_xml = os.path.join(mimicgen.__path__[0], "models/robosuite/assets/objects/serving_region.xml")
        super().__init__(path_to_serving_region_xml,
                         name=name, joints=None, obj_type="all", duplicate_collision_geoms=True)

    @property
    def horizontal_radius(self):
        return 0.123


class Kitchen_D0(KitchenEnv, SingleArmEnv_MG):
    """
    Augment BUDS kitchen task for mimicgen.
    """
    def __init__(self, **kwargs):
        KitchenEnv.__init__(self, **kwargs)

        # some additional variables for better success check
        self.has_stove_turned_on_with_pot_and_object = False

    def edit_model_xml(self, xml_str):
        # make sure we don't get a conflict for function implementation
        return SingleArmEnv_MG.edit_model_xml(self, xml_str)

    def _reset_internal(self):
        """
        Update from superclass to ensure we reset state variables.
        """
        KitchenEnv._reset_internal(self)

        # reset state variables
        self.has_stove_turned_on = False
        self.has_stove_turned_on_with_pot_and_object = False

    def _check_success(self):
        """
        Update from superclass to make a more stringent success check that ensures
        the pot has been placed on the stove while it is on with the object in the pot
        """
        pot_pos = self.sim.data.body_xpos[self.pot_object_id]
        serving_region_pos = self.sim.data.body_xpos[self.serving_region_id]
        dist_serving_pot = serving_region_pos - pot_pos
        pot_in_serving_region = np.abs(dist_serving_pot[0]) < 0.05 and np.abs(dist_serving_pot[1]) < 0.10 and np.abs(dist_serving_pot[2]) < 0.05
        
        # if pot bottom is in contact with stove
        pot_bottom_in_contact_with_stove = self.check_contact("PotObject_body_0", "Stove1_collision_burner")

        # if object is in pot
        object_in_pot = self.check_contact(self.bread_ingredient, self.pot_object)
        stove_turned_off = not self.buttons_on[1]
        if not stove_turned_off:
            self.has_stove_turned_on = True
            if self.has_stove_turned_on and object_in_pot and pot_bottom_in_contact_with_stove:
                self.has_stove_turned_on_with_pot_and_object = True

        return pot_in_serving_region and stove_turned_off and object_in_pot and self.has_stove_turned_on_with_pot_and_object

    def _get_observations(self, force_update=False):
        """
        Make sure switch states are up-to-date before observations are returned - this is also
        important for scripts that reset to intermediate demonstration states.
        """
        self._post_process()
        return KitchenEnv._get_observations(self, force_update=force_update)


class Kitchen_D1(Kitchen_D0):
    """
    Specify wider distribution for objects including objects that didn't move before. We also had to make some objects 
    movable that were fixtures before.
    """
    def _reset_internal(self):
        """
        Update to make sure placement initializer can be used to set poses of objects
        that used to be fixtures before.
        """
        SingleArmEnv._reset_internal(self)
        self.has_stove_turned_on = False 

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            for obj_pos, obj_quat, obj in object_placements.values():
                if obj.name in self._hardcoded_z_offsets:
                    # object is fixture - set pose in model
                    body_id = self.sim.model.body_name2id(obj.root_body)
                    obj_pos_to_set = np.array(obj_pos)
                    obj_pos_to_set[2] = self._hardcoded_z_offsets[obj.name] # hardcode z-value to correspond to parent class
                    self.sim.model.body_pos[body_id] = obj_pos_to_set
                    self.sim.model.body_quat[body_id] = obj_quat
                else:
                    # object has free joint - use it to set pose
                    self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        self.ee_force_bias = np.zeros(3)
        self.ee_torque_bias = np.zeros(3)
        self._history_force_torque = RingBuffer(dim=6, length=16)
        self._recent_force_torque = []

        # reset state variables
        self.buttons_on = {1: False}
        self.has_stove_turned_on = False
        self.has_stove_turned_on_with_pot_and_object = False

        # make sure switch is off
        self.sim.data.qpos[self.button_qpos_addrs[1]] = -0.3
        for stove_num, stove_status in self.buttons_on.items():
            self.stoves[stove_num].set_sites_visibility(sim=self.sim, visible=stove_status)

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
        # Broader bounds for all objects.
        return dict(
            bread=dict(
                x=(-0.2, 0.0),
                y=(-0.25, -0.05),
                # z_rot=(-np.pi / 2., -np.pi / 2.),
                z_rot=(-np.pi / 2., np.pi / 2.),
                reference=self.table_offset,
            ),
            pot=dict(
                x=(0.08, 0.18),
                y=(-0.2, -0.05),
                # z_rot=(-0.1, 0.1),
                z_rot=(-np.pi / 6., np.pi / 6.),
                reference=self.table_offset,
            ),
            stove=dict(
                x=(0.06, 0.23),
                y=(0.095, 0.25),
                z_rot=(0., 0.),
                reference=self.table_offset,
            ),
            button=dict(
                x=(-0.2, 0.06),
                y=(0.05, 0.2),
                z_rot=(np.pi, np.pi), # make z-rotation consistent with base env
                reference=self.table_offset,
            ),
            serving_region=dict(
                x=(0.345, 0.345),
                y=(-0.2, -0.05),
                z_rot=(0., 0.),
                reference=self.table_offset,
            ),
        )

    def _get_placement_initializer(self):
        bounds = self._get_initial_placement_bounds()
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        # object
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="ObjectSampler-bread",
                mujoco_objects=self.bread_ingredient,
                x_range=bounds["bread"]["x"],
                y_range=bounds["bread"]["y"],
                rotation=bounds["bread"]["z_rot"],
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["bread"]["reference"],
                z_offset=0.01,
            )
        )

        # pot
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="ObjectSampler-pot",
                mujoco_objects=self.pot_object,
                x_range=bounds["pot"]["x"],
                y_range=bounds["pot"]["y"],
                rotation=bounds["pot"]["z_rot"],
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["pot"]["reference"],
                # z_offset=0.02,
                z_offset=-0.11, # account for pot vertical sites being wrong
            )
        )

        # stove
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="ObjectSampler-stove",
                mujoco_objects=self.stove_object_1,
                x_range=bounds["stove"]["x"],
                y_range=bounds["stove"]["y"],
                rotation=bounds["stove"]["z_rot"],
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["stove"]["reference"],
                z_offset=0.02,
            )
        )

        # button
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="ObjectSampler-button",
                mujoco_objects=self.button_object_1,
                x_range=bounds["button"]["x"],
                y_range=bounds["button"]["y"],
                rotation=bounds["button"]["z_rot"],
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["button"]["reference"],
                z_offset=0.02,
            )
        )

        # serving region
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="ObjectSampler-serving",
                mujoco_objects=self.serving_region,
                x_range=bounds["serving_region"]["x"],
                y_range=bounds["serving_region"]["y"],
                rotation=bounds["serving_region"]["z_rot"],
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["serving_region"]["reference"],
                z_offset=0.003,
            )
        )

    def _load_model(self):
        """
        Update to include fixtures that didn't move before in placement initializer, so
        they can move on each episode reset. Also updates the list of objects so that
        we get observables for the button.
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
            quat=[0.4144233167171478, 0.3100920617580414,
            0.49641484022140503, 0.6968992352485657]
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

        self.stove_object_1 = StoveObjectNew(
            name="Stove1",
            joints=None,
        )

        # # old: manually set position in xml and add to mujoco arena
        # stove_body = self.stove_object_1.get_obj()
        # stove_body.set("pos", array_to_string((0.23, 0.095, 0.02)))
        # mujoco_arena.table_body.append(stove_body)

        self.button_object_1 = ButtonObjectNew(
            name="Button1",
        )

        # # old: manually set position in xml and add to mujoco arena
        # button_body = self.button_object_1.get_obj()
        # button_body.set("quat", array_to_string((0., 0., 0., 1.)))
        # button_body.set("pos", array_to_string((0.06, 0.10, 0.02)))
        # mujoco_arena.table_body.append(button_body)

        self.serving_region = ServingRegionObjectNew(
            name="ServingRegionRed"
        )

        # # old: manually set position in xml and add to mujoco arena
        # serving_region_object = self.serving_region.get_obj()
        # serving_region_object.set("pos", array_to_string((0.345, -0.15, 0.003)))
        # mujoco_arena.table_body.append(serving_region_object)
        
        self.pot_object = PotObject(
            name="PotObject",
        )
        
        for obj_body in [
                self.button_object_1,
                self.stove_object_1,
                self.serving_region,
        ]:
            for material in [darkwood, metal, redwood]:
                tex_element, mat_element, _, used = add_material(root=obj_body.worldbody,
                                                                 naming_prefix=obj_body.naming_prefix,
                                                                 custom_material=deepcopy(material))
                obj_body.asset.append(tex_element)
                obj_body.asset.append(mat_element)

        ingredient_size = [0.015, 0.025, 0.02]

        self.bread_ingredient = BoxObject(
            name="cube_bread",
            size_min=ingredient_size,
            size_max=ingredient_size,
            rgba=[1, 0, 0, 1],
            material=bread,
            density=500.,
        )
        
        # make placement initializer
        self._get_placement_initializer()
        
        mujoco_objects = [self.bread_ingredient,
                          self.pot_object,
                          self.stove_object_1,
                          self.button_object_1,
                          self.serving_region,
        ]

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots], 
            mujoco_objects=mujoco_objects,
        )
        self.stoves = {1: self.stove_object_1,
                       # 2: self.stove_object_2
        }

        self.num_stoves = len(self.stoves.keys())
        
        self.buttons = {1: self.button_object_1,
                        # 2: self.button_object_2,
        }

        self.objects = [
            self.stove_object_1,
            self.bread_ingredient,
            self.pot_object,
            self.serving_region,
            self.button_object_1,
        ]
        
        self.model.merge_assets(self.button_object_1)
        self.model.merge_assets(self.stove_object_1)
        self.model.merge_assets(self.serving_region)

        # hardcode some z-offsets here
        self._hardcoded_z_offsets = {
            self.stove_object_1.name : 0.895,
            self.button_object_1.name : 0.895,
            self.serving_region.name : 0.878,
        }

    def visualize(self, vis_settings):
        """
        Update site visualization to make sure stove object site visualization is kept up to date.
        """
        super(Kitchen_D1, self).visualize(vis_settings)
        self._post_process()
