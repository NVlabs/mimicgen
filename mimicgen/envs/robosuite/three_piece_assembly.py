# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

from collections import OrderedDict
import random
import numpy as np

from robosuite.utils.mjcf_utils import CustomMaterial

import robosuite.utils.transform_utils as T

from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.mjcf_utils import CustomMaterial, find_elements, string_to_array

from mimicgen.models.robosuite.objects import BoxPatternObject
from mimicgen.envs.robosuite.single_arm_env_mg import SingleArmEnv_MG


class ThreePieceAssembly(SingleArmEnv_MG):
    """
    This class corresponds to the three piece assembly task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1., 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def _get_piece_densities(self):
        """
        Subclasses can override this method to change the weight of the pieces.
        """
        return dict(
            # base=100.,
            # NOTE: changed to make base piece heavier and task easier
            base=10000.,
            piece_1=100.,
            piece_2=100.,
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        The sparse reward only consists of the threading component.

        Note that the final reward is normalized and scaled by
        reward_scale / 2.0 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.

        # sparse completion reward
        if self._check_success():
            reward = 1.0

        # use a shaping reward
        if self.reward_shaping:
            pass

        if self.reward_scale is not None:
            reward *= self.reward_scale

        return reward

    def _get_piece_patterns(self):
        """
        Helper function to get unit-box patterns to make each assembly piece.
        """
        blocks = [[1,1,1],[1,0,1],[1,1,0],[1,0,0]]
        hole_side = [[0,0,0],[0,1,0],[0,0,1],[0,1,1]]
        hole = [[[0,0,0],[0,0,0],[0,0,0]]]

        # Pick out two sides of block1
        pick = random.randint(0,len(blocks)-1)
        pick = 0
        side1 = blocks[pick]
        hole1 = hole_side[pick]
        pick = random.randint(0,len(blocks)-1)
        pick = 0
        side2 = blocks[pick]
        hole2 = hole_side[pick]

        block1 = [[[1,1,1],[1,1,1],[1,1,1]],
                [[0,0,0],[0.9,.9,.9],[0,0,0]],
                [[0,0,0],[.9,.9,.9],[0,0,0]],]

        block1[0][0] = side1
        block1[0][2] = side2
        hole[0][0] = hole1
        hole[0][2] = hole2

        ### NOTE: we changed base_x from 7 to 5, and hole offset from 2 to 1, to make base piece smaller in size ###

        # Generate hole
        # base_x = 7
        base_x = 5
        base_z = 1
        base = np.ones((base_z, base_x, base_x))

        offset_x = random.randint(1, base_x - 4)
        # offset_x = 2
        offset_x = 1
        offset_y = random.randint(1, base_x - 3)
        # offset_y = 2
        offset_y = 1

        for z in range(len(hole)):
            for y in range(len(hole[0])):
                for x in range(len(hole[0][0])):
                    base[z][offset_y + y][offset_x + x] = hole[z][y][x]
        # base = np.rot90(base,random.randint(0,3),(1,2))

        block2 = [[[1,1,1],[0,0,0],[1,1,1]],
                [[1,1,1],[0,0,0],[1,1,1]],
                [[1,1,1],[1,1,1],[1,1,1]],
                [[0,0,0],[0,1,0],[0,0,0]]]

        return block1, block2, base

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # Add camera with full tabletop perspective
        self._add_agentview_full_camera(mujoco_arena)

        # Set default agentview camera to be "agentview_full" (and send old agentview camera to agentview_full)
        old_agentview_camera = find_elements(root=mujoco_arena.worldbody, tags="camera", attribs={"name": "agentview"}, return_first=True)
        old_agentview_camera_pose = (old_agentview_camera.get("pos"), old_agentview_camera.get("quat"))
        old_agentview_full_camera = find_elements(root=mujoco_arena.worldbody, tags="camera", attribs={"name": "agentview_full"}, return_first=True)
        old_agentview_full_camera_pose = (old_agentview_full_camera.get("pos"), old_agentview_full_camera.get("quat"))
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=string_to_array(old_agentview_full_camera_pose[0]),
            quat=string_to_array(old_agentview_full_camera_pose[1]),
        )
        mujoco_arena.set_camera(
            camera_name="agentview_full",
            pos=string_to_array(old_agentview_camera_pose[0]),
            quat=string_to_array(old_agentview_camera_pose[1]),
        )

        # initialize objects of interest
        self.piece_1_pattern, self.piece_2_pattern, self.base_pattern = self._get_piece_patterns()

        self.piece_1_size = 0.017
        self.piece_2_size = 0.02
        self.base_size = 0.019

        # Define materials we want to use for this object
        # tex_attrib = {
        #     "type": "cube",
        # }
        # mat_attrib = {
        #     "texrepeat": "1 1",
        #     "specular": "0.4",
        #     "shininess": "0.1",
        # }
        # mat = CustomMaterial(
        #     texture="WoodDark",
        #     tex_name="darkwood",
        #     mat_name="darkwood_mat",
        #     tex_attrib=tex_attrib,
        #     mat_attrib=mat_attrib,
        # )
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        mat = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        piece_densities = self._get_piece_densities()

        self.piece_1 = BoxPatternObject(
            name="piece_1",
            unit_size=[self.piece_1_size, self.piece_1_size, self.piece_1_size],
            pattern=self.piece_1_pattern,
            rgba=None,
            material=mat,
            density=piece_densities["piece_1"],
            friction=None,
        )
        self.piece_2 = BoxPatternObject(
            name="piece_2",
            unit_size=[self.piece_2_size, self.piece_2_size, self.piece_2_size],
            pattern=self.piece_2_pattern,
            rgba=None,
            material=mat,
            density=piece_densities["piece_2"],
            friction=None,
        )
        self.base = BoxPatternObject(
            name="base",
            unit_size=[self.base_size, self.base_size, self.base_size],
            pattern=self.base_pattern,
            rgba=None,
            material=mat,
            density=piece_densities["base"],
            friction=None,
        )

        objects = [self.base, self.piece_1, self.piece_2]

        # Create placement initializer
        self._get_placement_initializer()

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=objects,
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
            base=dict(
                x=(0., 0.),
                y=(0., 0.),
                z_rot=(0., 0.),
                reference=self.table_offset,
            ),
            piece_1=dict(
                x=(-0.22, 0.22),
                y=(-0.22, 0.22),
                z_rot=(1.5708, 1.5708),
                reference=self.table_offset,
            ),
            piece_2=dict(
                x=(-0.22, 0.22),
                y=(-0.22, 0.22),
                z_rot=(1.5708, 1.5708),
                reference=self.table_offset,
            ),
        )

    def _get_placement_initializer(self):
        bounds = self._get_initial_placement_bounds()

        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="BaseSampler",
                mujoco_objects=self.base,
                x_range=bounds["base"]["x"],
                y_range=bounds["base"]["y"],
                rotation=bounds["base"]["z_rot"],
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["base"]["reference"],
                # z_offset=0.,
                z_offset=0.001,
            )
        )
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="Piece1Sampler",
                mujoco_objects=self.piece_1,
                x_range=bounds["piece_1"]["x"],
                y_range=bounds["piece_1"]["y"],
                rotation=bounds["piece_1"]["z_rot"],
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["piece_1"]["reference"],
                # z_offset=0.,
                z_offset=0.001,
            )
        )
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="Piece2Sampler",
                mujoco_objects=self.piece_2,
                x_range=bounds["piece_2"]["x"],
                y_range=bounds["piece_2"]["y"],
                rotation=bounds["piece_2"]["z_rot"],
                rotation_axis='z',
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=bounds["piece_2"]["reference"],
                # z_offset=0.,
                z_offset=0.001,
            )
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.obj_body_id = dict(
            base=self.sim.model.body_name2id(self.base.root_body),
            piece_1=self.sim.model.body_name2id(self.piece_1.root_body),
            piece_2=self.sim.model.body_name2id(self.piece_2.root_body)
        )

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # for conversion to relative gripper frame
            @sensor(modality=modality)
            def world_pose_in_gripper(obs_cache):
                return T.pose_inv(T.pose2mat((obs_cache[f"{pf}eef_pos"], obs_cache[f"{pf}eef_quat"]))) if\
                    f"{pf}eef_pos" in obs_cache and f"{pf}eef_quat" in obs_cache else np.eye(4)
            sensors = [world_pose_in_gripper]
            names = ["world_pose_in_gripper"]
            actives = [False]

            @sensor(modality=modality)
            def eef_control_frame_pose(obs_cache):
                return T.make_pose(
                    np.array(self.sim.data.site_xpos[self.sim.model.site_name2id(self.robots[0].controller.eef_name)]),
                    np.array(self.sim.data.site_xmat[self.sim.model.site_name2id(self.robots[0].controller.eef_name)].reshape([3, 3])),
                ) if \
                    f"{pf}eef_pos" in obs_cache and f"{pf}eef_quat" in obs_cache else np.eye(4)
            sensors += [eef_control_frame_pose]
            names += ["eef_control_frame_pose"]
            actives += [False]

            # add ground-truth poses (absolute and relative to eef) for all objects
            for obj_name in self.obj_body_id:
                obj_sensors, obj_sensor_names = self._create_obj_sensors(obj_name=obj_name, modality=modality)
                sensors += obj_sensors
                names += obj_sensor_names
                actives += [True] * len(obj_sensors)

            obj_centric_sensors, obj_centric_sensor_names = self._create_obj_centric_sensors(modality="object_centric")
            sensors += obj_centric_sensors
            names += obj_centric_sensor_names
            actives += [True] * len(obj_centric_sensors)

            # Create observables
            for name, s, active in zip(names, sensors, actives):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                    active=active,
                )

        return observables

    def _create_obj_sensors(self, obj_name, modality="object"):
        """
        Helper function to create sensors for a given object. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.

        Args:
            obj_name (str): Name of object to create sensors for
            modality (str): Modality to assign to all sensors

        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given obj
                names (list): array of corresponding observable names
        """

        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality=modality)
        def obj_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.obj_body_id[obj_name]])

        @sensor(modality=modality)
        def obj_quat(obs_cache):
            return T.convert_quat(self.sim.data.body_xquat[self.obj_body_id[obj_name]], to="xyzw")

        @sensor(modality=modality)
        def obj_to_eef_pos(obs_cache):
            # Immediately return default value if cache is empty
            if any([name not in obs_cache for name in
                    [f"{obj_name}_pos", f"{obj_name}_quat", "world_pose_in_gripper"]]):
                return np.zeros(3)
            obj_pose = T.pose2mat((obs_cache[f"{obj_name}_pos"], obs_cache[f"{obj_name}_quat"]))
            rel_pose = T.pose_in_A_to_pose_in_B(obj_pose, obs_cache["world_pose_in_gripper"])
            rel_pos, rel_quat = T.mat2pose(rel_pose)
            obs_cache[f"{obj_name}_to_{pf}eef_quat"] = rel_quat
            obs_cache[f"{obj_name}_pose"] = obj_pose
            return rel_pos

        @sensor(modality=modality)
        def obj_to_eef_quat(obs_cache):
            return obs_cache[f"{obj_name}_to_{pf}eef_quat"] if \
                f"{obj_name}_to_{pf}eef_quat" in obs_cache else np.zeros(4)

        sensors = [obj_pos, obj_quat, obj_to_eef_pos, obj_to_eef_quat]
        names = [f"{obj_name}_pos", f"{obj_name}_quat", f"{obj_name}_to_{pf}eef_pos", f"{obj_name}_to_{pf}eef_quat"]

        return sensors, names

    def _create_obj_centric_sensors(self, modality="object_centric"):
        """
        Creates sensors for poses relative to certain objects. This is abstracted in a separate 
        function call so that we don't have local function naming collisions during 
        the _setup_observables() call.

        Args:
            modality (str): Modality to assign to all sensors

        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given obj
                names (list): array of corresponding observable names
        """
        sensors = []
        names = []
        pf = self.robots[0].robot_model.naming_prefix

        # helper function for relative position sensors, to avoid code duplication
        def _pos_helper(obs_cache, obs_name, ref_name, quat_cache_name):
            # Immediately return default value if cache is empty
            if any([name not in obs_cache for name in
                    [obs_name, ref_name]]):
                return np.zeros(3)
            ref_pose = obs_cache[ref_name]
            obs_pose = obs_cache[obs_name]
            rel_pose = T.pose_in_A_to_pose_in_B(obs_pose, T.pose_inv(ref_pose))
            rel_pos, rel_quat = T.mat2pose(rel_pose)
            obs_cache[quat_cache_name] = rel_quat
            return rel_pos

        # helper function for relative quaternion sensors, to avoid code duplication
        def _quat_helper(obs_cache, quat_cache_name):
            return obs_cache[quat_cache_name] if \
                quat_cache_name in obs_cache else np.zeros(4)

        # eef pose relative to base
        @sensor(modality=modality)
        def eef_pos_rel_base(obs_cache):
            return _pos_helper(
                obs_cache=obs_cache,
                obs_name="eef_control_frame_pose",
                ref_name="base_pose",
                quat_cache_name="eef_quat_rel_base",
            )
        @sensor(modality=modality)
        def eef_quat_rel_base(obs_cache):
            return _quat_helper(
                obs_cache=obs_cache,
                quat_cache_name="eef_quat_rel_base",
            )

        # eef pose relative to piece 1
        @sensor(modality=modality)
        def eef_pos_rel_piece_1(obs_cache):
            return _pos_helper(
                obs_cache=obs_cache,
                obs_name="eef_control_frame_pose",
                ref_name="piece_1_pose",
                quat_cache_name="eef_quat_rel_piece_1",
            )
        @sensor(modality=modality)
        def eef_quat_rel_piece_1(obs_cache):
            return _quat_helper(
                obs_cache=obs_cache,
                quat_cache_name="eef_quat_rel_piece_1",
            )

        # eef pose relative to piece 2
        @sensor(modality=modality)
        def eef_pos_rel_piece_2(obs_cache):
            return _pos_helper(
                obs_cache=obs_cache,
                obs_name="eef_control_frame_pose",
                ref_name="piece_2_pose",
                quat_cache_name="eef_quat_rel_piece_2",
            )
        @sensor(modality=modality)
        def eef_quat_rel_piece_2(obs_cache):
            return _quat_helper(
                obs_cache=obs_cache,
                quat_cache_name="eef_quat_rel_piece_2",
            )

        sensors += [eef_pos_rel_base, eef_quat_rel_base, eef_pos_rel_piece_1, eef_quat_rel_piece_1, eef_pos_rel_piece_2, eef_quat_rel_piece_2]
        names += [f"{pf}eef_pos_rel_base", f"{pf}eef_quat_rel_base", f"{pf}eef_pos_rel_piece_1", f"{pf}eef_quat_rel_piece_1", f"{pf}eef_pos_rel_piece_2", f"{pf}eef_quat_rel_piece_2"]

        # piece 1 pose relative to base
        @sensor(modality=modality)
        def piece_1_pos_rel_base(obs_cache):
            return _pos_helper(
                obs_cache=obs_cache,
                obs_name="piece_1_pose",
                ref_name="base_pose",
                quat_cache_name="piece_1_quat_rel_base",
            )
        @sensor(modality=modality)
        def piece_1_quat_rel_base(obs_cache):
            return _quat_helper(
                obs_cache=obs_cache,
                quat_cache_name="piece_1_quat_rel_base",
            )

        # piece 2 pose relative to piece 1
        @sensor(modality=modality)
        def piece_2_pos_rel_piece_1(obs_cache):
            return _pos_helper(
                obs_cache=obs_cache,
                obs_name="piece_2_pose",
                ref_name="piece_1_pose",
                quat_cache_name="piece_2_quat_rel_piece_1",
            )
        @sensor(modality=modality)
        def piece_2_quat_rel_piece_1(obs_cache):
            return _quat_helper(
                obs_cache=obs_cache,
                quat_cache_name="piece_2_quat_rel_piece_1",
            )

        sensors += [piece_1_pos_rel_base, piece_1_quat_rel_base, piece_2_pos_rel_piece_1, piece_2_quat_rel_piece_1]
        names += ["piece_1_pos_rel_base", "piece_1_quat_rel_base", "piece_2_pos_rel_piece_1", "piece_2_quat_rel_piece_1"]

        return sensors, names

    def _check_success(self):
        """
        Check if task is complete.
        """
        metrics = self._get_partial_task_metrics()
        return metrics["task"]

    def _check_first_piece_is_assembled(self, xy_thresh=0.02):
        robot_and_piece_1_in_contact = self._check_grasp(
            gripper=self.robots[0].gripper,
            object_geoms=[g for g in self.piece_1.contact_geoms]
        )

        piece_1_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["piece_1"]])
        base_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["base"]])

        # assume that first piece is assembled when x-y position is close enough to base piece
        # and gripper is not holding the piece
        first_piece_is_assembled = (np.linalg.norm(piece_1_pos[:2] - base_pos[:2]) < xy_thresh) and (not robot_and_piece_1_in_contact)
        return first_piece_is_assembled

    def _check_second_piece_is_assembled(self, xy_thresh=0.02, z_thresh=0.02):
        robot_and_piece_2_in_contact = self._check_grasp(
            gripper=self.robots[0].gripper,
            object_geoms=[g for g in self.piece_2.contact_geoms]
        )

        piece_1_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["piece_1"]])
        piece_2_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["piece_2"]])
        base_pos = np.array(self.sim.data.body_xpos[self.obj_body_id["base"]])
        z_correct = base_pos[2] + self.piece_2_size * 4

        first_piece_is_assembled = self._check_first_piece_is_assembled(xy_thresh=xy_thresh)

        # second piece is assembled (and task is complete) when it is close enough to first piece in x-y, close
        # enough to first piece in z (and first piece is assembled) and gripper is not holding the piece
        second_piece_is_assembled = first_piece_is_assembled and (np.linalg.norm(piece_1_pos[:2] - piece_2_pos[:2]) < xy_thresh) and \
            (np.abs(piece_2_pos[2] - z_correct) < z_thresh) and (not robot_and_piece_2_in_contact)
        return second_piece_is_assembled

    def _get_partial_task_metrics(self):
        """
        Check if all three pieces have been assembled together.
        """
        metrics = {
            "first_piece_assembled": self._check_first_piece_is_assembled(),
            "task": self._check_second_piece_is_assembled(),
        }

        return metrics

        # if (np.linalg.norm(piece_1_pos[:2] - base_pos[:2]) < xy_thresh) and (np.linalg.norm(piece_1_pos[:2] - piece_2_pos[:2]) < xy_thresh) \
        #     and (np.abs(piece_2_pos[2] - z_correct) < z_thresh):
        #     return True
        # return False

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the needle.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.piece_1)


class ThreePieceAssembly_D0(ThreePieceAssembly):
    """Rename base class for convenience."""
    pass


class ThreePieceAssembly_D1(ThreePieceAssembly_D0):
    """
    All pieces still have fixed z-rotation, as in original task, but base can
    be placed anywhere.
    """
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
            base=dict(
                x=(-0.22, 0.22),
                y=(-0.22, 0.22),
                z_rot=(0., 0.),
                reference=self.table_offset,
            ),
            piece_1=dict(
                x=(-0.22, 0.22),
                y=(-0.22, 0.22),
                z_rot=(1.5708, 1.5708),
                reference=self.table_offset,
            ),
            piece_2=dict(
                x=(-0.22, 0.22),
                y=(-0.22, 0.22),
                z_rot=(1.5708, 1.5708),
                reference=self.table_offset,
            ),
        )


class ThreePieceAssembly_D2(ThreePieceAssembly_D1):
    """
    All pieces can move anywhere on table and in z-rotation.
    """
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
            base=dict(
                x=(-0.22, 0.22),
                y=(-0.22, 0.22),
                z_rot=(-np.pi / 4., np.pi / 4.),
                reference=self.table_offset,
            ),
            piece_1=dict(
                x=(-0.22, 0.22),
                y=(-0.22, 0.22),
                z_rot=(1.5708 - np.pi / 2., 1.5708 + np.pi / 2.),
                reference=self.table_offset,
            ),
            piece_2=dict(
                x=(-0.22, 0.22),
                y=(-0.22, 0.22),
                z_rot=(1.5708 - np.pi / 2., 1.5708 + np.pi / 2.),
                reference=self.table_offset,
            ),
        )
