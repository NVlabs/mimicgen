# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Task configs for robosuite.

See @Coffee_Config below for an explanation of each parameter.
"""
import mimicgen
from mimicgen.configs.config import MG_Config


class Coffee_Config(MG_Config):
    """
    Corresponds to robosuite Coffee task and variants.
    """
    NAME = "coffee"
    TYPE = "robosuite"

    def task_config(self):
        """
        This function populates the `config.task` attribute of the config, 
        which has settings for each object-centric subtask in a task. Each 
        dictionary should have kwargs for the @add_subtask method in the 
        @MG_TaskSpec object.
        """
        self.task.task_spec.subtask_1 = dict(
            # Each subtask involves manipulation with respect to a single object frame. 
            # This string should specify the object for this subtask. The name should be 
            # consistent with the "datagen_info" from the environment interface and dataset.
            object_ref="coffee_pod",
            # The "datagen_info" from the environment and dataset includes binary indicators 
            # for each subtask of the task at each timestep. This key should correspond
            # to the key in "datagen_info" that should be used to infer when this subtask 
            # is finished (e.g. on a 0 to 1 edge of the binary indicator). Should provide 
            # None for the final subtask.
            subtask_term_signal="grasp",
            # if not None, specifies time offsets to be used during data generation when splitting 
            # a trajectory into subtask segments. On each data generation attempt, an offset is sampled
            # and added to the boundary defined by @subtask_term_signal.
            subtask_term_offset_range=(5, 10),
            # specifies how the source subtask segment should be selected during data generation 
            # from the set of source human demos
            selection_strategy="random",
            # optional keyword arguments for the selection strategy function used
            selection_strategy_kwargs=None,
            # amount of action noise to apply during this subtask
            action_noise=0.05,
            # number of interpolation steps to bridge previous subtask segment to this one
            num_interpolation_steps=5,
            # number of additional steps (with constant target pose of beginning of this subtask segment) to 
            # add to give the robot time to reach the pose needed to carry out this subtask segment
            num_fixed_steps=0,
            # if True, apply action noise during interpolation phase leading up to this subtask, as 
            # well as during the execution of this subtask
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.subtask_2 = dict(
            object_ref="coffee_machine", 
            # end of final subtask does not need to be detected
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # allow downstream code to completely replace the task spec from an external config
        self.task.task_spec.do_not_lock_keys()


class Threading_Config(MG_Config):
    """
    Corresponds to robosuite Threading task and variants.
    """
    NAME = "threading"
    TYPE = "robosuite"

    def task_config(self):
        """
        This function populates the `config.task` attribute of the config, 
        which has settings for each object-centric subtask in a task. Each 
        dictionary should have kwargs for the @add_subtask method in the 
        @MG_TaskSpec object.
        """
        self.task.task_spec.subtask_1 = dict(
            object_ref="needle", 
            subtask_term_signal="grasp",
            subtask_term_offset_range=(5, 10),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.subtask_2 = dict(
            object_ref="tripod", 
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.do_not_lock_keys()


class ThreePieceAssembly_Config(MG_Config):
    """
    Corresponds to robosuite ThreePieceAssembly task and variants.
    """
    NAME = "three_piece_assembly"
    TYPE = "robosuite"

    def task_config(self):
        """
        This function populates the `config.task` attribute of the config, 
        which has settings for each object-centric subtask in a task. Each 
        dictionary should have kwargs for the @add_subtask method in the 
        @MG_TaskSpec object.
        """
        self.task.task_spec.subtask_1 = dict(
            object_ref="piece_1", 
            subtask_term_signal="grasp_1",
            subtask_term_offset_range=(5, 10),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.subtask_2 = dict(
            object_ref="base", 
            subtask_term_signal="insert_1",
            subtask_term_offset_range=(5, 10),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.subtask_3 = dict(
            object_ref="piece_2", 
            subtask_term_signal="grasp_2",
            subtask_term_offset_range=(5, 10),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.subtask_4 = dict(
            object_ref="piece_1", 
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.do_not_lock_keys()


class Square_Config(MG_Config):
    """
    Corresponds to robosuite Square task and variants.
    """
    NAME = "square"
    TYPE = "robosuite"

    def task_config(self):
        """
        This function populates the `config.task` attribute of the config, 
        which has settings for each object-centric subtask in a task. Each 
        dictionary should have kwargs for the @add_subtask method in the 
        @MG_TaskSpec object.
        """
        self.task.task_spec.subtask_1 = dict(
            object_ref="square_nut", 
            subtask_term_signal="grasp",
            subtask_term_offset_range=(10, 20),
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs=dict(nn_k=3),
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.subtask_2 = dict(
            object_ref="square_peg", 
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.do_not_lock_keys()


class Stack_Config(MG_Config):
    """
    Corresponds to robosuite Stack task and variants.
    """
    NAME = "stack"
    TYPE = "robosuite"

    def task_config(self):
        """
        This function populates the `config.task` attribute of the config, 
        which has settings for each object-centric subtask in a task. Each 
        dictionary should have kwargs for the @add_subtask method in the 
        @MG_TaskSpec object.
        """
        self.task.task_spec.subtask_1 = dict(
            object_ref="cubeA", 
            subtask_term_signal="grasp",
            subtask_term_offset_range=(10, 20),
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs=dict(nn_k=3),
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.subtask_2 = dict(
            object_ref="cubeB", 
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs=dict(nn_k=3),
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.do_not_lock_keys()


class StackThree_Config(MG_Config):
    """
    Corresponds to robosuite StackThree task and variants.
    """
    NAME = "stack_three"
    TYPE = "robosuite"

    def task_config(self):
        """
        This function populates the `config.task` attribute of the config, 
        which has settings for each object-centric subtask in a task. Each 
        dictionary should have kwargs for the @add_subtask method in the 
        @MG_TaskSpec object.
        """
        self.task.task_spec.subtask_1 = dict(
            object_ref="cubeA", 
            subtask_term_signal="grasp_1",
            subtask_term_offset_range=(10, 20),
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs=dict(nn_k=3),
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.subtask_2 = dict(
            object_ref="cubeB", 
            subtask_term_signal="stack_1",
            subtask_term_offset_range=(10, 20),
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs=dict(nn_k=3),
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.subtask_3 = dict(
            object_ref="cubeC", 
            subtask_term_signal="grasp_2",
            subtask_term_offset_range=(10, 20),
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs=dict(nn_k=3),
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.subtask_4 = dict(
            object_ref="cubeA", 
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs=dict(nn_k=3),
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.do_not_lock_keys()


class HammerCleanup_Config(MG_Config):
    """
    Corresponds to robosuite HammerCleanup task and variants.
    """
    NAME = "hammer_cleanup"
    TYPE = "robosuite"

    def task_config(self):
        """
        This function populates the `config.task` attribute of the config, 
        which has settings for each object-centric subtask in a task. Each 
        dictionary should have kwargs for the @add_subtask method in the 
        @MG_TaskSpec object.
        """
        self.task.task_spec.subtask_1 = dict(
            object_ref="drawer", 
            subtask_term_signal="open",
            subtask_term_offset_range=(10, 20),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.subtask_2 = dict(
            object_ref="hammer", 
            subtask_term_signal="grasp",
            subtask_term_offset_range=(10, 20),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.subtask_3 = dict(
            object_ref="drawer", 
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.do_not_lock_keys()


class MugCleanup_Config(MG_Config):
    """
    Corresponds to robosuite MugCleanup task and variants.
    """
    NAME = "mug_cleanup"
    TYPE = "robosuite"

    def task_config(self):
        """
        This function populates the `config.task` attribute of the config, 
        which has settings for each object-centric subtask in a task. Each 
        dictionary should have kwargs for the @add_subtask method in the 
        @MG_TaskSpec object.
        """
        self.task.task_spec.subtask_1 = dict(
            object_ref="drawer", 
            subtask_term_signal="open",
            subtask_term_offset_range=(10, 20),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.subtask_2 = dict(
            object_ref="object", 
            subtask_term_signal="grasp",
            subtask_term_offset_range=(10, 20),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.subtask_3 = dict(
            object_ref="drawer", 
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.do_not_lock_keys()


class NutAssembly_Config(MG_Config):
    """
    Corresponds to robosuite NutAssembly task and variants.
    """
    NAME = "nut_assembly"
    TYPE = "robosuite"

    def task_config(self):
        """
        This function populates the `config.task` attribute of the config, 
        which has settings for each object-centric subtask in a task. Each 
        dictionary should have kwargs for the @add_subtask method in the 
        @MG_TaskSpec object.
        """
        self.task.task_spec.subtask_1 = dict(
            object_ref="square_nut", 
            subtask_term_signal="grasp_square_nut",
            subtask_term_offset_range=(10, 20),
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs=dict(nn_k=3),
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.subtask_2 = dict(
            object_ref="square_peg", 
            subtask_term_signal="insert_square_nut",
            subtask_term_offset_range=(10, 20),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.subtask_3 = dict(
            object_ref="round_nut", 
            subtask_term_signal="grasp_round_nut",
            subtask_term_offset_range=(10, 20),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.subtask_4 = dict(
            object_ref="round_peg", 
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.do_not_lock_keys()


class PickPlace_Config(MG_Config):
    """
    Corresponds to robosuite PickPlace task and variants.
    """
    NAME = "pick_place"
    TYPE = "robosuite"

    def task_config(self):
        """
        This function populates the `config.task` attribute of the config, 
        which has settings for each object-centric subtask in a task. Each 
        dictionary should have kwargs for the @add_subtask method in the 
        @MG_TaskSpec object.
        """
        for i, obj in enumerate(["milk", "cereal", "bread", "can"]):
            self.task.task_spec["subtask_{}".format(2 * i + 1)] = dict(
                object_ref=obj, 
                subtask_term_signal="grasp_{}".format(obj),
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs=dict(nn_k=3),
                action_noise=0.05,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
            # last subtask does not need subtask termination signal but all others do
            self.task.task_spec["subtask_{}".format(2 * i + 2)] = dict(
                object_ref=None, 
                subtask_term_signal=None if (obj == "can") else "place_{}".format(obj),
                subtask_term_offset_range=None,
                selection_strategy="random",
                selection_strategy_kwargs=None,
                action_noise=0.05,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        self.task.task_spec.do_not_lock_keys()


class Kitchen_Config(MG_Config):
    """
    Corresponds to robosuite Kitchen task and variants.
    """
    NAME = "kitchen"
    TYPE = "robosuite"

    def task_config(self):
        """
        This function populates the `config.task` attribute of the config, 
        which has settings for each object-centric subtask in a task. Each 
        dictionary should have kwargs for the @add_subtask method in the 
        @MG_TaskSpec object.
        """
        self.task.task_spec.subtask_1 = dict(
            object_ref="button", 
            subtask_term_signal="stove_on",
            subtask_term_offset_range=(10, 20),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.subtask_2 = dict(
            object_ref="pot", 
            subtask_term_signal="grasp_pot",
            subtask_term_offset_range=(10, 20),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.subtask_3 = dict(
            object_ref="stove", 
            subtask_term_signal="place_pot_on_stove",
            subtask_term_offset_range=(10, 20),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.subtask_4 = dict(
            object_ref="bread", 
            subtask_term_signal="grasp_bread",
            subtask_term_offset_range=(10, 20),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.subtask_5 = dict(
            object_ref="pot", 
            subtask_term_signal="place_bread_in_pot",
            subtask_term_offset_range=(10, 20),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.subtask_6 = dict(
            object_ref="serving_region", 
            subtask_term_signal="serve",
            subtask_term_offset_range=(10, 20),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.subtask_7 = dict(
            object_ref="button", 
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.do_not_lock_keys()


class CoffeePreparation_Config(MG_Config):
    """
    Corresponds to robosuite CoffeePreparation task and variants.
    """
    NAME = "coffee_preparation"
    TYPE = "robosuite"

    def task_config(self):
        """
        This function populates the `config.task` attribute of the config, 
        which has settings for each object-centric subtask in a task. Each 
        dictionary should have kwargs for the @add_subtask method in the 
        @MG_TaskSpec object.
        """
        self.task.task_spec.subtask_1 = dict(
            object_ref="mug", 
            subtask_term_signal="mug_grasp",
            subtask_term_offset_range=(5, 10),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.subtask_2 = dict(
            object_ref="coffee_machine", 
            subtask_term_signal="mug_place",
            subtask_term_offset_range=(5, 10),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.subtask_3 = dict(
            object_ref="drawer", 
            subtask_term_signal="drawer_open",
            subtask_term_offset_range=(5, 10),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.subtask_4 = dict(
            object_ref="coffee_pod", 
            subtask_term_signal="pod_grasp",
            subtask_term_offset_range=(5, 10),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.subtask_5 = dict(
            object_ref="coffee_machine", 
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.do_not_lock_keys()
