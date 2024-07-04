# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Defines task specification objects, which are used to store task-specific settings
for data generation.
"""
import json

import mimicgen
from mimicgen.datagen.selection_strategy import assert_selection_strategy_exists

class MG_TaskSpec:
    """
    Stores task-specific settings for data generation. Each task is a sequence of
    object-centric subtasks, and each subtask stores relevant settings used during
    the data generation process.
    """
    def __init__(self):
        self.spec = []

    def add_subtask(
        self, 
        object_ref,
        subtask_term_signal,
        subtask_term_offset_range=None,
        selection_strategy="random",
        selection_strategy_kwargs=None,
        action_noise=0.,
        num_interpolation_steps=5,
        num_fixed_steps=0,
        apply_noise_during_interpolation=False,
    ):
        """
        Add subtask to this task spec.

        Args:
            object_ref (str): each subtask involves manipulation with 
                respect to a single object frame. This string should
                specify the object for this subtask. The name
                should be consistent with the "datagen_info" from the
                environment interface and dataset.

            subtask_term_signal (str or None): the "datagen_info" from the environment
                and dataset includes binary indicators for each subtask
                of the task at each timestep. This key should correspond
                to the key in "datagen_info" that should be used to
                infer when this subtask is finished (e.g. on a 0 to 1
                edge of the binary indicator). Should provide None for the final 
                subtask.

            subtask_term_offset_range (2-tuple): if provided, specifies time offsets to 
                be used during data generation when splitting a trajectory into 
                subtask segments. On each data generation attempt, an offset is sampled
                and added to the boundary defined by @subtask_term_signal.

            selection_strategy (str): specifies how the source subtask segment should be
                selected during data generation from the set of source human demos

            selection_strategy_kwargs (dict or None): optional keyword arguments for the selection
                strategy function used

            action_noise (float): amount of action noise to apply during this subtask

            num_interpolation_steps (int): number of interpolation steps to bridge previous subtask segment 
                to this one

            num_fixed_steps (int): number of additional steps (with constant target pose of beginning of 
                this subtask segment) to add to give the robot time to reach the pose needed to carry 
                out this subtask segment

            apply_noise_during_interpolation (bool): if True, apply action noise during interpolation phase 
                leading up to this subtask, as well as during the execution of this subtask
        """
        if subtask_term_offset_range is None:
            # corresponds to no offset
            subtask_term_offset_range = (0, 0)
        assert isinstance(subtask_term_offset_range, tuple)
        assert len(subtask_term_offset_range) == 2
        assert subtask_term_offset_range[0] <= subtask_term_offset_range[1]
        assert_selection_strategy_exists(selection_strategy)
        self.spec.append(dict(
            object_ref=object_ref,
            subtask_term_signal=subtask_term_signal,
            subtask_term_offset_range=subtask_term_offset_range,
            selection_strategy=selection_strategy,
            selection_strategy_kwargs=selection_strategy_kwargs,
            action_noise=action_noise,
            num_interpolation_steps=num_interpolation_steps,
            num_fixed_steps=num_fixed_steps,
            apply_noise_during_interpolation=apply_noise_during_interpolation,
        ))

    @classmethod
    def from_json(cls, json_string=None, json_dict=None):
        """
        Instantiate a TaskSpec object from a json string. This should
        be consistent with the output of @serialize.

        Args:
            json_string (str): top-level of json has a key per subtask in-order (e.g.
                "subtask_1", "subtask_2", "subtask_3") and under each subtask, there should
                be an entry for each argument of @add_subtask

            json_dict (dict): optionally directly pass json dict
        """
        if json_dict is None:
            json_dict = json.loads(json_string)
        task_spec = cls()
        for subtask_name in json_dict:
            if json_dict[subtask_name]["subtask_term_offset_range"] is not None:
                json_dict[subtask_name]["subtask_term_offset_range"] = tuple(json_dict[subtask_name]["subtask_term_offset_range"])  
            task_spec.add_subtask(**json_dict[subtask_name])
        return task_spec

    def serialize(self):
        """
        Return a json string corresponding to this task spec object. Compatible with
        @from_json classmethod.
        """
        json_dict = dict()
        for i, elem in enumerate(self.spec):
            json_dict["subtask_{}".format(i + 1)] = elem
        return json.dumps(json_dict, indent=4)

    def __len__(self):
        return len(self.spec)

    def __getitem__(self, ind):
        """Support list-like indexing"""
        return self.spec[ind]

    def __iter__(self):
        """Support list-like iteration."""
        return iter(self.spec)

    def __repr__(self):
        return json.dumps(self.spec, indent=4)
