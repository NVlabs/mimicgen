# Task Spec

The sequence of object-centric subtasks and other important data generation settings for each data generation run are communicated to MimicGen through the TaskSpec object (`configs/task_spec.py`). The TaskSpec object is instantiated from the MimicGen task config (`config.task.task_spec`) using the `TaskSpec.from_json` method.

<div class="admonition note">
<p class="admonition-title">Note</p>

See [this section](https://mimicgen.github.io/docs/tutorials/datagen_custom.html#step-2-implement-task-specific-config) of the Data Generation for Custom Environments tutorial for an example of how to implement a task config for a new task.

</div>

We describe components of the TaskSpec object in more detail below. The TaskSpec is essentially list of dictionaries with each dictionary corresponding to exactly one subtask. The method below highlights the important settings that the TaskSpec object holds for each task:

```python
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
```

- The `object_ref` for each subtask determines the reference object frame for the motion in that subtask. The name here should be consistent with the  `get_object_poses` method of the relevant task-specific [Environment Interface](https://mimicgen.github.io/docs/modules/env_interfaces.html)) object.

- The `subtask_term_signal` and `subtask_term_offset_range` settings determine how subtask segments for this subtask is parsed from the source demonstrations -- see the [Subtask Termination Signals](https://mimicgen.github.io/docs/tutorials/subtask_termination_signals.html) page for more information.

- The `selection_strategy` and `selection_strategy_kwargs` determine the [SelectionStrategy](https://mimicgen.github.io/docs/modules/datagen.html#selection-strategy) class used to select a source subtask segment at the start of each subtask during data generation. See Appendix N.3 in the MimicGen paper for more details.

- The `action_noise` setting determines the magnitude of action noise added when executing actions during data generation. See Appendix N.4 of the MimicGen paper for more details.

- The `num_interpolation_steps` and `num_fixed_steps` settings determines the number of interpolation waypoints in the interpolation segment that bridges this subtask segment and the previous subtask segment during data generation. See Appendix N.2 of the MimicGen paper for more details.
