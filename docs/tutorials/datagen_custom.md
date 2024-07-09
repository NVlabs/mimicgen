# Data Generation for Custom Environments

In this section, we provide guidance on using MimicGen to generate data for custom tasks and simulation frameworks.

<div class="admonition note">
<p class="admonition-title">Note</p>

We recommend going through the [Getting Started](https://mimicgen.github.io/docs/tutorials/getting_started.html) tutorial first, so that you are familiar with the typical data generation workflow. We will refer back to the steps in the [Data Generation Workflow Overview](https://mimicgen.github.io/docs/tutorials/getting_started.html#overview-of-typical-data-generation-workflow) in this section.

</div>

## Generating Data for New Tasks

In this section, we will assume we are trying to generate data for a new task implemented in [robosuite](https://robosuite.ai/), and we will use the robosuite Stack Three task as a running example. The same instructions can be used for any task in any simulation framework, as long as an [Environment Interface](https://mimicgen.github.io/docs/modules/env_interfaces.html) base class already exists for the simulation framework. See the [Generating Data for New Simulation Frameworks](https://mimicgen.github.io/docs/tutorials/datagen_custom.html#generating-data-for-new-simulation-frameworks)) below for guidance on setting up a new simulation framework if this has not happened yet.

### Step 1: Implement Task-Specific Environment Interface

The first step is to subclass the appropriate base Environment Interface class -- for robosuite, this is the `RobosuiteInterface` class at the top of `env_interfaces/robosuite.py`. We create a new class as below:

```python
class MG_StackThree(RobosuiteInterface):
    """
    Corresponds to robosuite StackThree task and variants.
    """
    pass
```

The `MG_EnvInterface` abstract base class in `env_interfaces/base.py` (which `RobosuiteInterface` inherits from) describes the methods that task-specific subclasses must implement. There are two important methods:

```python
    """
    These should be filled out by each simulation domain (e.g. nut assembly, coffee).
    """
    @abc.abstractmethod
    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_subtask_term_signals(self):
        """
        Gets a dictionary of binary flags for each subtask in a task. The flag is 1
        when the subtask has been completed and 0 otherwise. MimicGen only uses this
        when parsing source demonstrations at the start of data generation, and it only
        uses the first 0 -> 1 transition in this signal to detect the end of a subtask.

        Returns:
            subtask_term_signals (dict): dictionary that maps subtask name to termination flag (0 or 1)
        """
        raise NotImplementedError
```

Recall that MimicGen generates data by composing object-centric subtask segments together (see the paper for more information). During data generation, MimicGen requires a way to observe the pose of the relevant object at the start of each subtask. The `get_object_poses` method will be used for this purpose - it should return a dictionary mapping object name to a pose matrix. 

The `RobosuiteInterface` base class offers a helper method `get_object_pose(self, obj_name, obj_type)` to make retrieving object poses from robosuite easily - we use it below to get the poses of each cube in the `StackThree` task.

```python
    def get_object_poses(self):
        """
        Gets the pose of each object relevant to MimicGen data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 np.array)
        """

        # three relevant objects - three cubes
        return dict(
            cubeA=self.get_object_pose(obj_name=self.env.cubeA.root_body, obj_type="body"),
            cubeB=self.get_object_pose(obj_name=self.env.cubeB.root_body, obj_type="body"),
            cubeC=self.get_object_pose(obj_name=self.env.cubeC.root_body, obj_type="body"),
        )
```

Next we need to implement `get_subtask_term_signals`. This function has only one purpose - it is used to provide [Subtask Termination Signals](https://mimicgen.github.io/docs/tutorials/subtask_termination_signals.html) for each timestep in the source demonstrations (this is part of what happens in `scripts/prepare_src_dataset.py`). These signals are used to determine where each subtask ends and the next one starts -- the first 0 to 1 transition in this signal during a source demonstration determines the end of the subtask.

The StackThree tasks consists of 4 object-centric subtasks:

```
1. grasping cubeA (motion relative to cubeA)
2. placing cubeA on cubeB (motion relative to cubeB)
3. grasping cubeC (motion relative to cubeC)
4. placing cubeC on cubeA (motion relative to cubeA)
```

To define the end of subtask 1 and 3, we can just check for a successful grasp, and for the end of subtask 2, we can check for a placement (re-using part of the success check for the robosuite StackThree task):

```python
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

        # first subtask is grasping cubeA (motion relative to cubeA)
        signals["grasp_1"] = int(self.env._check_grasp(gripper=self.env.robots[0].gripper, object_geoms=self.env.cubeA))

        # second subtask is placing cubeA on cubeB (motion relative to cubeB)
        signals["stack_1"] = int(self.env._check_cubeA_stacked())

        # third subtask is grasping cubeC (motion relative to cubeC)
        signals["grasp_2"] = int(self.env._check_grasp(gripper=self.env.robots[0].gripper, object_geoms=self.env.cubeC))

        # final subtask is placing cubeC on cubeA (motion relative to cubeA) - but final subtask signal is not needed
        return signals
```

<div class="admonition warning">
<p class="admonition-title">Warning</p>

The final subtask in a task never requires a subtask termination signal, since its end is determined by the end of the source demonstration.

</div>

<div class="admonition note">
<p class="admonition-title">Note</p>

Providing a proper implementation for the `get_subtask_term_signals` function is entirely optional. In most cases it is easy to specify heuristics to define these subtask boundaries as we did above, but sometimes you may want to just directly annotate the boundaries between subtasks. We provide an annotation script (`scripts/annotate_subtasks.py`) for this purpose. If you plan to do this, you can just return an empty dict for the `get_subtask_term_signals` function.

</div>

### Step 2: Implement Task-Specific Config

The next step is to implement a task-specific Config object that inherits from the `MG_Config` base class (`configs/config.py`). There are three things for the subclass to implement:

```python
    @property
    @classmethod
    def NAME(cls):
        # must be specified by subclasses
        raise NotImplementedError

    @property
    @classmethod
    def TYPE(cls):
        # must be specified by subclasses
        raise NotImplementedError

    def task_config(self):
        """
        This function populates the `config.task` attribute of the config, 
        which has settings for each object-centric subtask in a task.
        """
        raise NotImplementedError
```

The `NAME` and `TYPE` are used to store the new subclass into the config registry, and chiefly determine where the auto-generated config template is stored in the repository (e.g. `mimicgen/exps/templates/<TYPE>/<NAME>.json`).

The `task_config` is consistent with [TaskSpec objects](https://mimicgen.github.io/docs/modules/task_spec.html) -- there is an entry for each subtask which consists of the arguments to the `add_subtask` function in the TaskSpec object (`configs/task_spec.py`):

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
```

We show the implementation for the StackThree config below:

```python
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
```

Notice that we set the `object_ref` for each subtask to be consistent with the object names in the `get_object_poses` method in the `MG_StackThree` environment interface we implemented. We also set the `subtask_term_signal` for each subtask to be consistent with the subtask signals in the `get_subtask_term_signals` method in the `MG_StackThree` class as well. Please see the [TaskSpec page](https://mimicgen.github.io/docs/modules/task_spec.html) for more information on the other settings.

<div class="admonition note">
<p class="admonition-title">Note</p>

If you used or plan to use `scripts/annotate_subtasks.py` to manually annotate the end of each subtask in the source demos, you should use signal names that are consistent with the `--signals` argument that you will pass to that script that give a name to each subtask. Internally, the annotations are stored as subtask termination signals with those names.

</div>

<div class="admonition note">
<p class="admonition-title">Note</p>

You should make sure that the config class you implemented is being imported somewhere in your codebase to make sure it gets registered in the config registry. In the MimicGen codebase, we do this in `mimicgen/configs/__init__.py`.

</div>

Finally, run `scripts/generate_config_templates.py` to generate a config template for this new task. It should appear under `mimicgen/exps/templates/<TYPE>/<NAME>.json`. Ensure that the default settings look correct. These settings can be overridden using config generators ([see this tutorial](https://mimicgen.github.io/docs/tutorials/launching_several.html)). 

### Step 3: Execute Data Generation Workflow

You are now all set to try data generation. You should be able to follow the steps documented in the [Data Generation Workflow Overview](https://mimicgen.github.io/docs/tutorials/getting_started.html#overview-of-typical-data-generation-workflow) with some minor changes.

<div class="admonition note">
<p class="admonition-title">Note</p>

Now that you have followed these steps, you can generate datasets for any other variants of this task, as long as the TaskSpec does not change -- e.g. different object placements, different robot arms, and different object instances. The [Reproducing Experiments](https://mimicgen.github.io/docs/tutorials/reproducing_experiments.html) tutorial provides examples of all three variations.

</div>


## Generating Data for New Simulation Frameworks

<div class="admonition note">
<p class="admonition-title">Note</p>

Before starting, you should ensure that a [robomimic environment wrapper](https://robomimic.github.io/docs/modules/environments.html) exists for the simulation framework you are using. See [this link](https://robomimic.github.io/docs/modules/environments.html#implement-an-environment-wrapper) for guidance on how to create one. The environment metadata in the source hdf5 should point to this environment wrapper.

</div>

In this section, we will show how to apply MimicGen to new simulation frameworks. The key step is to implement an [Environment Interface](https://mimicgen.github.io/docs/modules/env_interfaces.html) base class for the simulation framework. We will use robosuite as a running example in this section.

The `MG_EnvInterface` abstract base class in `env_interfaces/base.py` describes the methods that base subclasses for new simulators must implement. There are five important methods:

```python
    """
    These should be filled out by simulator subclasses (e.g. robosuite).
    """
    @property
    @classmethod
    def INTERFACE_TYPE(self):
        """
        Returns string corresponding to interface type. This is used to group
        all subclasses together in the interface registry (for example, all robosuite
        interfaces) and helps avoid name conflicts.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_robot_eef_pose(self):
        """
        Get current robot end effector pose. Should be the same frame as used by the robot end-effector controller.

        Returns:
            pose (np.array): 4x4 eef pose matrix
        """
        raise NotImplementedError

    @abc.abstractmethod
    def target_pose_to_action(self, target_pose, relative=True):
        """
        Takes a target pose for the end effector controller and returns an action 
        (usually a normalized delta pose action) to try and achieve that target pose. 

        Args:
            target_pose (np.array): 4x4 target eef pose
            relative (bool): if True, use relative pose actions, else absolute pose actions

        Returns:
            action (np.array): action compatible with env.step (minus gripper actuation)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def action_to_target_pose(self, action, relative=True):
        """
        Converts action (compatible with env.step) to a target pose for the end effector controller.
        Inverse of @target_pose_to_action. Usually used to infer a sequence of target controller poses
        from a demonstration trajectory using the recorded actions.

        Args:
            action (np.array): environment action
            relative (bool): if True, use relative pose actions, else absolute pose actions

        Returns:
            target_pose (np.array): 4x4 target eef pose that @action corresponds to
        """
        raise NotImplementedError

    @abc.abstractmethod
    def action_to_gripper_action(self, action):
        """
        Extracts the gripper actuation part of an action (compatible with env.step).

        Args:
            action (np.array): environment action

        Returns:
            gripper_action (np.array): subset of environment action for gripper actuation
        """
        raise NotImplementedError
```

The `INTERFACE_TYPE` method is just used to make sure there are no class name conflicts in the environment interface registry. You should just make sure to choose a name unique to your simulation framework. For robosuite, we simply used `"robosuite"`.

The `get_robot_eef_pose` method returns the current pose of the robot end effector and corresponds to the same frame used by the end effector controller in the simulation environment. In robosuite, the Operational Space Controller uses a specific MuJoCo site, so we return its pose:

```python
    def get_robot_eef_pose(self):
        """
        Get current robot end effector pose. Should be the same frame as used by the robot end-effector controller.

        Returns:
            pose (np.array): 4x4 eef pose matrix
        """

        # OSC control frame is a MuJoCo site - just retrieve its current pose
        return self.get_object_pose(
            obj_name=self.env.robots[0].controller.eef_name, 
            obj_type="site",
        )
```

The next two methods (`target_pose_to_action`, `action_to_target_pose`) are used to translate between simulator actions (e.g. those given to `env.step`) and absolute target poses for the end effector controller. This typically consists of simple scaling factors and transformations between different rotation conventions (as described in Appendix N.1 of the MimicGen paper):

```python
    def target_pose_to_action(self, target_pose, relative=True):
        """
        Takes a target pose for the end effector controller and returns an action 
        (usually a normalized delta pose action) to try and achieve that target pose. 

        Args:
            target_pose (np.array): 4x4 target eef pose
            relative (bool): if True, use relative pose actions, else absolute pose actions

        Returns:
            action (np.array): action compatible with env.step (minus gripper actuation)
        """

        # version check for robosuite - must be v1.2+, so that we're using the correct controller convention
        assert (robosuite.__version__.split(".")[0] == "1")
        assert (robosuite.__version__.split(".")[1] >= "2")

        # target position and rotation
        target_pos, target_rot = PoseUtils.unmake_pose(target_pose)

        # current position and rotation
        curr_pose = self.get_robot_eef_pose()
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        # get maximum position and rotation action bounds
        max_dpos = self.env.robots[0].controller.output_max[0]
        max_drot = self.env.robots[0].controller.output_max[3]

        if relative:
            # normalized delta position action
            delta_position = target_pos - curr_pos
            delta_position = np.clip(delta_position / max_dpos, -1., 1.)

            # normalized delta rotation action
            delta_rot_mat = target_rot.dot(curr_rot.T)
            delta_quat = T.mat2quat(delta_rot_mat)
            delta_rotation = T.quat2axisangle(delta_quat)
            delta_rotation = np.clip(delta_rotation / max_drot, -1., 1.)
            return np.concatenate([delta_position, delta_rotation])

        # absolute position and rotation action
        target_quat = T.mat2quat(target_rot)
        abs_rotation = T.quat2axisangle(target_quat)
        return np.concatenate([target_pos, abs_rotation])

    def action_to_target_pose(self, action, relative=True):
        """
        Converts action (compatible with env.step) to a target pose for the end effector controller.
        Inverse of @target_pose_to_action. Usually used to infer a sequence of target controller poses
        from a demonstration trajectory using the recorded actions.

        Args:
            action (np.array): environment action
            relative (bool): if True, use relative pose actions, else absolute pose actions

        Returns:
            target_pose (np.array): 4x4 target eef pose that @action corresponds to
        """

        # version check for robosuite - must be v1.2+, so that we're using the correct controller convention
        assert (robosuite.__version__.split(".")[0] == "1")
        assert (robosuite.__version__.split(".")[1] >= "2")

        if (not relative):
            # convert absolute action to absolute pose
            target_pos = action[:3]
            target_quat = T.axisangle2quat(action[3:6])
            target_rot = T.quat2mat(target_quat)
        else:
            # get maximum position and rotation action bounds
            max_dpos = self.env.robots[0].controller.output_max[0]
            max_drot = self.env.robots[0].controller.output_max[3]

            # unscale actions
            delta_position = action[:3] * max_dpos
            delta_rotation = action[3:6] * max_drot

            # current position and rotation
            curr_pose = self.get_robot_eef_pose()
            curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

            # get pose target
            target_pos = curr_pos + delta_position
            delta_quat = T.axisangle2quat(delta_rotation)
            delta_rot_mat = T.quat2mat(delta_quat)
            target_rot = delta_rot_mat.dot(curr_rot)

        target_pose = PoseUtils.make_pose(target_pos, target_rot)
        return target_pose
```

Finally, the `action_to_gripper_action` extracts the part of the simulator action that corresponds to gripper actuation:

```python
    def action_to_gripper_action(self, action):
        """
        Extracts the gripper actuation part of an action (compatible with env.step).

        Args:
            action (np.array): environment action

        Returns:
            gripper_action (np.array): subset of environment action for gripper actuation
        """

        # last dimension is gripper action
        return action[-1:]
```

Finally, you can follow the instructions in the [Generating Data for New Tasks](https://mimicgen.github.io/docs/tutorials/datagen_custom.html#generating-data-for-new-tasks) section to setup data generation for specific tasks in this simulation framework.
