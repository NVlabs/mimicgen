# Data Generation Classes

This section discusses the key classes related to data generation.

## Data Generator

The `DataGenerator` class (`datagen/data_generator.py`) is responsible for generating new demonstration trajectories. First, the internal `_load_dataset` method is used to parse the source dataset (using [Subtask Termination Signals](https://mimicgen.github.io/docs/tutorials/subtask_termination_signals.html)) into source subtask segments. Each segment is a sequence of [DatagenInfo](https://mimicgen.github.io/docs/modules/datagen.html#datagen-info) objects. Then, the `generate` method is called repeatedly (by the main script `scripts/generate_dataset.py`) to keep attempting to generate new trajectories using the source subtask segments. During each new attempt, the `select_source_demo` method is used to employ a [SelectionStrategy](https://mimicgen.github.io/docs/modules/datagen.html#selection-strategy) to pick a reference source subtask segment to transform. [WaypointTrajectory](https://mimicgen.github.io/docs/modules/datagen.html#waypoint) objects are used to transform and compose subtask segments together.

## Datagen Info

DatagenInfo objects keep track of important information used during data generation. These objects are added to source demonstrations with the `prepare_src_dataset.py` script, or provided directly by an [Environment Interface](https://mimicgen.github.io/docs/modules/env_interfaces.html) object.

The structure of the object is below:

```python
class DatagenInfo(object):
    """
    Structure of information needed from an environment for data generation. To allow for
    flexibility, not all information must be present.
    """
    def __init__(
        self,
        eef_pose=None,
        object_poses=None,
        subtask_term_signals=None,
        target_pose=None,
        gripper_action=None,
    ):
        """
        Args:
            eef_pose (np.array or None): robot end effector poses of shape [..., 4, 4]
            object_poses (dict or None): dictionary mapping object name to object poses
                of shape [..., 4, 4]
            subtask_term_signals (dict or None): dictionary mapping subtask name to a binary 
                indicator (0 or 1) on whether subtask has been completed. Each value in the
                dictionary could be an int, float, or np.array of shape [..., 1].
            target_pose (np.array or None): target end effector poses of shape [..., 4, 4]
            gripper_action (np.array or None): gripper actions of shape [..., D] where D
                is the dimension of the gripper actuation action for the robot arm
        """
```

## Selection Strategy

<div class="admonition note">
<p class="admonition-title">Note</p>

See Appendix N.3 in the MimicGen paper for a more thorough explanation of source subtask segment selection and some further intuition on when to use different settings.

</div>

Each data generation attempt requires choosing one or more subtask segments from the source demonstrations to transform -- this is carried out by a SelectionStrategy instance:

```python
@six.add_metaclass(MG_SelectionStrategyMeta)
class MG_SelectionStrategy(object):
    """
    Defines methods and functions for selection strategies to implement.
    """
    def __init__(self):
        pass

    @property
    @classmethod
    def NAME(self):
        """
        This name (str) will be used to register the selection strategy class in the global
        registry.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def select_source_demo(
        self,
        eef_pose,
        object_pose,
        src_subtask_datagen_infos,
    ):
        """
        Selects source demonstration index using the current robot pose, relevant object pose
        for the current subtask, and relevant information from the source demonstrations for the
        current subtask.

        Args:
            eef_pose (np.array): current 4x4 eef pose
            object_pose (np.array): current 4x4 object pose, for the object in this subtask
            src_subtask_datagen_infos (list): DatagenInfo instance for the relevant subtask segment
                in the source demonstrations

        Returns:
            source_demo_ind (int): index of source demonstration - indicates which source subtask segment to use
        """
        raise NotImplementedError
```

Every SelectionStrategy class must subclass this base class and implement the `NAME` and `select_source_demo` methods. The `NAME` field is used to register the SelectionStrategy class into the global registry, and `select_source_demo` implements the heuristic for selecting a source demonstration index.

Each data generation config json specifies how source segment selection should be done during data generation. First, `config.experiment.generation.select_src_per_subtask` determines whether to select a different source demonstration for each subtask during data generation, or keep the same source demonstration as the one used for the first subtask. This corresponds to the `per-subtask` parameter described in the "Selection Frequency" paragraph of Appendix N.3 in the paper.

The specific task config (`config.task.task_spec`), which corresponds to the [Task Spec](https://mimicgen.github.io/docs/modules/task_spec.html) object used in data generation, also specifies the selection strategy to use for each subtask via the `selection_strategy` parameter and the `selection_strategy_kwargs` parameter. The `selection_strategy` parameter corresponds to the `NAME` for the SelectionStrategy class, and the `selection_strategy_kwargs` correspond to any additional kwargs to specify when invoking the `select_source_demo` method. 

<div class="admonition note">
<p class="admonition-title">Note</p>

Note that if `config.experiment.generation.select_src_per_subtask` is False, only the first subtask's selection strategy matters, since the selected source demonstration will be used for the remainder of the data generation attempt.

</div>

As an example, the `NearestNeighborObjectStrategy` (see implementation below) can be specified by passing `nearest_neighbor_object` for the `selection_strategy` parameter and you can use the `selection_strategy_kwargs` parameter to specify a dictionary containing values for the `pos_weight`, `rot_weight`, and `nn_k` parameters.

```python
class NearestNeighborObjectStrategy(MG_SelectionStrategy):
    """
    Pick source demonstration to be the one with the closest object pose to the object 
    in the current scene.
    """

    # name for registering this class into registry
    NAME = "nearest_neighbor_object"

    def select_source_demo(
        self,
        eef_pose,
        object_pose,
        src_subtask_datagen_infos,
        pos_weight=1.,
        rot_weight=1.,
        nn_k=3,
    ):
        """
        Selects source demonstration index using the current robot pose, relevant object pose
        for the current subtask, and relevant information from the source demonstrations for the
        current subtask.

        Args:
            eef_pose (np.array): current 4x4 eef pose
            object_pose (np.array): current 4x4 object pose, for the object in this subtask
            src_subtask_datagen_infos (list): DatagenInfo instance for the relevant subtask segment
                in the source demonstrations
            pos_weight (float): weight on position for minimizing pose distance
            rot_weight (float): weight on rotation for minimizing pose distance
            nn_k (int): pick source demo index uniformly at randomly from the top @nn_k nearest neighbors

        Returns:
            source_demo_ind (int): index of source demonstration - indicates which source subtask segment to use
        """
```

## Waypoint

### Waypoint Class Variants

MimicGen uses a collection of convenience classes to represent waypoints and trajectories (`datagen/waypoint.py`).

The `Waypoint` class represents a single 6-DoF target pose and the gripper action for that timestep:

```python
class Waypoint(object):
    """
    Represents a single desired 6-DoF waypoint, along with corresponding gripper actuation for this point.
    """
    def __init__(self, pose, gripper_action, noise=None):
```

The `WaypointSequence` class represents a sequence of these `Waypoint` objects:

```python
class WaypointSequence(object):
    """
    Represents a sequence of Waypoint objects.
    """
    def __init__(self, sequence=None):
```

It can easily be instantiated from a collection of poses (e.g. `WaypointSequence.from_poses`):

```python
    @classmethod
    def from_poses(cls, poses, gripper_actions, action_noise):
        """
        Instantiate a WaypointSequence object given a sequence of poses, 
        gripper actions, and action noise.

        Args:
            poses (np.array): sequence of pose matrices of shape (T, 4, 4)
            gripper_actions (np.array): sequence of gripper actions
                that should be applied at each timestep of shape (T, D).
            action_noise (float or np.array): sequence of action noise
                magnitudes that should be applied at each timestep. If a 
                single float is provided, the noise magnitude will be
                constant over the trajectory.
        """
```

Finally, the `WaypointTrajectory` class is a sequence of the `WaypointSequence` objects, and is a convenient way to represent 6-DoF trajectories and execute them:

```python
class WaypointTrajectory(object):
    """
    A sequence of WaypointSequence objects that corresponds to a full 6-DoF trajectory.
    """
```

`WaypointSequence` objects can be added directly to a `WaypointTrajectory` object:

```python
    def add_waypoint_sequence(self, sequence):
        """
        Directly append sequence to list (no interpolation).

        Args:
            sequence (WaypointSequence instance): sequence to add
        """
```

Interpolation segments can also be added easily using this helper method:

```python
    def add_waypoint_sequence_for_target_pose(
        self,
        pose,
        gripper_action,
        num_steps,
        skip_interpolation=False,
        action_noise=0.,
    ):
        """
        Adds a new waypoint sequence corresponding to a desired target pose. A new WaypointSequence
        will be constructed consisting of @num_steps intermediate Waypoint objects. These can either
        be constructed with linear interpolation from the last waypoint (default) or be a
        constant set of target poses (set @skip_interpolation to True).

        Args:
            pose (np.array): 4x4 target pose

            gripper_action (np.array): value for gripper action

            num_steps (int): number of action steps when trying to reach this waypoint. Will
                add intermediate linearly interpolated points between the last pose on this trajectory
                and the target pose, so that the total number of steps is @num_steps.

            skip_interpolation (bool): if True, keep the target pose fixed and repeat it @num_steps
                times instead of using linearly interpolated targets.

            action_noise (float): scale of random gaussian noise to add during action execution (e.g.
                when @execute is called)
        """
```

The `merge` method is a thin wrapper around the above method, easily allowing for linear interpolation between two `WaypointTrajectory` objects:

```python
    def merge(
        self,
        other,
        num_steps_interp=None,
        num_steps_fixed=None,
        action_noise=0.,
    ):
        """
        Merge this trajectory with another (@other).

        Args:
            other (WaypointTrajectory object): the other trajectory to merge into this one

            num_steps_interp (int or None): if not None, add a waypoint sequence that interpolates
                between the end of the current trajectory and the start of @other

            num_steps_fixed (int or None): if not None, add a waypoint sequence that has constant 
                target poses corresponding to the first target pose in @other

            action_noise (float): noise to use during the interpolation segment
        """
```

Finally, the `execute` method makes it easy to execute the waypoint sequences in the simulation environment:

```python
    def execute(
        self, 
        env,
        env_interface, 
        render=False, 
        video_writer=None, 
        video_skip=5, 
        camera_names=None,
    ):
        """
        Main function to execute the trajectory. Will use env_interface.target_pose_to_action to
        convert each target pose at each waypoint to an action command, and pass that along to
        env.step.

        Args:
            env (robomimic EnvBase instance): environment to use for executing trajectory
            env_interface (MG_EnvInterface instance): environment interface for executing trajectory
            render (bool): if True, render on-screen
            video_writer (imageio writer): video writer
            video_skip (int): determines rate at which environment frames are written to video
            camera_names (list): determines which camera(s) are used for rendering. Pass more than
                one to output a video with multiple camera views concatenated horizontally.

        Returns:
            results (dict): dictionary with the following items for the executed trajectory:
                states (list): simulator state at each timestep
                observations (list): observation dictionary at each timestep
                datagen_infos (list): datagen_info at each timestep
                actions (list): action executed at each timestep
                success (bool): whether the trajectory successfully solved the task or not
        """
```

### Waypoint Class Usage during Data Generation

Each data generation attempt consists of executing a particular sequence of waypoints for each subtask. A `WaypointTrajectory` object is constructed and executed for each subtask in the `generate` method of the `DataGenerator`. 

For a given subtask, a `WaypointTrajectory` object is initialized with a single pose (usually the current robot end effector pose, or the last target pose from the previous subtask execution attempt). Next, a reference source subtask segment is selected, and then transformed using the `transform_source_data_segment_using_object_pose` method from `utils/pose_utils.py`. It is then merged into the trajectory object with linear interpolation using the `merge` method. Finally, the `execute` method is used to carry out the subtask. This process repeats for each subtask
