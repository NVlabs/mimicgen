# Environment Interfaces

Environment interface objects allow simulation environments to provide MimicGen with [DatagenInfo](https://mimicgen.github.io/docs/modules/datagen.html#datagen-info) class instances, which is a collection of information that MimicGen needs during data generation. 

These environment interface objects are used by `scripts/prepare_src_dataset.py` to add these DatagenInfo instances to the source datasets, along with metadata that stores which environment interface is to be used with this source dataset. They are also used during data generation to provide DatagenInfo instances that correspond to the current timestep of the environment being used for generation (via the `get_datagen_info` method). The environment interface objects are also used to go back and forth between environment actions (used by `env.step`) and target poses for the end effector controller (as described in MimicGen Appendix N.1).

Every simulation framework should implement a base environment interface class that subclasses the `MG_EnvInterface` abstract base class in `env_interfaces/base.py`. For example, the `RobosuiteInterface` class in `env_interfaces/robosuite.py` is the base environment interface class for robosuite simulation environments.

<div class="admonition note">
<p class="admonition-title">Note</p>

The [Generating Data for New Simulation Frameworks](https://mimicgen.github.io/docs/tutorials/datagen_custom.html#generating-data-for-new-simulation-frameworks) tutorial provides a concrete example of how to implement a base environment interface class for a new simulation environment.

</div>

Every simulation task should also implement a task-specific environment interface class that subclasses the corresponding base environment interface class for that simulation framework.

<div class="admonition note">
<p class="admonition-title">Note</p>

The [Generating Data for New Tasks](https://mimicgen.github.io/docs/tutorials/datagen_custom.html#generating-data-for-new-tasks) tutorial provides a concrete example of how to implement a task-specific environment interface class for new tasks.

</div>
