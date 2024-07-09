# Codebase Overview

## Codebase Structure

We outline some important folders and files below.

- `mimicgen/scripts`: utility scripts
  - `generate_dataset.py`: main script for data generation
- `mimicgen/exps/templates`: collection of data generation config json templates for each task
- `mimicgen/configs`: implementation of data generation config classes
  - `config.py`: base config class
  - `task_spec.py`: [TaskSpec](https://mimicgen.github.io/docs/modules/task_spec.html) object for specifying sequence of subtasks for each task
  - `robosuite.py`: robosuite-specific config classes
- `mimicgen/env_interfaces`: implementation of [Environment Interface](https://mimicgen.github.io/docs/modules/env_interfaces.html) classes that help simulation environments provide datagen info during data generation
- `mimicgen/datagen`: implementation of core [Data Generation](https://mimicgen.github.io/docs/modules/datagen.html) classes
  - `data_generator.py`: [DataGenerator](https://mimicgen.github.io/docs/modules/datagen.html#data-generator) class used to generate new trajectories
  - `datagen_info.py`: [DatagenInfo](https://mimicgen.github.io/docs/modules/datagen.html#datagen-info) class to group information from the sim environment needed during data generation
  - `selection_strategy.py`: [SelectionStrategy](https://mimicgen.github.io/docs/modules/datagen.html#selection-strategy) classes that contain different heuristics for selecting source demos during each data generation trial
  - `waypoint.py`: collection of [Waypoint](https://mimicgen.github.io/docs/modules/datagen.html#waypoint) classes to help end effector controllers execute waypoint targets and waypoint sequences 
- `mimicgen/envs` and `mimicgen/models`: files containing collection of robosuite simulation environments and assets released with this project
- `mimicgen/utils`: collection of utility functions and classes
- `docs`: files related to documentation

## Important Modules

We provide some more guidance on some important modules and how they relate to one another.

MimicGen starts with a handful of source demonstrations and generates new demonstrations automatically. MimicGen treats each task as a sequence of object-centric subtasks, and attempts to generate trajectories one subtask at a time. MimicGen must parse source demonstrations into contiguous subtask segments -- it uses [Subtask Termination Signals](https://mimicgen.github.io/docs/tutorials/subtask_termination_signals.html) to do this. It also requires object poses at the start of each subtask, both in the source demonstrations and in the current scene during data generation. Information on object poses, subtask termination signals, and other information needed at data generation time is collected into [DatagenInfo](https://mimicgen.github.io/docs/modules/datagen.html#datagen-info) objects, which are read from the source demonstrations, and also read from the current scene. This information is provided through [Environment Interface](https://mimicgen.github.io/docs/modules/env_interfaces.html) classes which connect underlying simulation environments to DatagenInfo objects. 

Data generation is carried out by the [DataGenerator](https://mimicgen.github.io/docs/modules/datagen.html#data-generator) class. Each data generation attempt requires choosing one or more subtask segments from the source demonstrations to transform -- this is carried out by a [SelectionStrategy](https://mimicgen.github.io/docs/modules/datagen.html#selection-strategy) instance. The transformation consists of keeping track of a collection of end effector target poses for a controller to execute -- this is managed by [Waypoint](https://mimicgen.github.io/docs/modules/datagen.html#waypoint) classes. 

The sequence of object-centric subtasks and other important data generation settings for each data generation run are communicated to MimicGen through the [TaskSpec object](https://mimicgen.github.io/docs/modules/task_spec.html), which is read as part of the MimicGen config.
