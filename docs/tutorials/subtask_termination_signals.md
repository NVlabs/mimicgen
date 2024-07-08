# Subtask Termination Signals

## What are these signals?

MimicGen expects subtask termination signals to be present in each episode of the source demonstrations. Each signal is a flat numpy array with binary entries (e.g. 0 or 1). 

The [get_source_info.py script](https://mimicgen.github.io/docs/tutorials/debugging_datagen.html#get-source-dataset-information) can be used to print the signals present in a given source demonstration. For an episode (e.g. `demo_0`) they are expected to be at `f["data/demo_0/datagen_info/subtask_term_signals"]`. Under this hdf5 group, there will be one or more hdf5 datasets with the name of the subtask termination signal, and the corresponding flat numpy array.

## How are they used?

These signals are read from the source dataset by the `parse_source_dataset` function (see `mimicgen/utils/file_utils.py`) during data generation and used to split each source demonstration into contiguous subtask segments. Each subtask corresponds to a specific subtask termination signal, with the exception of the final subtask, which ends at the end of the source demonstration. This mapping between the subtask and the corresponding signal is specified through the [TaskSpec](https://mimicgen.github.io/docs/modules/task_spec.html) object that comes from the MimicGen config json.

The end of each subtask is inferred from the first 0 to 1 transition in the corresponding signal. For example, let us consider the source demonstrations for the robosuite StackThree task. There are 4 subtasks with corresponding subtask termination signals:

```
1. (signal: grasp_1) grasping cubeA (motion relative to cubeA)
2. (signal: place_1) placing cubeA on cubeB (motion relative to cubeB)
3. (signal: grasp_2) grasping cubeC (motion relative to cubeC)
4. (signal: None) placing cubeC on cubeA (motion relative to cubeA)
```

For the first source demonstration, the first 0 to 1 transition for grasp_1 is 50, for place_1 is 94, and for grasp_2 is 148. This splits the first source demonstration into four subtask segments with start and end indices as follows:

```
1. (signal: grasp_1) [0, 50]
2. (signal: place_1) [50, 94]
3. (signal: grasp_2) [94, 148]
4. (signal: None) [148, 210]
```

These source subtask segments are subsequently transformed and stitched together through linear interpolation to generate data for new scenes.

However, MimicGen also supports randomization of subtask boundaries -- this is where the `subtask_term_offset_range` parameter of the `TaskSpec` becomes relevant (see [TaskSpec](https://mimicgen.github.io/docs/modules/task_spec.html)). At the start of each data generation attempt, the subtask boundaries (indices 50, 94, and 148 above) can be randomized with an additive offset uniformly sampled in the given offset range bounds. 

The offset parameter can also be used to ensure that the end of each subtask happens at least N timesteps after the first 0 to 1 transition in the corresponding signal. For example, grasp_1 detects a successful grasp, but perhaps you would like the end of the subtask to be a little after grasping (e.g. when the lift begins). An easy way to do this is to specify an offset_range like (5, 10), so that the true subtask boundary will always occur 5 to 10 timesteps after the 0 to 1 transition.

## How are they added to the source data?

There are two ways that subtask termination signals can be added to a source dataset. The first is through the `prepare_src_dataset.py` script -- this will use the task-specific [Environment Interface](https://mimicgen.github.io/docs/modules/env_interfaces.html) (specifically, the `get_subtask_term_signals` method that is used to populate [DatagenInfo](https://mimicgen.github.io/docs/modules/datagen.html#datagen-info) objects) to get subtask termination signals for each timestep in each source demonstration. This relies on some heuristics per task. An example of this process can be found in the [Data Generation for Custom Environments](https://mimicgen.github.io/docs/tutorials/datagen_custom.html) tutorial.

The second is to use manual human annotations for the end of each subtask in each source demonstration -- this can be done by using `scripts/annotate_subtasks.py`. 

<div class="admonition note">
<p class="admonition-title">Note</p>

This script requires the `pygame` module to be installed (it will not be installed by default when installing the Mimicgen repo).

</div>

The script plays back demonstrations (using visual observations and the pygame renderer) in order to allow a user to annotate portions of the demonstrations. This is useful to annotate the end of each object-centric subtask in each source demonstration used by MimicGen, as an alternative to implementing subtask termination signals directly in the simulation environment. Some example invocations of the script:

```sh
# specify the sequence of signals that should be annotated and the dataset images to render on-screen
$ python annotate_subtasks.py --dataset /path/to/demo.hdf5 --signals grasp_1 insert_1 grasp_2 \
    --render_image_names agentview_image robot0_eye_in_hand_image

# limit annotation to first 2 demos
$ python annotate_subtasks.py --dataset /path/to/demo.hdf5 --signals grasp_1 insert_1 grasp_2 \
    --render_image_names agentview_image robot0_eye_in_hand_image --n 2

# limit annotation to demo 2 and 3
$ python annotate_subtasks.py --dataset /path/to/demo.hdf5 --signals grasp_1 insert_1 grasp_2 \
    --render_image_names agentview_image robot0_eye_in_hand_image --n 2 --start 1

# scale up dataset images when rendering to screen by factor of 10
$ python annotate_subtasks.py --dataset /path/to/demo.hdf5 --signals grasp_1 insert_1 grasp_2 \
    --render_image_names agentview_image robot0_eye_in_hand_image --image_scale 10
```

<div class="admonition note">
<p class="admonition-title">Note</p>

We provide some utilities to debug your choice of subtask termination signals and offsets -- see the [Debugging Data Generation](https://mimicgen.github.io/docs/tutorials/debugging_datagen.html) tutorial for more information.

</div>
