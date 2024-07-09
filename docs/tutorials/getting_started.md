# Getting Started

<div class="admonition note">
<p class="admonition-title">Note</p>

This section helps users get started with data generation. If you would just like to download our existing datasets and use them with policy learning methods please see the [Reproducing Experiments](https://mimicgen.github.io/docs/tutorials/reproducing_experiments.html) tutorial for a guide, or the [Datasets](https://mimicgen.github.io/docs/datasets/mimicgen_corl_2023.html) page to get details on the datasets.

</div>


## Quick Data Generation Run

Let's run a quick data generation example.

Before starting, make sure you are at the base repo path:
```sh
$ cd {/path/to/mimicgen}
```

### Step 1: Prepare source human dataset

MimicGen requires a handful of human demonstrations to get started.

Download the source demonstrations for the Square task below:
```sh
$ python mimicgen/scripts/download_datasets.py --dataset_type source --tasks square
```

This is a basic robomimic dataset collected via teleoperation in robosuite (see [here](https://robomimic.github.io/docs/datasets/robosuite.html)). We need to add in extra information to the hdf5 to make it compatible with MimicGen:
```sh
$ python mimicgen/scripts/prepare_src_dataset.py \
--dataset datasets/source/square.hdf5 \
--env_interface MG_Square \
--env_interface_type robosuite
```

The `--env_interface` and `--env_interface_type` arguments allow the script to find the correct [Environment Interface](https://mimicgen.github.io/docs/modules/env_interfaces.html) class to extract [DatagenInfo objects](https://mimicgen.github.io/docs/modules/datagen.html#datagen-info) at each timestep. In general, each task needs to have an environment interface class to tell MimicGen how to retrieve object poses and other information needed during data generation.

### Step 2: Prepare data generation config

Each data generation run requires a config json (similar to robomimic [Configs](https://robomimic.github.io/docs/modules/configs.html)) that allows us to configure different settings. Template configs for each task are at `mimicgen/exps/templates` and are auto-generated (with `scripts/generate_config_templates.py`). The repository has easy ways to modify these templates to generate new config jsons. 

For now, we will use a script to produce experiment configs consistent with the MimicGen paper. Open `scripts/generate_core_configs.py` and set `NUM_TRAJ = 10` and `GUARANTEE = False` -- this means we will attempt to generate 10 new trajectories. 

<div class="admonition warning">
<p class="admonition-title">Warning</p>

If you do not edit `scripts/generate_core_configs.py` the default settings will run data generation until 1000 success trajectories have been collected. This is why it is important to set `NUM_TRAJ = 10` and `GUARANTEE = False` for a quick run. Alternatively, pass the `--debug` flag to the command in Step 3, which will run an even smaller data generation run.

</div>

Next, run the script:

```sh
$ python mimicgen/scripts/generate_core_configs.py
```

It generates a set of configs (and prints their paths) and also prints lines that correspond to data generation runs for each config.

### Step 3: Run data generation and view outputs

Next, we run data generation on the Square D1 task (this will take a couple minutes):
```sh
$ python mimicgen/scripts/generate_dataset.py \
--config /tmp/core_configs/demo_src_square_task_D1.json \
--auto-remove-exp
```

<div class="admonition note">
<p class="admonition-title">Note</p>

If you run into a `RuntimeError: No ffmpeg exe could be found.` at the end of the script, this means rendering the dataset to video failed. We found that a simple `conda install ffmpeg` fixed the problem on our end (as documented on the [troubleshooting page](https://mimicgen.github.io/docs/miscellaneous/troubleshooting.html)).

</div>

By default, the data generation folder can be found at `/tmp/core_datasets/square/demo_src_square_task_D1`. The contents of this folder are as follows:
```
demo.hdf5                                       # generated hdf5 containing successful demonstrations
demo_failed.hdf5.                               # generated hdf5 containing failed demonstrations, up to a certain limit
important_stats.json                            # json file summarizing success rate and other statistics
log.txt                                         # terminal output
logs/                                           # longer experiment runs will regularly sync progress jsons to this folder
mg_config.json                                  # config used for this experiment
playback_demo_src_square_task_D1.mp4            # video that shows successful demonstrations, up to a certain limit
playback_demo_src_square_task_D1_failed.mp4.    # video that shows failed demonstrations, up to a certain limit
```

<div class="admonition note">
<p class="admonition-title">Note</p>

The generated `demo.hdf5` file is fully compatible with robomimic, which makes it easy to inspect the data (see [here](https://robomimic.github.io/docs/tutorials/dataset_contents.html)) or launch training jobs (we have a helper script at `scripts/generate_core_training_configs.py`).

</div>

Next, we outline the typical MimicGen workflow.


## Overview of Typical Data Generation Workflow

A typical application of MimicGen consists of the following steps.


### Step 1: Collect source demonstrations

Collect human demonstrations for a task of interest. Typically, this is done using a teleoperation pipeline. For example, [this](https://robosuite.ai/docs/algorithms/demonstrations.html) is how robosuite demonstrations can be collected. Make sure you end up with an hdf5 [compatible with robomimic](https://robomimic.github.io/docs/datasets/mimicgen_corl_2023.html#dataset-structure) -- this typically involves a postprocessing script (for example [this one](https://robomimic.github.io/docs/datasets/robosuite.html#converting-robosuite-hdf5-datasets) for robosuite).

<div class="admonition note">
<p class="admonition-title">Note</p>

You should ensure that a [robomimic environment wrapper](https://robomimic.github.io/docs/modules/environments.html) exists for the simulation framework you are using. See [this link](https://robomimic.github.io/docs/modules/environments.html#implement-an-environment-wrapper) for guidance on how to create one. The environment metadata in the source hdf5 should point to this environment wrapper.

</div>

### Step 2: Prepare source demonstrations with additional annotations

Next, information must be added to the source hdf5 to make it compatible with MimicGen. This is typically done with `scripts/prepare_src_dataset.py` just like we did above. However, this requires an [Environment Interface](https://mimicgen.github.io/docs/modules/env_interfaces.html) class to be implemented for your simulation framework (usually a base class) and your task (a subclass of the base class). 

These classes typically specify how to translate between environment actions and target poses for the end effector controller in the environment (see the MimicGen paper for more details on why this is needed). Furthermore, for each task, they provide a dictionary that maps object name to object pose -- recall that MimicGen requires observing an object pose at the start of each object-centric subtask. The structure of the extracted information can be found [here](https://mimicgen.github.io/docs/modules/datagen.html#datagen-info). Finally, the class can optionally provide [subtask termination signals](https://mimicgen.github.io/docs/tutorials/subtask_termination_signals.html) that provide heuristics for splitting source demonstrations into subtask segments. As an example, the `MG_Square` environment interface senses when the nut has been grasped and provides this heuristic in the `grasp` subtask termination signal. 

Instead of using heuristics from the environment interface class, the source demonstrations can be manually segmented into subtasks using the annotation interface in `scripts/annotate_subtasks.py`. This step should be performed after running `scripts/prepare_src_dataset.py`.

<div class="admonition note">
<p class="admonition-title">Note</p>

See the [Data Generation for Custom Environments](https://mimicgen.github.io/docs/tutorials/datagen_custom.html) tutorial for a more comprehensive description of implementing environment interfaces for new simulators and new tasks.

</div>

### Step 3: Run data generation

Next, you can set up your MimicGen config and launch data generation. See the [Launching Several Data Generation Runs](https://mimicgen.github.io/docs/tutorials/launching_several.html) tutorial to see how data generation configs can be generated with ease. Once ready, run `scripts/generate_dataset.py`.

<div class="admonition note">
<p class="admonition-title">Note</p>

Data generation does not necessarily need to run on the exact same task. As described in the paper, you can run it on tasks with possibly different reset distributions, different robot arms, or different object instances. The [Reproducing Experiments](https://mimicgen.github.io/docs/tutorials/reproducing_experiments.html) tutorial provides examples of all three variations.

</div>

### Step 4: Run policy learning on generated data

You can now run any policy learning algorithm on the generated data to train an agent. A common choice is to run Behavioral Cloning. The generated data is compatible with [robomimic](https://robomimic.github.io/), which is an easy way to train an agent on generated datasets, and compare the performance of different learning methods. The [Reproducing Experiments](https://mimicgen.github.io/docs/tutorials/reproducing_experiments.html) tutorial shows examples of how to train agents on the generated data.
