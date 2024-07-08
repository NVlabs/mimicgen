# Debugging Data Generation

We provide some useful suggestions for debugging data generation runs.

## Source Demo Validation

### Get Source Dataset Information

You can use the `get_source_info.py` script to validate whether the source demonstrations you are using have the expected [DatagenInfo](https://mimicgen.github.io/docs/modules/datagen.html#datagen-info) structure, and are using the correct [Environment Interface](https://mimicgen.github.io/docs/modules/env_interfaces.html):

```sh
$ python mimicgen/scripts/get_source_info.py --dataset datasets/source/square.hdf5
```

It will print out information that looks like the following. This is a good way to also validate the object poses and subtask termination signals present in the file:

```
Environment Interface: MG_Square
Environment Interface Type: robosuite

Structure of datagen_info in episode demo_0:
  eef_pose: shape (127, 4, 4)
  gripper_action: shape (127, 1)
  object_poses:
    square_nut: shape (127, 4, 4)
    square_peg: shape (127, 4, 4)
  subtask_term_signals:
    grasp: shape (127,)
  target_pose: shape (127, 4, 4)
```

### Visualizing Subtasks in Source Dataset

You can visualize each subtask segment in a source demonstration using the `visualize_subtasks.py` script.

The script needs to be aware of the order of the subtask signals as well as the maximum termination offsets being used (see the [Subtask Termination Signals](https://mimicgen.github.io/docs/tutorials/subtask_termination_signals.html)) page for more information on offsets) -- this can be specified by providing a config json (`--config`) or by providing the sequence of signals (`--signals`) and offsets (`--offsets`) for all except the last subtask. The end of each subtask is defined as the timestep of the first 0 to 1 transition in the corresponding signal added to the maximum offset value.

The script supports either on-screen rendering (`--render`) or off-screen rendering to a video (`--video_path`). If using on-screen rendering, the script will pause after every subtask segment. If using off-screen rendering, the video alternates between no borders and red borders to show each subtask segment.

```sh
# render on-screen
$ python visualize_subtasks.py --dataset /path/to/demo.hdf5 --config /path/to/config.json --render

# render to video
$ python visualize_subtasks.py --dataset /path/to/demo.hdf5 --config /path/to/config.json --video_path /path/to/video.mp4

# specify subtask information manually instead of using a config
$ python visualize_subtasks.py --dataset /path/to/demo.hdf5 --signals grasp_1 insert_1 grasp_2 --offsets 10 10 10 --render
```

The script is useful to tune the [Subtask Termination Signals](https://mimicgen.github.io/docs/tutorials/subtask_termination_signals.html) specified through the environment interface (the `get_subtask_term_signals` method) or the manual annotations provided through `scripts/annotate_subtasks.py` have defined that subtask segments properly, as well as the offsets you are using in the data generation config.

<div class="admonition warning">
<p class="admonition-title">Warning</p>

When you change the `get_subtask_term_signals` method, you should re-run the `prepare_src_dataset.py` script on the source data to re-write the subtask termination signals to the dataset.

</div>

## Visualization during Data Generation

The main data generation script (`scripts/generate_dataset.py`) also has some useful features for debugging, including on-screen visualization (`--render`), off-screen rendering to video (`--video_path`), running a quick run for debugging (`--debug`), and pausing after each subtask execution (`--pause_subtask`):

```sh
# run normal data generation
$ python generate_dataset.py --config /path/to/config.json

# render all data generation attempts on-screen
$ python generate_dataset.py --config /path/to/config.json --render

# render all data generation attempts to a video
$ python generate_dataset.py --config /path/to/config.json --video_path /path/to/video.mp4

# run a quick debug run
$ python generate_dataset.py --config /path/to/config.json --debug

# pause after every subtask to debug data generation
$ python generate_dataset.py --config /path/to/config.json --render --pause_subtask
```
