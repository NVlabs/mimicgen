# Installation

## Requirements

- Mac OS X or Linux machine
- Python >= 3.6 (recommended 3.8.0)
- [conda](https://www.anaconda.com/products/individual) 
  - [virtualenv](https://virtualenv.pypa.io/en/latest/) is also an acceptable alternative, but we assume you have conda installed in our examples below

## Install MimicGen

We recommend installing the repo into a new conda environment (it is called `mimicgen` in the example below):

```sh
conda create -n mimicgen python=3.8
conda activate mimicgen
```

You can install most of the dependencies by cloning the repository and then installing from source:

```sh
cd <PATH_TO_YOUR_INSTALL_DIRECTORY>
git clone https://github.com/NVlabs/mimicgen.git
cd mimicgen
pip install -e .
```

However, there are some additional dependencies that we list below.

### Additional Dependencies

Most of these additional dependencies are best installed from source.

#### robosuite

<div class="admonition note">
<p class="admonition-title">Note</p>

[robosuite](https://robosuite.ai/) is an optional dependency that is only needed if running the examples provided with this repository. The MimicGen source code does not rely on robosuite and can be used with other simulation frameworks.

</div>

```sh
$ cd <PATH_TO_YOUR_INSTALL_DIRECTORY>
$ git clone https://github.com/ARISE-Initiative/robosuite.git
$ cd robosuite
$ git checkout b9d8d3de5e3dfd1724f4a0e6555246c460407daa
$ pip install -e .
```

For more detailed instructions, see the [robosuite installation page](https://robosuite.ai/docs/installation.html).

<div class="admonition note">
<p class="admonition-title">Note</p>

The git checkout command corresponds to the commit we used for testing our policy learning results. However, you should also be able to use the `v1.4.1` branch of robosuite.

</div>

<div class="admonition warning">
<p class="admonition-title">Warning</p>

The current codebase does not support robosuite `v1.5+`. Supporting this would require changing some details due to the changed controller conventions in the new versions.

</div>

#### robomimic

[robomimic](https://robomimic.github.io/) is a required dependency that provides a standardized dataset format, wrappers around simulation environments, and policy learning utilities.

```sh
$ cd <PATH_TO_YOUR_INSTALL_DIRECTORY>
$ git clone https://github.com/ARISE-Initiative/robomimic.git
$ cd robomimic
$ git checkout d0b37cf214bd24fb590d182edb6384333f67b661
$ pip install -e .
```

For more detailed instructions, see the [robomimic installation page](https://robomimic.github.io/docs/introduction/installation.html).

<div class="admonition note">
<p class="admonition-title">Note</p>

The git checkout command corresponds to the commit we used for testing our policy learning results. In general the `master` branch (`v0.3+`) should be fine, as long as it is after the above commit.

</div>

#### robosuite_task_zoo

<div class="admonition note">
<p class="admonition-title">Note</p>

[robosuite_task_zoo](https://github.com/ARISE-Initiative/robosuite-task-zoo) is an optional dependency that is only needed if running the Kitchen and Hammer Cleanup environments and datasets provided with this repository.

</div>

<div class="admonition warning">
<p class="admonition-title">Warning</p>

We recommend removing the dependencies in the `setup.py` file (the `install_requires` list) before installation, as it uses deprecated dependencies (such as mujoco-py).

</div>

```sh
$ cd <PATH_TO_YOUR_INSTALL_DIRECTORY>
$ git clone https://github.com/ARISE-Initiative/robosuite-task-zoo
$ cd robosuite-task-zoo
$ git checkout 74eab7f88214c21ca1ae8617c2b2f8d19718a9ed
# NOTE: should remove dependencies in setup.py in the "install_requires" before the last step
$ pip install -e .
```

#### mujoco

If using robosuite, **please downgrade MuJoCo to 2.3.2**:

```sh
$ pip install mujoco==2.3.2
```

<div class="admonition warning">
<p class="admonition-title">Warning</p>

This MuJoCo version (`2.3.2`) can be important -- in our testing, we found that other versions of MuJoCo could be problematic, especially for the Sawyer arm datasets (e.g. `2.3.5` causes problems with rendering and `2.3.7` changes the dynamics of the robot arm significantly from the collected datasets). More modern versions of MuJoCo (e.g. `3.0`+) might be fine.

</div>

#### pygame

If you plan on using our subtask annotation interface (`scripts/annotate_subtasks.py`) you should also install pygame with `pip install pygame`. See the [Subtask Termination Signals](https://mimicgen.github.io/docs/tutorials/subtask_termination_signals.html) page for more information.

## Test Your Installation

The following script can be used to try random actions in one of our custom robosuite tasks.
```sh
$ cd mimicgen/scripts
$ python demo_random_action.py
```

<div class="admonition note">
<p class="admonition-title">Note</p>

To test data generation please move on to the [Getting Started](https://mimicgen.github.io/docs/tutorials/getting_started.html) tutorial.

</div>

## Install Documentation Dependencies

If you plan to contribute to the repository and add new features, you must install the additional requirements required to build the documentation locally:

```sh
$ pip install -r requirements-docs.txt
```

You can test generating the documentation and viewing it locally in a web browser:
```sh
$ cd <PATH_TO_MIMICGEN_INSTALL_DIR>/docs
$ make clean
$ make apidoc
$ make html
$ cp -r images _build/html/
```

There should be a generated `_build` folder - navigate to `_build/html/` and open `index.html` in a web browser to view the documentation.
